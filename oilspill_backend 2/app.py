# ==============================================================================
# OILSPILL INTELLIGENCE — DETECTION API (DeepLabV3+ ResNet-50)
# Hugging Face Space edition: always-on REST API for the React frontend.
# ==============================================================================
import base64
import io
import os
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import json
import tempfile

import cv2
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from pyproj import Geod, Transformer
from rasterio import features
from scipy.ndimage import uniform_filter
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

import attribution as attr

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_NAME = "best_oil_deeplabv3plus.pth"
# 1) file uploaded next to app.py in this Space, or 2) downloaded from a (private)
#    model repo set in the Space settings: OIL_MODEL_REPO=<user>/<repo> (+ HF_TOKEN secret)
CHECKPOINT_PATH = os.environ.get("OIL_CHECKPOINT", os.path.join(HERE, CHECKPOINT_NAME))
MODEL_REPO = os.environ.get("OIL_MODEL_REPO")

# Optional demo scenes for POST /api/detect {caseId}: put GeoTIFFs in scenes/
CASE_SCENES = {
    cid: os.path.join(HERE, "scenes", fname)
    for cid, fname in {"OS-2026-001": "00249.tif"}.items()
    if os.path.exists(os.path.join(HERE, "scenes", fname))
}
TILE = 512            # training patch size
STRIDE = int(os.environ.get("OIL_STRIDE", "384"))  # tile overlap — 384 = the setting used on Colab
THRESHOLD = 0.5       # same as the validation loop in the notebook
MIN_BLOB_PX = 30      # drop specks smaller than this (matches the ">= 30 px = real oil" rule)
DEFAULT_PIXEL_M = 10.0  # Sentinel-1 IW GRD pixel spacing, used when the file has no georeference

# Validation scorecard of the best checkpoint (from the training run)
MODEL_METRICS = {
    "architecture": "DeepLabV3+",
    "encoder": "ResNet-50",
    "inputChannels": ["VV", "VH", "ΔPol"],
    "loss": "Focal Tversky (α=0.3, β=0.7, γ=1.33)",
    "epochs": 30,
    "realOilIoU": 62.58,
    "f1": 77.70,
    "recall": 87.84,
    "precision": 69.66,
    "lookalikeFPR": 12.43,
    "threshold": THRESHOLD,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
GEOD = Geod(ellps="WGS84")


# ------------------------------------------------------------------------------
# MODEL — identical architecture to the training cell
# ------------------------------------------------------------------------------
MODEL: Optional[torch.nn.Module] = None
MODEL_ERROR: Optional[str] = None


def resolve_checkpoint() -> str:
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    if MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(MODEL_REPO, CHECKPOINT_NAME, token=os.environ.get("HF_TOKEN"))
    raise FileNotFoundError(
        f"{CHECKPOINT_NAME} not found. Upload it to this Space (next to app.py) "
        "or set OIL_MODEL_REPO in the Space settings.")


def load_model() -> torch.nn.Module:
    path = resolve_checkpoint()
    m = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                          in_channels=3, classes=1, activation=None)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state)
    print(f"✅ Loaded checkpoint: {path}")
    return m.to(DEVICE).eval()


def get_model() -> torch.nn.Module:
    global MODEL, MODEL_ERROR
    if MODEL is None:
        try:
            MODEL = load_model()
            MODEL_ERROR = None
        except Exception as e:  # keep the API up and report the problem
            MODEL_ERROR = str(e)
            print(f"❌ Model load failed: {e}")
            raise HTTPException(503, f"Model not loaded: {e}")
    return MODEL


# ------------------------------------------------------------------------------
# PRE-PROCESSING — identical to RobustSSDSARDataset.__getitem__
# ------------------------------------------------------------------------------
def refined_lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    img = np.nan_to_num(img, nan=-35.0, posinf=0.0, neginf=-35.0)
    img_mean = uniform_filter(img, (window_size, window_size))
    img_sqr_mean = uniform_filter(img ** 2, (window_size, window_size))
    img_var = np.maximum(0.0, img_sqr_mean - img_mean ** 2)
    overall_var = np.var(img)
    weights = np.clip(img_var / (img_var + overall_var + 1e-8), 0.0, 1.0)
    return img_mean + weights * (img - img_mean)


def to_3ch(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    vv_c = refined_lee_filter(vv, 5)
    vh_c = refined_lee_filter(vh, 5)
    vv_n = np.clip((vv_c + 35.0) / 35.0, 0.0, 1.0)
    vh_n = np.clip((vh_c + 35.0) / 35.0, 0.0, 1.0)
    dp = np.clip(vv_n - vh_n, 0.0, 1.0)
    return np.stack([vv_n, vh_n, dp], axis=0).astype(np.float32)


def split_bands(arr: np.ndarray):
    """arr: (bands, H, W) or (H, W), values in dB. Returns vv, vh (dB)."""
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        vv = arr
        vh = vv - 6.5
    else:
        vv = arr[0]
        vh = arr[1] if arr.shape[0] > 1 else vv - 6.5
    # If the file is linear sigma0 (not dB), convert — training data was dB
    finite = vv[np.isfinite(vv)]
    if finite.size and np.nanmedian(finite) > 0:
        vv = 10.0 * np.log10(np.clip(vv, 1e-6, None))
        vh = 10.0 * np.log10(np.clip(vh, 1e-6, None))
    return vv, vh


# ------------------------------------------------------------------------------
# INFERENCE — sliding 512x512 window with overlap averaging
# ------------------------------------------------------------------------------
def _positions(n: int):
    if n <= TILE:
        return [0]
    pos = list(range(0, n - TILE + 1, STRIDE))
    if pos[-1] != n - TILE:
        pos.append(n - TILE)
    return pos


@torch.no_grad()
def predict_prob(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    model = get_model()
    H, W = vv.shape
    ph, pw = max(0, TILE - H), max(0, TILE - W)
    if ph or pw:  # pad small scenes with "calm sea floor" value
        vv = np.pad(vv, ((0, ph), (0, pw)), constant_values=-35.0)
        vh = np.pad(vh, ((0, ph), (0, pw)), constant_values=-35.0)
    h, w = vv.shape
    prob = np.zeros((h, w), np.float32)
    hits = np.zeros((h, w), np.float32)
    for y in _positions(h):
        for x in _positions(w):
            t = to_3ch(vv[y:y + TILE, x:x + TILE], vh[y:y + TILE, x:x + TILE])
            logit = model(torch.from_numpy(t)[None].to(DEVICE))
            prob[y:y + TILE, x:x + TILE] += torch.sigmoid(logit)[0, 0].cpu().numpy()
            hits[y:y + TILE, x:x + TILE] += 1.0
    return (prob / np.maximum(hits, 1.0))[:H, :W]


def clean_mask(prob: np.ndarray) -> np.ndarray:
    mask = (prob > THRESHOLD).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_PX:
            mask[lab == i] = 0
    return mask


# ------------------------------------------------------------------------------
# GEOMETRY — polygons, area, MRR, PCA axis (in lon/lat when georeferenced)
# ------------------------------------------------------------------------------
def slick_geometry(mask: np.ndarray, transform, crs) -> dict:
    georef = crs is not None and transform is not None and not transform.is_identity
    polys = [shape(g) for g, v in features.shapes(mask, mask=mask.astype(bool),
                                                  transform=transform if georef else rasterio.Affine.identity())
             if v == 1]
    if not polys:
        return {"detected": False, "areaKm2": 0.0, "parts": 0, "pixelCount": 0,
                "geojson": {"type": "FeatureCollection", "features": []}}

    geom = unary_union(polys)
    if georef and not crs.is_geographic:
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        geom_ll = shp_transform(to_ll, geom)
    else:
        geom_ll = geom

    if georef:
        a, per = GEOD.geometry_area_perimeter(geom_ll)
        area_km2, perim_km = abs(a) / 1e6, per / 1000.0
    else:
        area_km2 = int(mask.sum()) * (DEFAULT_PIXEL_M ** 2) / 1e6
        perim_km = geom.length * DEFAULT_PIXEL_M / 1000.0

    # Minimum rotated rectangle (length/width in km) and PCA axis on the pixel mask
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    px_m = DEFAULT_PIXEL_M
    if georef:
        c = geom_ll.centroid
        dx = abs(transform.a); dy = abs(transform.e)
        if crs.is_geographic:  # degrees -> metres at centroid latitude
            _, _, mx = GEOD.inv(c.x, c.y, c.x + dx, c.y)
            _, _, my = GEOD.inv(c.x, c.y, c.x, c.y + dy)
            px_m = (mx + my) / 2.0
        else:
            px_m = (dx + dy) / 2.0
    length_km = max(rw, rh) * px_m / 1000.0
    width_km = min(rw, rh) * px_m / 1000.0

    cov = np.cov((pts - pts.mean(0)).T) if len(pts) > 1 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    axis_deg = float((np.degrees(np.arctan2(vx, -vy)) + 360.0) % 180.0)  # bearing from north, 0–180

    c = geom_ll.centroid
    minx, miny, maxx, maxy = geom_ll.bounds
    parts = len(geom_ll.geoms) if geom_ll.geom_type == "MultiPolygon" else 1
    return {
        "detected": True,
        "georeferenced": bool(georef),
        "areaKm2": round(area_km2, 4),
        "perimeterKm": round(perim_km, 3),
        "parts": parts,
        "pixelCount": int(mask.sum()),
        "centroid": {"lon": round(c.x, 6), "lat": round(c.y, 6)} if georef else None,
        "bbox": [minx, miny, maxx, maxy] if georef else None,
        "mrrLengthKm": round(length_km, 3),
        "mrrWidthKm": round(width_km, 3),
        "elongation": round(length_km / width_km, 2) if width_km > 0 else None,
        "pcaAxisDeg": round(axis_deg, 3),
        "geojson": {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "properties": {"class": "oil_slick", "areaKm2": round(area_km2, 4)},
            "geometry": mapping(geom_ll),
        }]},
    }


# ------------------------------------------------------------------------------
# PNG PREVIEWS for the before/after split slider
# ------------------------------------------------------------------------------
def _png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _shrink(a: np.ndarray, max_side=1024, interp=cv2.INTER_AREA):
    h, w = a.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return a if s == 1.0 else cv2.resize(a, (int(w * s), int(h * s)), interpolation=interp)


def previews(vv: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> dict:
    v = np.nan_to_num(vv, nan=-35.0)
    p2, p98 = np.percentile(v, (2, 98))
    gray = (np.clip((v - p2) / (p98 - p2 + 1e-7), 0, 1) * 255).astype(np.uint8)
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask == 1] = (255, 176, 32, 200)             # amber slick, transparent sea
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return {
        "sarPng": _png_b64(_shrink(gray)),
        "maskPng": _png_b64(_shrink(rgba, interp=cv2.INTER_NEAREST)),
        "probabilityPng": _png_b64(_shrink(np.ascontiguousarray(heat))),
    }


# ------------------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------------------
def inspect_input(arr: np.ndarray, name: str) -> dict:
    """Checks the upload looks like calibrated Sentinel-1 backscatter in dB (what the model was trained on)."""
    a = arr if arr.ndim == 3 else arr[None]
    b1 = a[0].astype(np.float64)
    fin = b1[np.isfinite(b1)]
    stats = {"bands": int(a.shape[0]), "dtype": str(arr.dtype)}
    warnings = []
    if fin.size == 0:
        return {**stats, "warnings": ["Band 1 contains no valid pixels."]}
    p1, p50, p99 = np.percentile(fin, [1, 50, 99])
    uniq = np.unique(fin[:: max(1, fin.size // 200000)])
    stats.update({"p1": round(float(p1), 3), "median": round(float(p50), 3), "p99": round(float(p99), 3),
                  "uniqueValues": int(len(uniq))})
    if len(uniq) <= 3 and set(np.round(uniq).tolist()) <= {0, 1, 255}:
        warnings.append("This looks like a binary MASK (only 0/1 values), not a SAR image. "
                        "Upload the file from the 'images' folder, not 'masks'.")
    elif arr.dtype == np.uint8 or (p1 >= 0 and p99 > 50):
        warnings.append("Values look like an 8-bit / scaled image, not SAR backscatter in dB "
                        "(expected roughly -35 to 0 dB, sea around -25 to -12 dB). Detection may be unreliable.")
    elif p50 > 0:
        warnings.append("Values look linear (sigma0 > 0); they were converted to dB before detection.")
    elif not (-40 <= p50 <= -5):
        warnings.append(f"Median backscatter {p50:.1f} dB is outside the usual sea range (-25 to -12 dB). "
                        "Check the file is calibrated Sentinel-1 VV in dB.")
    if a.shape[0] == 1:
        warnings.append("Only one band found — VH was approximated as VV − 6.5 dB, as in training.")
    return {**stats, "warnings": warnings}


def run_detection(arr: np.ndarray, transform=None, crs=None, scene_name="scene", tags=None) -> dict:
    t0 = time.time()
    input_check = inspect_input(arr, scene_name)
    vv, vh = split_bands(arr)
    # Same fallback as the attribution notebook (Cell 1c): no CRS -> assumed georeference
    georef_source = "file"
    if crs is None or transform is None or transform.is_identity:
        h, w = vv.shape
        hw, hh = w * attr.ASSUMED_PIXEL_DEG / 2, h * attr.ASSUMED_PIXEL_DEG / 2
        transform = rasterio.transform.from_bounds(
            attr.ASSUMED_CENTER_LON - hw, attr.ASSUMED_CENTER_LAT - hh,
            attr.ASSUMED_CENTER_LON + hw, attr.ASSUMED_CENTER_LAT + hh, w, h)
        crs = rasterio.crs.CRS.from_epsg(4326)
        georef_source = "SYNTHETIC_ASSUMED"
    capture, capture_src = attr.parse_t0(tags or {}, scene_name)
    prob = predict_prob(vv, vh)
    mask = clean_mask(prob)
    geometry = slick_geometry(mask, transform, crs)
    georef = crs is not None and transform is not None and not transform.is_identity
    bounds = None
    if georef:
        west, south, east, north = rasterio.transform.array_bounds(vv.shape[0], vv.shape[1], transform)
        if not crs.is_geographic:
            to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            west, south = to_ll.transform(west, south)
            east, north = to_ll.transform(east, north)
        bounds = [[south, west], [north, east]]  # Leaflet imageOverlay order
    return {
        "runId": str(uuid.uuid4()),
        "scene": {"name": scene_name, "width": int(vv.shape[1]), "height": int(vv.shape[0]),
                  "crs": str(crs) if crs else None, "leafletBounds": bounds,
                  "georefSource": georef_source,
                  "captureTime": attr._iso(capture) if capture is not None else None,
                  "captureTimeSource": capture_src},
        "metrics": MODEL_METRICS,
        "geometry": geometry,
        "inference": {
            "meanConfidence": round(float(prob[mask == 1].mean()), 4) if mask.any() else 0.0,
            "maxConfidence": round(float(prob.max()), 4),
            "device": str(DEVICE),
            "runtimeSec": round(time.time() - t0, 2),
        },
        "input": input_check,
        "images": previews(vv, prob, mask),
    }


def read_raster(src_bytes_or_path, filename: str = ""):
    """Returns (arr, transform, crs, tags). Accepts GeoTIFF or .npy (bands,H,W in dB)."""
    if filename.lower().endswith(".npy"):
        data = src_bytes_or_path if isinstance(src_bytes_or_path, bytes) else open(src_bytes_or_path, "rb").read()
        return np.load(io.BytesIO(data)), None, None, {}
    if isinstance(src_bytes_or_path, bytes):
        with rasterio.MemoryFile(src_bytes_or_path) as mf, mf.open() as src:
            return src.read(), src.transform, src.crs, src.tags()
    with rasterio.open(src_bytes_or_path) as src:
        return src.read(), src.transform, src.crs, src.tags()


# ------------------------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------------------------
app = FastAPI(title="OilSpill Intelligence — Detection API", version="1.1")


@app.on_event("startup")
def _warm_up():
    # Load weights at boot so the first user request is not slow
    try:
        get_model()
    except HTTPException:
        pass


@app.get("/")
def root():
    return {"service": "OilSpill Intelligence detection API", "docs": "/docs", "health": "/api/health"}
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class DetectRequest(BaseModel):
    caseId: str


@app.get("/api/health")
def health():
    return {"status": "ok" if MODEL is not None else "error", "device": str(DEVICE),
            "modelLoaded": MODEL is not None, "error": MODEL_ERROR,
            "stride": STRIDE, "cases": list(CASE_SCENES)}


@app.get("/api/model/metrics")
def metrics():
    return MODEL_METRICS


@app.post("/api/detect")
def detect_case(req: DetectRequest):
    path = CASE_SCENES.get(req.caseId)
    if not path:
        raise HTTPException(404, f"Unknown caseId '{req.caseId}'. Known: {list(CASE_SCENES)}")
    if not os.path.exists(path):
        raise HTTPException(500, f"Scene file for {req.caseId} not found on the server: {path}")
    arr, tr, crs, tags = read_raster(path, path)
    return {"caseId": req.caseId, **run_detection(arr, tr, crs, os.path.basename(path), tags)}


@app.post("/api/detect/upload")
def detect_upload(file: UploadFile = File(...), caseId: str = Form("ADHOC")):
    name = file.filename or "upload.tif"
    if not name.lower().endswith((".tif", ".tiff", ".npy")):
        raise HTTPException(400, "Upload a GeoTIFF (.tif/.tiff) or a .npy patch in dB.")
    data = file.file.read()
    try:
        arr, tr, crs, tags = read_raster(data, name)
    except Exception as e:
        raise HTTPException(400, f"Could not read {name}: {e}")
    with MODEL_LOCK:
        return {"caseId": caseId, **run_detection(arr, tr, crs, name, tags)}


# ------------------------------------------------------------------------------
# DETECTION JOBS — CPU inference can exceed browser request timeouts (Safari ~60 s),
# so the frontend starts a job and polls for the result.
# ------------------------------------------------------------------------------
MODEL_LOCK = threading.Lock()
DETECT_JOBS: dict = {}
DETECT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _run_detect_job(job_id, data, name, case_id):
    try:
        DETECT_JOBS[job_id].update(status="running", updatedAt=time.time())
        arr, tr, crs, tags = read_raster(data, name)
        with MODEL_LOCK:
            res = run_detection(arr, tr, crs, name, tags)
        DETECT_JOBS[job_id].update(status="done", result={"caseId": case_id, **res}, updatedAt=time.time())
    except HTTPException as e:
        DETECT_JOBS[job_id].update(status="error", error=str(e.detail), updatedAt=time.time())
    except Exception as e:
        import traceback
        traceback.print_exc()
        DETECT_JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}", updatedAt=time.time())


@app.post("/api/detect/jobs")
async def start_detect_job(file: UploadFile = File(...), caseId: str = Form("ADHOC")):
    name = file.filename or "upload.tif"
    if not name.lower().endswith((".tif", ".tiff", ".npy")):
        raise HTTPException(400, "Upload a GeoTIFF (.tif/.tiff) or a .npy patch in dB.")
    data = await file.read()
    now = time.time()
    for jid in [j for j, v in list(DETECT_JOBS.items()) if now - v["updatedAt"] > 3600]:
        DETECT_JOBS.pop(jid, None)
    job_id = uuid.uuid4().hex
    DETECT_JOBS[job_id] = {"jobId": job_id, "status": "queued", "createdAt": now, "updatedAt": now}
    DETECT_EXECUTOR.submit(_run_detect_job, job_id, data, name, caseId)
    return {"jobId": job_id, "status": "queued"}


@app.get("/api/detect/jobs/{job_id}")
def get_detect_job(job_id: str):
    job = DETECT_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired detection job — upload the image again.")
    out = {k: v for k, v in job.items()}
    out["elapsedSec"] = round(time.time() - job["createdAt"], 1)
    return out


# ------------------------------------------------------------------------------
# ATTRIBUTION JOBS — reverse hindcast + AIS scoring can take minutes (ERA5 queue),
# so it runs in the background and the frontend polls for progress.
# ------------------------------------------------------------------------------
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("ATTR_WORKERS", "2")))
JOB_TTL_S = 3 * 3600


def _set_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw, updatedAt=time.time())


VIDEO_ROOT = os.path.join(tempfile.gettempdir(), "oilspill_videos")


def _run_job(job_id, slick, t0, t0_source, ais_path, overrides):
    try:
        _set_job(job_id, status="running")
        res = attr.run_attribution(
            slick, t0=t0, t0_source=t0_source, ais_path=ais_path, overrides=overrides,
            progress=lambda i, name: _set_job(job_id, stageIndex=i, stage=name),
            video_dir=os.path.join(VIDEO_ROOT, job_id))
        # URLs the frontend can put straight into <video src>
        res["videoUrls"] = {k: f"/api/attribution/{job_id}/video/{k}" for k in res.get("videos", {})}
        _set_job(job_id, status="done", stageIndex=len(attr.STAGES), stage="Complete", result=res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        if ais_path and ais_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(ais_path)
            except OSError:
                pass


@app.post("/api/attribution")
async def start_attribution(
    slick: str = Form(..., description="Slick GeoJSON (FeatureCollection/Feature/geometry) in EPSG:4326"),
    t0: Optional[str] = Form(None, description="Acquisition time UTC, e.g. 2019-10-26T05:53:00Z"),
    hoursBack: Optional[int] = Form(None),
    nParticles: Optional[int] = Form(None),
    videos: Optional[str] = Form(None, description="'both' (default) | 'forward' | 'backward' | 'none'"),
    ais: Optional[UploadFile] = File(None, description="Optional AIS .parquet/.csv"),
):
    try:
        slick_geo = json.loads(slick)
    except Exception:
        raise HTTPException(400, "slick must be valid GeoJSON")
    t0_ts, t0_src = None, None
    if t0:
        try:
            t0_ts, t0_src = attr._to_naive_utc(t0), "user"
        except Exception:
            raise HTTPException(400, f"Could not parse t0 '{t0}'")
    overrides = {}
    if hoursBack:
        overrides["HOURS_BACK"] = max(1, min(int(hoursBack), 72))
    if nParticles:
        overrides["N_PARTICLES"] = max(100, min(int(nParticles), 10000))
    if videos:
        if videos not in ("both", "forward", "backward", "none"):
            raise HTTPException(400, "videos must be both | forward | backward | none")
        overrides["VIDEO_MODE"] = videos
    ais_path = None
    if ais is not None and ais.filename:
        suffix = ".parquet" if ais.filename.lower().endswith(".parquet") else ".csv"
        fd, ais_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(await ais.read())

    # drop finished jobs older than the TTL
    now = time.time()
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if now - v["updatedAt"] > JOB_TTL_S]:
            JOBS.pop(jid, None)
            import shutil
            shutil.rmtree(os.path.join(VIDEO_ROOT, jid), ignore_errors=True)
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"jobId": job_id, "status": "queued", "stageIndex": 0, "stage": "Queued",
                        "totalStages": len(attr.STAGES), "stages": attr.STAGES,
                        "createdAt": now, "updatedAt": now}
    EXECUTOR.submit(_run_job, job_id, slick_geo, t0_ts, t0_src, ais_path, overrides)
    return {"jobId": job_id, "status": "queued", "stages": attr.STAGES}


@app.get("/api/attribution/{job_id}")
def get_attribution(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown or expired job (the server may have restarted) — start it again.")
        return dict(job)


@app.get("/api/attribution/{job_id}/video/{kind}")
def get_attribution_video(job_id: str, kind: str):
    if kind not in ("backward", "forward"):
        raise HTTPException(404, "kind must be backward or forward")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    fname = ((job or {}).get("result") or {}).get("videos", {}).get(kind)
    if not fname:
        raise HTTPException(404, "Video not available for this job (not rendered, still running, or expired).")
    path = os.path.join(VIDEO_ROOT, job_id, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "Video file expired — run the analysis again.")
    return FileResponse(path, media_type="video/mp4" if fname.endswith(".mp4") else "image/gif", filename=fname)
