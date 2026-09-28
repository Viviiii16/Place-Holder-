# ==============================================================================
# OILSPILL INTELLIGENCE — REVERSE LAGRANGIAN HINDCAST + AIS CULPRIT ATTRIBUTION
# Server version of oil_spill_culprit_attribution.ipynb (Cells 2–8).
# Input: the slick polygon produced by the detection model (EPSG:4326) + T0.
# ==============================================================================
import datetime as dt
import os
import re
import tempfile
import zipfile
from typing import Callable, Optional

import contourpy
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shp_transform, unary_union

# Local runs: read credentials from a .env file next to this script (if present)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# ------------------------------------------------------------------------------
# CONFIG — same defaults as the notebook's CFG. Credentials come from env vars
# (never hard-code them): CDS_API_KEY, CMEMS_USERNAME, CMEMS_PASSWORD
# ------------------------------------------------------------------------------
DEFAULT_CFG = {
    "CDS_API_KEY": os.environ.get("CDS_API_KEY", ""),
    "CMEMS_USERNAME": os.environ.get("CMEMS_USERNAME", ""),
    "CMEMS_PASSWORD": os.environ.get("CMEMS_PASSWORD", ""),
    # analysis/forecast (≈ last 2 years, hourly) first, then multi-year reanalysis (daily) for older scenes
    "CMEMS_DATASET_IDS": os.environ.get(
        "CMEMS_DATASET_IDS",
        "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m,cmems_mod_glo_phy_my_0.083deg_P1D-m",
    ).split(","),
    "AIS_PATH": os.environ.get("AIS_PATH", ""),   # optional server-side AIS file (.parquet/.csv)
    "HOURS_BACK": 24,
    "DT_SEC": -600,
    "N_PARTICLES": 2500,
    "ALPHA_WIND": 0.03,
    "D_DIFF": 1.0,
    "BBOX_PAD_DEG": 1.0,
    "RANDOM_SEED": 42,
    "KDE_BW": "silverman",
    "KDE_LEVEL": 0.95,
    "KDE_GRID_N": 220,
    "AIS_COLUMNS": {"mmsi": "mmsi", "timestamp": "timestamp",
                    "lat": "latitude", "lon": "longitude", "sog": "sog", "cog": "cog"},
    "AIS_TS_IS_EPOCH_SECONDS": False,
    "AIS_MAX_GAP_SEC": 2 * 3600,
    "AIS_MIN_PINGS": 2,
    "W_SPATIAL": 0.45, "W_TEMPORAL": 0.20, "W_COURSE": 0.20, "W_SPEED": 0.15,
    "TOP_N_SUSPECTS": 10,
    "PARTICLES_PER_HOUR_OUT": 1000,   # particles returned to the browser per hour (subsampled)
    # Cell 12 — simulation videos
    "VIDEO_MODE": os.environ.get("VIDEO_MODE", "both"),   # 'forward' | 'backward' | 'both' | 'none'
    "VIDEO_FPS": 10,
    "VIDEO_FRAME_MIN": 20,        # minutes of simulated time per frame
    "VIDEO_DPI": 110,
    "VIDEO_SHOW_SUSPECT": True,   # overlay the rank-1 vessel
}

STAGES = [
    "Seeding particles",
    "Loading ocean forcing",
    "Reverse Lagrangian simulation",
    "Computing KDE envelope",
    "Checking AIS traffic",
    "Scoring candidate vessels",
    "Rendering simulation videos",
]

R_EARTH = 6_371_000.0
KN2MS = 0.514444

# Fallback georeference used by the notebook when a tif has no CRS (Cell 1c)
ASSUMED_CENTER_LON, ASSUMED_CENTER_LAT = 18.30, 37.95
ASSUMED_PIXEL_DEG = 0.0002


# ------------------------------------------------------------------------------
# T0 parsing (Cell 2) — tags first, then filename. No interactive input on a server.
# ------------------------------------------------------------------------------
_TAG_KEYS = ["ACQUISITION_TIME", "ACQUISITION_DATETIME", "SENSING_TIME", "TIFFTAG_DATETIME",
             "DATETIME", "TIMESTAMP", "T0", "datetime", "timestamp", "acquisition_time"]
_FNAME_PATTERNS = [
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_ -]?(\d{2})[:_-]?(\d{2})[:_-]?(\d{2})",
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_ -]?(\d{2})[:_-]?(\d{2})",
    r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})",
]


def _to_naive_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def parse_t0(tags: dict, filename: str):
    """Returns (Timestamp, source) or (None, None)."""
    for k in _TAG_KEYS:
        if k in (tags or {}):
            raw = str(tags[k]).strip()
            if re.match(r"^\d{4}:\d{2}:\d{2} ", raw):
                raw = raw[:4] + "-" + raw[5:7] + "-" + raw[8:]
            try:
                return _to_naive_utc(raw), f"tag:{k}"
            except Exception:
                pass
    for pat in _FNAME_PATTERNS:
        m = re.search(pat, os.path.basename(filename or ""))
        if m:
            g = [int(x) for x in m.groups()] + [0, 0, 0]
            try:
                t = pd.Timestamp(dt.datetime(*g[:6]))
                if 1990 <= t.year <= 2100:
                    return t, "filename"
            except Exception:
                continue
    return None, None


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def bearing_from_vector(dx, dy) -> float:
    return float(np.degrees(np.arctan2(dx, dy)) % 180.0)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def circ_diff(a, b, period=360.0):
    d = np.abs((np.asarray(a) - np.asarray(b)) % period)
    return np.minimum(d, period - d)


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_placeholder(s) -> bool:
    return (not s) or str(s).startswith("PASTE_")


# ------------------------------------------------------------------------------
# Forcing (Cell 4)
# ------------------------------------------------------------------------------
def synthetic_forcing(bbox, t_start, t_end, kind):
    import xarray as xr
    times = pd.date_range(t_start.floor("h"), t_end.ceil("h"), freq="h")
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + 0.25, 0.25)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + 0.25, 0.25)
    tt, la, lo = np.meshgrid((times - times[0]).total_seconds().values / 3600.0, lats, lons, indexing="ij")
    if kind == "wind":
        ang = np.deg2rad(225 + 20 * np.sin(2 * np.pi * tt / 24))
        spd = 6.0 + 1.5 * np.sin(2 * np.pi * tt / 12) + 0.8 * np.cos(3 * lo)
        u, v = -spd * np.sin(ang), -spd * np.cos(ang)
    else:
        u = 0.25 * np.cos(2 * np.pi * (la - lats.mean()) / 2.0) + 0.05 * np.sin(2 * np.pi * tt / 12.4)
        v = 0.15 * np.sin(2 * np.pi * (lo - lons.mean()) / 2.0) + 0.05 * np.cos(2 * np.pi * tt / 12.4)
    return xr.Dataset({"u": (("time", "latitude", "longitude"), u), "v": (("time", "latitude", "longitude"), v)},
                      coords={"time": times, "latitude": lats, "longitude": lons},
                      attrs={"source": f"SYNTHETIC_{kind}"})


def standardize(ds, u_name, v_name, source):
    ren = {}
    for cand in ["valid_time", "time_counter"]:
        if cand in ds.dims:
            ren[cand] = "time"
    for cand in ["lat", "y", "nav_lat"]:
        if cand in ds.dims:
            ren[cand] = "latitude"
    for cand in ["lon", "x", "nav_lon"]:
        if cand in ds.dims:
            ren[cand] = "longitude"
    ds = ds.rename({**ren, u_name: "u", v_name: "v"})
    for d in ["expver", "number", "depth"]:
        if d in ds.dims:
            ds = ds.isel({d: 0})
    lonv = ds.longitude.values
    if lonv.max() > 180:
        ds = ds.assign_coords(longitude=((lonv + 180) % 360) - 180)
    ds = ds.sortby("longitude").sortby("latitude").sortby("time")
    ds = ds[["u", "v"]].transpose("time", "latitude", "longitude")
    ds.attrs["source"] = source
    return ds


def fetch_era5_wind(bbox, t_start, t_end, out_dir, token):
    import cdsapi
    days = pd.date_range(t_start.floor("D"), t_end.floor("D"), freq="D")
    req = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": sorted({f"{d.year:04d}" for d in days}),
        "month": sorted({f"{d.month:02d}" for d in days}),
        "day": sorted({f"{d.day:02d}" for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]],
    }
    target = os.path.join(out_dir, "era5_wind.nc")
    cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=token, quiet=True).retrieve(
        "reanalysis-era5-single-levels", req, target)
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            z.extractall(out_dir)
            member = [n for n in z.namelist() if n.endswith(".nc")][0]
        target = os.path.join(out_dir, member)
    return target


def fetch_cmems_currents(bbox, t_start, t_end, out_dir, dataset_id, user, pwd):
    import copernicusmarine
    fname = f"cmems_{dataset_id}.nc"
    copernicusmarine.subset(
        dataset_id=dataset_id, variables=["uo", "vo"],
        minimum_longitude=bbox["lon_min"], maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"], maximum_latitude=bbox["lat_max"],
        start_datetime=t_start.strftime("%Y-%m-%dT%H:%M:%S"), end_datetime=t_end.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0, maximum_depth=1,
        output_filename=fname, output_directory=out_dir, disable_progress_bar=True,
        username=user, password=pwd,
    )
    return os.path.join(out_dir, fname)


class ForcingField:
    """Trilinear (t, lat, lon) interpolator; t in seconds relative to T0. Returns m/s."""

    def __init__(self, ds, t_ref, name):
        self.name = name
        self.source = ds.attrs.get("source", "?")
        t = ((ds.time.values - np.datetime64(t_ref)) / np.timedelta64(1, "s")).astype(float)
        lat, lon = ds.latitude.values.astype(float), ds.longitude.values.astype(float)
        if len(t) < 2 or len(lat) < 2 or len(lon) < 2:
            raise ValueError(f"{name}: need ≥2 samples along every axis (t={len(t)}, lat={len(lat)}, lon={len(lon)})")
        u = np.nan_to_num(ds.u.values.astype(float))
        v = np.nan_to_num(ds.v.values.astype(float))
        self.nan_frac = float(np.isnan(ds.u.values).mean())
        self.bounds = {"t": (t.min(), t.max()), "lat": (lat.min(), lat.max()), "lon": (lon.min(), lon.max())}
        self._iu = RegularGridInterpolator((t, lat, lon), u, method="linear", bounds_error=False, fill_value=None)
        self._iv = RegularGridInterpolator((t, lat, lon), v, method="linear", bounds_error=False, fill_value=None)

    def __call__(self, t_sec, lon, lat):
        lon = np.clip(np.asarray(lon, float), *self.bounds["lon"])
        lat = np.clip(np.asarray(lat, float), *self.bounds["lat"])
        tt = np.full_like(lon, np.clip(t_sec, *self.bounds["t"]))
        q = np.column_stack([tt, lat, lon])
        return self._iu(q), self._iv(q)


# Forcing downloads: run ERA5 + CMEMS in parallel, give up after FORCING_TIMEOUT_S (→ synthetic,
# with a warning), and cache finished files so a re-run of the same scene is instant. A download that
# times out keeps running in the background and lands in the cache for the next run.
import shutil as _shutil
import threading as _threading
import time as _time
from concurrent.futures import ThreadPoolExecutor as _Pool, TimeoutError as _FutTimeout

FORCING_CACHE = os.environ.get("FORCING_CACHE", os.path.join(tempfile.gettempdir(), "oilspill_forcing"))
FORCING_TIMEOUT_S = float(os.environ.get("FORCING_TIMEOUT_S", "240"))
_DL_POOL = _Pool(max_workers=4)
_INFLIGHT: dict = {}
_INFLIGHT_LOCK = _threading.Lock()


def _forcing_key(bbox, t_start, t_end):
    b = "_".join(f"{bbox[k]:.2f}" for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    return f"{b}_{t_start:%Y%m%dT%H}_{t_end:%Y%m%dT%H}"


def _cached(name, job):
    """Returns a Future resolving to the cached file path; starts the download once per name."""
    os.makedirs(FORCING_CACHE, exist_ok=True)
    final = os.path.join(FORCING_CACHE, name)
    with _INFLIGHT_LOCK:
        fut = _INFLIGHT.get(name)
        if fut is not None and not (fut.done() and fut.exception() is not None):
            return fut
        if os.path.exists(final):
            fut = _DL_POOL.submit(lambda: final)
        else:
            def run():
                tmp = tempfile.mkdtemp(dir=FORCING_CACHE)
                try:
                    os.replace(job(tmp), final)
                    return final
                finally:
                    _shutil.rmtree(tmp, ignore_errors=True)
            fut = _DL_POOL.submit(run)
        _INFLIGHT[name] = fut
        return fut


def load_forcing(bbox, t0, t_start, t_end, cfg, workdir, warnings, status=None):
    import xarray as xr
    key = _forcing_key(bbox, t_start, t_end)
    timeout = float(cfg.get("FORCING_TIMEOUT_S", FORCING_TIMEOUT_S))

    # ---- start both downloads in parallel
    wind_fut = None
    if _is_placeholder(cfg["CDS_API_KEY"]):
        warnings.append("CDS_API_KEY not configured — SYNTHETIC wind field used (not valid for attribution).")
    else:
        wind_fut = _cached(f"era5_{key}.nc",
                           lambda d: fetch_era5_wind(bbox, t_start, t_end, d, cfg["CDS_API_KEY"]))

    curr_futs = []   # (dataset_id, window_start, window_end, future) in preference order
    if _is_placeholder(cfg["CMEMS_USERNAME"]) or _is_placeholder(cfg["CMEMS_PASSWORD"]):
        warnings.append("CMEMS credentials not configured — SYNTHETIC currents used (not valid for attribution).")
    else:
        for ds_id in [d.strip() for d in cfg["CMEMS_DATASET_IDS"] if d.strip()]:
            # daily products need a wider window to have ≥2 time steps
            ts, te = (t_start.floor("D") - pd.Timedelta(days=1), t_end.ceil("D") + pd.Timedelta(days=1)) \
                if "P1D" in ds_id else (t_start, t_end)
            fut = _cached(f"cmems_{ds_id}_{_forcing_key(bbox, ts, te)}.nc",
                          lambda d, ds_id=ds_id, ts=ts, te=te: fetch_cmems_currents(
                              bbox, ts, te, d, ds_id, cfg["CMEMS_USERNAME"], cfg["CMEMS_PASSWORD"]))
            curr_futs.append((ds_id, ts, te, fut))

    # ---- wait (shared deadline), reporting progress
    t_begin = _time.time()
    deadline = t_begin + timeout
    pending = [f for f in [wind_fut] + [c[3] for c in curr_futs] if f is not None]
    while pending and _time.time() < deadline:
        pending = [f for f in pending if not f.done()]
        if not pending:
            break
        # stop early once ERA5 and one usable CMEMS product are ready
        if (wind_fut is None or wind_fut.done()) and any(c[3].done() and c[3].exception() is None for c in curr_futs):
            break
        if status:
            waiting = []
            if wind_fut is not None and not wind_fut.done():
                waiting.append("ERA5 winds (Copernicus queue)")
            if curr_futs and not any(c[3].done() for c in curr_futs):
                waiting.append("CMEMS currents")
            status(f"Loading ocean forcing — waiting for {' & '.join(waiting) or 'downloads'} "
                   f"({int(_time.time() - t_begin)}s)")
        _time.sleep(2)

    # ---- winds
    wind_ds = None
    if wind_fut is not None:
        if not wind_fut.done():
            warnings.append(f"ERA5 still queued at Copernicus after {int(timeout)} s — SYNTHETIC wind used for "
                            "this run. The download continues in the background: run the analysis again in a "
                            "few minutes to use real ERA5 winds.")
        elif wind_fut.exception() is not None:
            e = wind_fut.exception()
            warnings.append(f"ERA5 download failed ({type(e).__name__}: {e}) — SYNTHETIC wind used.")
        else:
            try:
                wind_ds = standardize(xr.open_dataset(wind_fut.result()), "u10", "v10", "ERA5") \
                    .sel(time=slice(t_start, t_end)).load()
            except Exception as e:
                warnings.append(f"ERA5 file unreadable ({type(e).__name__}: {e}) — SYNTHETIC wind used.")
    if wind_ds is None:
        wind_ds = synthetic_forcing(bbox, t_start, t_end, "wind")

    # ---- currents (first usable product in preference order)
    curr_ds, errs, still_waiting = None, [], False
    for ds_id, ts, te, fut in curr_futs:
        if not fut.done():
            still_waiting = True
            continue
        if fut.exception() is not None:
            e = fut.exception()
            errs.append(f"{ds_id}: {type(e).__name__}: {e}")
            continue
        try:
            ds = standardize(xr.open_dataset(fut.result()), "uo", "vo", f"CMEMS:{ds_id}").sel(time=slice(ts, te)).load()
            if ds.sizes.get("time", 0) >= 2:
                curr_ds = ds
                break
            errs.append(f"{ds_id}: fewer than 2 time steps for this window")
        except Exception as e:
            errs.append(f"{ds_id}: {type(e).__name__}: {e}")
    if curr_futs and curr_ds is None:
        msg = "CMEMS currents not available — SYNTHETIC currents used."
        if still_waiting:
            msg += f" (download still running after {int(timeout)} s; run again in a few minutes)"
        if errs:
            msg += " " + " | ".join(errs)[:400]
        warnings.append(msg)
    if curr_ds is None:
        curr_ds = synthetic_forcing(bbox, t_start, t_end, "current")
    return ForcingField(wind_ds, t0, "wind"), ForcingField(curr_ds, t0, "current")


# ------------------------------------------------------------------------------
# AIS (Cell 7)
# ------------------------------------------------------------------------------
def make_demo_ais(rng, t0, bbox, centroid_by_hour, slick_axis, hours_back, n_vessels=30, culprit_hour=10,
                  ping_every_s=300):
    tgrid = np.arange(-(hours_back + 1) * 3600, 3600 + 1, ping_every_s, dtype=float)
    recs = []
    for i in range(n_vessels):
        mmsi = 200_000_000 + int(rng.integers(1e7, 9e7))
        if i == 0:
            lon_c, lat_c = centroid_by_hour[min(culprit_hour, hours_back)]
            t_c = -culprit_hour * 3600
            brg = (slick_axis + rng.normal(0, 8)) % 360
            sog = rng.uniform(9, 13)
        else:
            lon_c = rng.uniform(bbox["lon_min"], bbox["lon_max"])
            lat_c = rng.uniform(bbox["lat_min"], bbox["lat_max"])
            t_c = rng.uniform(tgrid[0], tgrid[-1])
            brg = rng.uniform(0, 360)
            sog = rng.uniform(5, 18)
        v = sog * KN2MS
        dlat = np.degrees(v * np.cos(np.radians(brg)) / R_EARTH) * (tgrid - t_c)
        dlon = np.degrees(v * np.sin(np.radians(brg)) / (R_EARTH * np.cos(np.radians(lat_c)))) * (tgrid - t_c)
        recs.append(pd.DataFrame({
            "mmsi": mmsi, "ts": t0 + pd.to_timedelta(tgrid, unit="s"),
            "lat": lat_c + dlat + rng.normal(0, 2e-4, len(tgrid)),
            "lon": lon_c + dlon + rng.normal(0, 2e-4, len(tgrid)),
            "sog": sog + rng.normal(0, 0.3, len(tgrid)),
            "cog": (brg + rng.normal(0, 2, len(tgrid))) % 360,
        }))
    df = pd.concat(recs, ignore_index=True)
    df = df[(df.lat.between(bbox["lat_min"], bbox["lat_max"])) & (df.lon.between(bbox["lon_min"], bbox["lon_max"]))]
    return df.sort_values(["mmsi", "ts"]).reset_index(drop=True), int(recs[0].mmsi.iloc[0])


def load_ais_duckdb(path, bbox, t_start, t_end, cfg):
    import duckdb
    cols = cfg["AIS_COLUMNS"]
    reader = f"read_parquet('{path}')" if path.endswith(".parquet") else f"read_csv_auto('{path}', header=true)"
    ts_expr = f'to_timestamp("{cols["timestamp"]}")' if cfg["AIS_TS_IS_EPOCH_SECONDS"] \
        else f'CAST("{cols["timestamp"]}" AS TIMESTAMP)'
    sql = f"""
        WITH src AS (
            SELECT CAST("{cols['mmsi']}" AS BIGINT) AS mmsi, {ts_expr} AS ts,
                   CAST("{cols['lat']}" AS DOUBLE) AS lat, CAST("{cols['lon']}" AS DOUBLE) AS lon,
                   CAST("{cols['sog']}" AS DOUBLE) AS sog, CAST("{cols['cog']}" AS DOUBLE) AS cog
            FROM {reader})
        SELECT * FROM src
        WHERE ts BETWEEN TIMESTAMP '{t_start}' AND TIMESTAMP '{t_end}'
          AND lat BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
          AND lon BETWEEN {bbox['lon_min']} AND {bbox['lon_max']}
          AND mmsi IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL
        ORDER BY mmsi, ts
    """
    con = duckdb.connect()
    df = con.execute(sql).df()
    con.close()
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df


# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------
def run_attribution(slick_geojson: dict, t0: Optional[pd.Timestamp] = None, t0_source: Optional[str] = None,
                    ais_path: Optional[str] = None, overrides: Optional[dict] = None,
                    progress: Optional[Callable[[int, str], None]] = None,
                    video_dir: Optional[str] = None) -> dict:
    cfg = {**DEFAULT_CFG, **(overrides or {})}
    assert abs(cfg["W_SPATIAL"] + cfg["W_TEMPORAL"] + cfg["W_COURSE"] + cfg["W_SPEED"] - 1) < 1e-9
    rng = np.random.default_rng(cfg["RANDOM_SEED"])
    warnings: list = []
    step = progress or (lambda i, s: None)

    # ---------------------------------------------------------------- slick geometry (Cell 2)
    feats = slick_geojson.get("features") if slick_geojson.get("type") == "FeatureCollection" else [slick_geojson]
    geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
    slick_geom = unary_union(geoms).buffer(0)
    if slick_geom.is_empty:
        raise ValueError("Slick geometry is empty — nothing detected to hindcast.")
    if slick_geom.geom_type == "Polygon":
        slick_geom = MultiPolygon([slick_geom])

    if t0 is None:
        t0 = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=10)).floor("h")
        t0_source = "synthetic"
        warnings.append(f"No acquisition time for the scene — SYNTHETIC T0 = {_iso(t0)} "
                        "(provide the real capture time for a valid attribution).")
    t0 = _to_naive_utc(t0)

    c = slick_geom.centroid
    local_crs = utm_crs_for(c.x, c.y)
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    to_wgs = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    slick_local = shp_transform(to_local.transform, slick_geom)

    bpts = np.vstack([np.asarray(p.exterior.coords) for p in slick_local.geoms])
    q = bpts - bpts.mean(0)
    _, svals, vt = np.linalg.svd(q, full_matrices=False)
    ev = svals ** 2 / max(len(q) - 1, 1)
    pca_bearing = bearing_from_vector(*vt[0])
    pca_elong = float(np.sqrt(ev[0] / max(ev[1], 1e-9)))
    mrr = slick_local.minimum_rotated_rectangle
    cc = np.asarray(mrr.exterior.coords)[:4]
    edges = [(cc[i + 1] - cc[i]) for i in range(3)] + [(cc[0] - cc[3])]
    lens = [np.hypot(*v) for v in edges]
    mrr_bearing = bearing_from_vector(*edges[int(np.argmax(lens))])
    slick_axis = pca_bearing

    lon_min, lat_min, lon_max, lat_max = slick_geom.bounds
    pad = cfg["BBOX_PAD_DEG"]
    bbox = {"lat_min": lat_min - pad, "lat_max": lat_max + pad, "lon_min": lon_min - pad, "lon_max": lon_max + pad}
    hours_back = int(cfg["HOURS_BACK"])
    t_start = t0 - pd.Timedelta(hours=hours_back + 1)
    t_end = t0 + pd.Timedelta(hours=1)

    slick_info = {
        "t0": _iso(t0), "t0Source": t0_source,
        "centroidLon": c.x, "centroidLat": c.y, "nParts": len(slick_geom.geoms),
        "areaKm2": slick_local.area / 1e6, "perimeterKm": slick_local.length / 1e3,
        "pcaAxisBearingDeg": pca_bearing, "pcaElongationRatio": pca_elong,
        "mrrLengthKm": max(lens) / 1e3, "mrrWidthKm": min(lens) / 1e3, "mrrBearingDeg": mrr_bearing,
        "localCrs": str(local_crs),
    }

    # ---------------------------------------------------------------- 1. seeding (Cell 3)
    step(1, STAGES[0])
    n = int(cfg["N_PARTICLES"])
    minx, miny, maxx, maxy = slick_geom.bounds
    acc = slick_geom.area / max((maxx - minx) * (maxy - miny), 1e-12)
    pts, it = [], 0
    while sum(len(p) for p in pts) < n and it < 200:
        need = n - sum(len(p) for p in pts)
        batch = int(need / max(acc, 1e-3) * 1.2) + 50
        xs, ys = rng.uniform(minx, maxx, batch), rng.uniform(miny, maxy, batch)
        keep = shapely.contains_xy(slick_geom, xs, ys)
        pts.append(np.column_stack([xs[keep], ys[keep]]))
        it += 1
    particles0 = np.vstack(pts)[:n]
    if len(particles0) < n:
        raise RuntimeError(f"Rejection sampling produced only {len(particles0)}/{n} particles.")

    # ---------------------------------------------------------------- 2. forcing (Cell 4)
    step(2, STAGES[1])
    with tempfile.TemporaryDirectory() as workdir:
        winds, currents = load_forcing(bbox, t0, t_start, t_end, cfg, workdir, warnings,
                                       status=lambda msg: step(2, msg))

    # ---------------------------------------------------------------- 3. RK4 hindcast (Cell 5)
    step(3, STAGES[2])
    DT = float(cfg["DT_SEC"])
    alpha, D = cfg["ALPHA_WIND"], cfg["D_DIFF"]
    steps_per_hour = int(round(3600 / abs(DT)))

    def ms_to_degps(u, v, lat):
        return np.degrees(u / (R_EARTH * np.cos(np.radians(lat)))), np.degrees(v / R_EARTH)

    def drift(t, X):
        uc, vc = currents(t, X[:, 0], X[:, 1])
        uw, vw = winds(t, X[:, 0], X[:, 1])
        dlon, dlat = ms_to_degps(uc + alpha * uw, vc + alpha * vw, X[:, 1])
        return np.column_stack([dlon, dlat])

    def rk4_step(t, X, dt):
        k1 = drift(t, X)
        k2 = drift(t + dt / 2, X + dt / 2 * k1)
        k3 = drift(t + dt / 2, X + dt / 2 * k2)
        k4 = drift(t + dt, X + dt * k3)
        return X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def random_walk_deg(X, dt, r):
        R_ms = np.sqrt(6.0 * D / abs(dt)) * r.uniform(-1.0, 1.0, size=X.shape)
        dlon, dlat = ms_to_degps(R_ms[:, 0] * abs(dt), R_ms[:, 1] * abs(dt), X[:, 1])
        return np.column_stack([dlon, dlat])

    X = particles0.copy()
    history = {0: X.copy()}
    hour_times = {0: t0}
    t = 0.0
    for s in range(1, hours_back * steps_per_hour + 1):
        X = rk4_step(t, X, DT)
        X = X + random_walk_deg(X, DT, rng)
        t += DT
        if s % steps_per_hour == 0:
            k = s // steps_per_hour
            history[k] = X.copy()
            hour_times[k] = t0 + pd.Timedelta(seconds=t)

    # ---------------------------------------------------------------- 4. KDE envelopes (Cell 6)
    step(4, STAGES[3])

    def kde_envelope(P, level, grid_n, bw, kpad=0.20):
        x, y = to_local.transform(P[:, 0], P[:, 1])
        x, y = np.asarray(x), np.asarray(y)
        try:
            kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
            dx, dy = np.ptp(x), np.ptp(y)
            gx = np.linspace(x.min() - kpad * dx, x.max() + kpad * dx, grid_n)
            gy = np.linspace(y.min() - kpad * dy, y.max() + kpad * dy, grid_n)
            GX, GY = np.meshgrid(gx, gy)
            Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            zs = np.sort(Z.ravel())[::-1]
            cdf = np.cumsum(zs) / zs.sum()
            thr = zs[min(np.searchsorted(cdf, level), len(zs) - 1)]
            polys = []
            for ln in contourpy.contour_generator(GX, GY, Z).lines(thr):
                if len(ln) >= 4:
                    pg = Polygon(ln).buffer(0)
                    if not pg.is_empty and pg.area > 0:
                        polys.append(pg)
            env_local = unary_union(polys) if polys else MultiPoint(np.column_stack([x, y])).convex_hull
            method = "kde"
        except Exception as e:
            env_local = MultiPoint(np.column_stack([x, y])).convex_hull.buffer(50)
            method = f"hull_fallback({type(e).__name__})"
        env_wgs = shp_transform(to_wgs.transform, env_local)
        cx, cy = to_wgs.transform(x.mean(), y.mean())
        return env_wgs, (float(cx), float(cy)), env_local.area / 1e6, method

    envelopes = []
    for k in sorted(history):
        env, (cx, cy), area_km2, method = kde_envelope(history[k], cfg["KDE_LEVEL"], cfg["KDE_GRID_N"], cfg["KDE_BW"])
        envelopes.append({"hourBack": k, "timestamp": hour_times[k], "centroidLon": cx, "centroidLat": cy,
                          "areaKm2": area_km2, "method": method, "geometry": env})
    env_by_hour = {e["hourBack"]: e["geometry"] for e in envelopes}

    # Wind / current sampled at the particle-cloud centroid each hour (for the UI forcing card)
    forcing_series = []
    for e in envelopes:
        k = e["hourBack"]
        lon_c, lat_c = np.array([e["centroidLon"]]), np.array([e["centroidLat"]])
        uw, vw = winds(-k * 3600.0, lon_c, lat_c)
        uc, vc = currents(-k * 3600.0, lon_c, lat_c)
        forcing_series.append({"hourBack": k, "windU": float(uw[0]), "windV": float(vw[0]),
                               "currentU": float(uc[0]), "currentV": float(vc[0])})
    centroid_by_hour = {e["hourBack"]: (e["centroidLon"], e["centroidLat"]) for e in envelopes}
    env_df = pd.DataFrame([{k: v for k, v in e.items() if k != "geometry"} for e in envelopes])

    # ---------------------------------------------------------------- 5. AIS screening (Cell 7)
    step(5, STAGES[4])
    planted = None
    ais_file = ais_path or cfg["AIS_PATH"]
    if ais_file and os.path.exists(ais_file):
        ais = load_ais_duckdb(ais_file, bbox, t_start, t_end, cfg)
        ais_source = "file"
    else:
        ais, planted = make_demo_ais(rng, t0, bbox, centroid_by_hour, slick_axis, hours_back)
        ais_source = "DEMO_SYNTHETIC"
        warnings.append("No AIS data supplied — SYNTHETIC demo AIS used (one vessel is planted as a known culprit). "
                        "Rankings are illustrative only.")
    ais["t_s"] = (ais["ts"] - t0).dt.total_seconds()

    HOURS = np.arange(0, hours_back + 1)
    T_K = -HOURS * 3600.0

    def interp_vessel(g):
        g = g.sort_values("t_s")
        ts = g.t_s.values
        if len(ts) < cfg["AIS_MIN_PINGS"]:
            return None
        idx = np.clip(np.searchsorted(ts, T_K), 1, len(ts) - 1)
        gap = ts[idx] - ts[idx - 1]
        ok = (T_K >= ts[0]) & (T_K <= ts[-1]) & (gap <= cfg["AIS_MAX_GAP_SEC"])
        if not ok.any():
            return None
        near = np.where((T_K - ts[idx - 1]) <= (ts[idx] - T_K), idx - 1, idx)
        out = pd.DataFrame({
            "mmsi": g.mmsi.iloc[0], "hour_back": HOURS, "t_s": T_K,
            "lon": np.interp(T_K, ts, g.lon.values), "lat": np.interp(T_K, ts, g.lat.values),
            "sog": g.sog.values[near], "cog": g.cog.values[near], "ping_gap_s": np.abs(T_K - ts[near]),
        })
        return out[ok]

    parts = [d for d in (interp_vessel(g) for _, g in ais.groupby("mmsi")) if d is not None] if len(ais) else []
    ais_summary = {"source": ais_source, "pings": int(len(ais)), "vessels": int(ais.mmsi.nunique()) if len(ais) else 0,
                   "interpolableVessels": 0, "flaggedVessels": 0,
                   "window": {"start": _iso(t_start), "end": _iso(t_end)}, "bbox": bbox}
    suspects, tracks = [], {}
    if parts:
        vh = pd.concat(parts, ignore_index=True)
        vh["timestamp"] = t0 + pd.to_timedelta(vh.t_s, unit="s")
        vh["inside_envelope"] = False
        for k, env in env_by_hour.items():
            m = vh.hour_back == k
            if m.any():
                vh.loc[m, "inside_envelope"] = shapely.contains_xy(env, vh.loc[m, "lon"].values, vh.loc[m, "lat"].values)
        ais_summary["interpolableVessels"] = int(vh.mmsi.nunique())
        ais_summary["flaggedVessels"] = int(vh[vh.inside_envelope].mmsi.nunique())

        # ------------------------------------------------------------ 6. scoring (Cell 8)
        step(6, STAGES[5])
        vh = vh.merge(env_df[["hourBack", "centroidLon", "centroidLat", "areaKm2"]]
                      .rename(columns={"hourBack": "hour_back"}), on="hour_back", how="left")
        vh["dist_km"] = haversine_km(vh.lon, vh.lat, vh.centroidLon, vh.centroidLat)

        rows = []
        for mmsi, g in vh.groupby("mmsi"):
            g = g.sort_values("hour_back")
            i = int(g.dist_km.values.argmin())
            r = g.iloc[i]
            k = int(r.hour_back)
            L = max(1.0, np.sqrt(r.areaKm2 / np.pi))
            S_s = float(np.exp(-r.dist_km / L))
            hours_inside = int(g.inside_envelope.sum())
            S_t = 0.6 * float(np.exp(-r.ping_gap_s / 1800.0)) + 0.4 * min(1.0, hours_inside / 3.0)
            dev = float(circ_diff(r.cog, slick_axis, 180.0))
            S_c = float(1.0 - dev / 90.0)
            med_sog = float(g.sog.median())
            nb = g[(g.hour_back >= k - 1) & (g.hour_back <= k + 1)]
            cog_change = float(circ_diff(nb.cog.values[:-1], nb.cog.values[1:]).max()) if len(nb) > 1 else 0.0
            loiter, speed_dev, turn = r.sog < 1.0, abs(r.sog - med_sog) > 3.0, cog_change > 30.0
            anomalous = bool(loiter or speed_dev or turn)
            S_v = 1.0 if anomalous else (0.6 if 6.0 <= r.sog <= 18.0 else 0.3)
            score = 100 * (cfg["W_SPATIAL"] * S_s + cfg["W_TEMPORAL"] * S_t + cfg["W_COURSE"] * S_c
                           + cfg["W_SPEED"] * S_v)
            flag = (("loiter " if loiter else "") + ("speed_dev " if speed_dev else "")
                    + ("turn>30° " if turn else "")).strip() or "cruising"
            rows.append({
                "mmsi": str(int(mmsi)), "mdoKm": round(float(r.dist_km), 3), "closestHourBack": k,
                "timeClosestApproach": _iso(r.timestamp), "lonClosest": float(r.lon), "latClosest": float(r.lat),
                "sogKn": round(float(r.sog), 1), "cogDeg": round(float(r.cog), 1),
                "cogVsSlickAxisDeg": round(dev, 1), "hoursInsideEnvelope": hours_inside,
                "pingGapS": int(r.ping_gap_s), "manoeuvreFlag": flag,
                "sSpatial": round(S_s, 3), "sTemporal": round(S_t, 3), "sCourse": round(S_c, 3),
                "sSpeed": round(S_v, 3), "culpritScorePct": round(float(score), 1),
            })
        rows.sort(key=lambda d: d["culpritScorePct"], reverse=True)
        for i, d in enumerate(rows, 1):
            d["rank"] = i
            d["isPlantedDemoCulprit"] = planted is not None and d["mmsi"] == str(planted)
        suspects = rows
        top_ids = {int(d["mmsi"]) for d in rows[: cfg["TOP_N_SUSPECTS"]]}
        for mmsi, g in ais[ais.mmsi.isin(top_ids)].groupby("mmsi"):
            g = g.sort_values("ts")
            g = g.iloc[:: max(1, len(g) // 150)]
            tracks[str(int(mmsi))] = [[round(float(a), 5), round(float(b), 5), _iso(t_)]
                                      for a, b, t_ in zip(g.lon, g.lat, g.ts)]
    else:
        step(6, STAGES[5])
        warnings.append("No AIS vessels had interpolable positions in the hindcast window — no suspects ranked.")

    # ---------------------------------------------------------------- 7. simulation videos (Cell 12)
    videos = {}
    mode = str(cfg["VIDEO_MODE"]).lower()
    if video_dir and mode in ("forward", "backward", "both"):
        step(7, STAGES[6])
        try:
            videos = render_simulation_videos(
                cfg, mode, video_dir, particles0, history, hours_back, DT, t0, slick_geom,
                rk4_step, random_walk_deg,
                suspect=(suspects[0]["mmsi"] if suspects else None), ais=ais)
        except Exception as e:
            warnings.append(f"Simulation video rendering failed ({type(e).__name__}: {e}).")

    # ---------------------------------------------------------------- output
    per_hour = int(cfg["PARTICLES_PER_HOUR_OUT"])
    sub = rng.choice(n, size=min(per_hour, n), replace=False)
    return {
        "t0": _iso(t0),
        "t0Source": t0_source,
        "config": {k: cfg[k] for k in ["HOURS_BACK", "DT_SEC", "N_PARTICLES", "ALPHA_WIND", "D_DIFF",
                                       "KDE_LEVEL", "W_SPATIAL", "W_TEMPORAL", "W_COURSE", "W_SPEED"]},
        "slick": {**slick_info, "geojson": mapping(slick_geom)},
        "forcing": {"wind": winds.source, "current": currents.source,
                    "window": {"start": _iso(t_start), "end": _iso(t_end)}, "bbox": bbox},
        "particles": [{"hourBack": k, "timestamp": _iso(hour_times[k]),
                       "positions": np.round(history[k][sub], 5).tolist()} for k in sorted(history)],
        "envelopes": [{**{k: v for k, v in e.items() if k != "geometry"},
                       "timestamp": _iso(e["timestamp"]), "geometry": mapping(e["geometry"])} for e in envelopes],
        "driftTrack": [[e["centroidLon"], e["centroidLat"]] for e in envelopes],
        "forcingSeries": forcing_series,   # m/s at the drift-track centroid, one per hour back
        "ais": ais_summary,
        "suspects": suspects[: cfg["TOP_N_SUSPECTS"]],
        "allSuspectsCount": len(suspects),
        "tracks": tracks,
        "videos": videos,          # {'backward': 'backward_simulation.mp4', 'forward': ...}
        "warnings": warnings,
        "disclaimer": "Attribution scores are analytical decision-support from model correlations and do not "
                      "constitute legal proof of discharge responsibility.",
    }


# ------------------------------------------------------------------------------
# Cell 12 — simulation videos (backward hindcast and forward re-simulation)
# ------------------------------------------------------------------------------
def render_simulation_videos(cfg, mode, out_dir, particles0, history, hours_back, DT, t0, slick_geom,
                             rk4_step, random_walk_deg, suspect=None, ais=None) -> dict:
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    steps_per_frame = max(1, int(round(cfg["VIDEO_FRAME_MIN"] * 60 / abs(DT))))

    def simulate(X0, t_start, t_end, dt, r):
        X, t, frames = X0.copy(), float(t_start), [(float(t_start), X0.copy())]
        n = int(round(abs(t_end - t_start) / abs(dt)))
        for s in range(1, n + 1):
            X = rk4_step(t, X, dt) + random_walk_deg(X, dt, r)
            t += dt
            if s % steps_per_frame == 0 or s == n:
                frames.append((t, X.copy()))
        return frames

    allp = np.vstack(list(history.values()))
    pad = 0.08 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))
    XLIM = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    YLIM = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)

    sus_track = None
    if cfg["VIDEO_SHOW_SUSPECT"] and suspect is not None and ais is not None and len(ais):
        g = ais[ais.mmsi == int(suspect)].sort_values("t_s")
        if len(g):
            sus_track = (int(suspect), g.t_s.values, g.lon.values, g.lat.values)

    def render(frames, title, out_path):
        aspect = 1 / np.cos(np.radians(np.mean(YLIM)))
        fig, ax = plt.subplots(figsize=(6.4, 6.4 * (YLIM[1] - YLIM[0]) / ((XLIM[1] - XLIM[0]) / aspect)))
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect(aspect)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(alpha=0.25)
        for poly in slick_geom.geoms:
            ax.plot(*poly.exterior.xy, color="black", lw=1.4, zorder=3)
        sc = ax.scatter([], [], s=3, c=[], cmap="viridis", vmin=-hours_back, vmax=0, alpha=0.75, zorder=2)
        ship, = ax.plot([], [], marker=(3, 0, 0), ms=11, color="crimson", ls="", zorder=4)
        trail, = ax.plot([], [], color="crimson", lw=1, alpha=0.6, zorder=4)
        stamp = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        handles = [Patch(facecolor="none", edgecolor="black", label="Observed slick @T0")]
        if sus_track:
            handles.append(plt.Line2D([], [], color="crimson", marker=">", ls="", label=f"Suspect MMSI {sus_track[0]}"))
        ax.legend(handles=handles, loc="lower right", fontsize=8)
        fig.suptitle(title, fontsize=12)

        def draw(i):
            t, X = frames[i]
            sc.set_offsets(X); sc.set_array(np.full(len(X), t / 3600))
            stamp.set_text(f"{(t0 + pd.Timedelta(seconds=t)):%Y-%m-%d %H:%M} UTC   (T0 {t / 3600:+.1f} h)")
            if sus_track:
                _, ts, lo, la = sus_track
                if ts[0] <= t <= ts[-1]:
                    ship.set_data([np.interp(t, ts, lo)], [np.interp(t, ts, la)])
                    m = ts <= t
                    trail.set_data(lo[m], la[m])
                else:
                    ship.set_data([], []); trail.set_data([], [])
            return sc, ship, trail, stamp

        anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / cfg["VIDEO_FPS"], blit=False)
        if shutil.which("ffmpeg"):
            anim.save(out_path, writer=animation.FFMpegWriter(
                fps=cfg["VIDEO_FPS"], bitrate=2400,
                extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]), dpi=cfg["VIDEO_DPI"])
        else:
            out_path = out_path.replace(".mp4", ".gif")
            anim.save(out_path, writer=animation.PillowWriter(fps=cfg["VIDEO_FPS"]), dpi=cfg["VIDEO_DPI"] // 2)
        plt.close(fig)
        return os.path.basename(out_path)

    os.makedirs(out_dir, exist_ok=True)
    out = {}
    r = np.random.default_rng(cfg["RANDOM_SEED"] + 1)
    if mode in ("backward", "both"):
        fr = simulate(particles0, 0.0, -hours_back * 3600, DT, r)
        out["backward"] = render(fr, f"Backward hindcast  T0 → T0-{hours_back}h",
                                 os.path.join(out_dir, "backward_simulation.mp4"))
    if mode in ("forward", "both"):
        fr = simulate(history[hours_back], -hours_back * 3600, 0.0, abs(DT), r)
        out["forward"] = render(fr, f"Forward re-simulation  T0-{hours_back}h → T0",
                                os.path.join(out_dir, "forward_simulation.mp4"))
    return out
