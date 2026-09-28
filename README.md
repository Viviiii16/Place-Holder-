# OilSpill Intelligence

SAR-based oil-slick detection and culprit attribution for Sentinel-1 scenes.

A DeepLabV3+ (ResNet-50) segmentation model finds oil slicks in Sentinel-1 VV/VH GeoTIFFs. The detected slick polygon is then hindcast backwards in time with a reverse Lagrangian particle model driven by ERA5 winds and CMEMS surface currents, and the resulting origin envelopes are cross-checked against AIS vessel traffic to produce a ranked list of suspect vessels. Both stages are exposed as a FastAPI service that a React/Vite dashboard consumes.

```
Sentinel-1 GeoTIFF ──► DeepLabV3+ ──► slick polygon (GeoJSON) ──► reverse hindcast ──► 95 % KDE origin envelopes
                                                                        (ERA5 + CMEMS)            │
                                                                                                  ▼
                                                     ranked suspects ◄── scoring ◄── AIS interception (DuckDB)
```

## Repository layout

| Path | What it is |
|---|---|
| `Oil spill detection.ipynb` | Colab notebook: downloads the Zenodo Sentinel-1 benchmark, stages shards to local SSD, builds a balanced oil / look-alike cache and trains the DeepLabV3+ model. Also contains sanity-check and Copernicus Data Space query cells. |
| `Oil spill analysis.ipynb` | Colab notebook: the full attribution pipeline (Cells 1–12) — mask ingestion, particle seeding, forcing retrieval, RK4 hindcast, KDE envelopes, AIS screening, scoring, Folium map, exports and simulation videos. |
| `oilspill_backend 2/app.py` | FastAPI service — detection endpoints, background detection and attribution jobs, PNG previews and video delivery. |
| `oilspill_backend 2/attribution.py` | Server-side port of the attribution notebook (Cells 2–8 and 12) with synthetic fallbacks. |
| `oilspill_backend 2/best_oil_deeplabv3plus.pth` | Trained checkpoint (107 MB, stored with **Git LFS**). |
| `oilspill_backend 2/modal_app.py` | One-file deployment to [Modal](https://modal.com) (CPU or T4 GPU). |
| `oilspill_backend 2/Dockerfile`, `deploy_cloudrun.sh` | Container image for Hugging Face Spaces / Google Cloud Run. |
| `oilspill_backend 2/requirements.txt`, `.env.example` | Python dependencies and the environment variables the service reads. |

The React/Vite dashboard that talks to this API lives in a separate repository; it needs only two variables, `VITE_DETECTION_API_URL` and `VITE_ATTRIBUTION_API_URL`, both pointing at the deployed backend URL.

## Detection model

| | |
|---|---|
| Architecture | DeepLabV3+ with a ResNet-50 encoder (`segmentation_models_pytorch`), ImageNet-initialised, 1 output class |
| Input | 3 channels: VV, VH and ΔPol = VV − VH, each in dB, normalised from [−35, 0] dB to [0, 1] after a 5×5 Refined Lee speckle filter |
| Patch size | 512 × 512 |
| Loss | Focal Tversky (α = 0.3, β = 0.7, γ = 1.33) — penalises missed thin trails more than false positives |
| Optimiser | AdamW (lr 1.5e-4, weight decay 1e-3), cosine annealing over 30 epochs, batch 16, flip augmentation |
| Training data | Balanced 50/50 cache of oil tiles and hard-negative look-alike / no-oil tiles from the Zenodo Sentinel-1 benchmark (records [8346860](https://zenodo.org/records/8346860), [8253899](https://zenodo.org/records/8253899), [13761290](https://zenodo.org/records/13761290) — ≈ 97 GB of images across the three parts) |

Validation scorecard of the shipped checkpoint (also served at `GET /api/model/metrics`):

| Real-oil IoU | F1 | Recall | Precision | Look-alike FPR | Threshold |
|---|---|---|---|---|---|
| 62.58 % | 77.70 % | 87.84 % | 69.66 % | 12.43 % | 0.5 |

At inference the scene is tiled with a 512-px sliding window (stride 384, overlap-averaged probabilities), thresholded at 0.5, and connected components smaller than 30 px are discarded. If a file has only one band, VH is approximated as VV − 6.5 dB, as in training; linear σ⁰ inputs are converted to dB automatically. The service reports warnings when an upload does not look like calibrated Sentinel-1 backscatter (e.g. an 8-bit image or a binary mask uploaded by mistake).

## Attribution pipeline

Given the slick polygon and its acquisition time T₀, `attribution.py` runs seven stages and reports progress on each:

1. **Seeding** — 2 500 particles are placed uniformly by area inside the slick using rejection sampling.
2. **Forcing** — ERA5 10 m winds (`reanalysis-era5-single-levels`, via `cdsapi`) and CMEMS surface currents (`cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`, falling back to the multi-year daily reanalysis for older scenes, via `copernicusmarine`) are fetched for a bounding box padded by 1° and a window of T₀ − 25 h to T₀ + 1 h, then wrapped in a trilinear (t, lon, lat) interpolator. Downloads are cached on disk.
3. **Reverse Lagrangian hindcast** — particles are integrated backwards with RK4 at Δt = −600 s for 24 hours using u_drift = u_current + 0.03·u_wind plus a random walk consistent with a diffusivity of 1 m²/s. Hourly snapshots are kept.
4. **KDE envelopes** — for every hindcast hour a Gaussian KDE (Silverman bandwidth) in the local UTM frame gives the 95 % highest-density region; its centroid forms the drift track.
5. **AIS screening** — an AIS dump (`.parquet` / `.csv`) is read with DuckDB so the time-window and bbox predicates are pushed into the scan. Each vessel track is interpolated to the hindcast hours and tested against the envelope of the same hour.
6. **Scoring** — each flagged vessel gets a culprit score (0–100) from four components: spatial (distance to the same-hour origin centroid), temporal (ping quality and hours inside the envelope), course alignment with the slick axis, and a speed/manoeuvre profile, weighted 0.45 / 0.20 / 0.20 / 0.15. The top 10 are returned.
7. **Videos** — backward hindcast and forward re-run animations rendered with Matplotlib (MP4 via ffmpeg, GIF if ffmpeg is unavailable), with the rank-1 suspect overlaid.

If Copernicus credentials are missing or a download fails, the stage substitutes clearly labelled **synthetic** forcing (and demo AIS when no file is provided) so the pipeline still runs end to end — those results are for validating the workflow only, never for real attribution. Every response carries a `warnings` list and a disclaimer to that effect.

## Getting started

### Prerequisites

Python 3.11, Git LFS (for the checkpoint), and optionally ffmpeg for MP4 output. For real forcing data you need a [Copernicus CDS](https://cds.climate.copernicus.eu/profile) personal access token and a free [Copernicus Marine](https://data.marine.copernicus.eu/register) account.

### Clone and install

```bash
git lfs install
git clone https://github.com/Viviiii16/Place-Holder-.git
cd "Place-Holder-/oilspill_backend 2"

python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # macOS: plain `pip install torch torchvision`
pip install -r requirements.txt
```

If you cloned without LFS, `best_oil_deeplabv3plus.pth` will be a 134-byte pointer file — run `git lfs pull` to fetch the real weights.

### Configure

```bash
cp .env.example .env      # then fill in the values
```

| Variable | Required | Purpose |
|---|---|---|
| `CDS_API_KEY` | for real winds | Copernicus CDS personal access token (ERA5) |
| `CMEMS_USERNAME`, `CMEMS_PASSWORD` | for real currents | Copernicus Marine credentials |
| `AIS_PATH` | no | Server-side AIS file used when a request does not upload one |
| `OIL_CHECKPOINT` | no | Path to the `.pth` file (default: next to `app.py`) |
| `OIL_MODEL_REPO`, `HF_TOKEN` | no | Download the checkpoint from a (private) Hugging Face model repo instead |
| `OIL_STRIDE` | no | Sliding-window stride in px (default 384) |
| `ATTR_WORKERS` | no | Concurrent attribution jobs (default 2) |
| `CMEMS_DATASET_IDS` | no | Comma-separated CMEMS dataset IDs to try in order |
| `FORCING_CACHE`, `FORCING_TIMEOUT_S` | no | Forcing download cache directory and timeout (default 240 s) |
| `VIDEO_MODE` | no | `both` (default), `forward`, `backward` or `none` |

Never commit `.env`; it is git-ignored.

### Run

```bash
uvicorn app:app --port 8000
```

Interactive OpenAPI docs are at <http://localhost:8000/docs>. The model is loaded at startup, so the first request is not slow; `GET /api/health` tells you whether the checkpoint loaded.

To serve a demo scene through `POST /api/detect {"caseId": "OS-2026-001"}`, place the GeoTIFF at `scenes/00249.tif` (the mapping is in `CASE_SCENES` in `app.py`).

## API reference

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Status, device, whether the model is loaded, available demo cases |
| `GET /api/model/metrics` | Validation scorecard of the checkpoint |
| `POST /api/detect/upload` (multipart `file`, optional `caseId`) | Synchronous detection on an uploaded GeoTIFF or `.npy` patch in dB |
| `POST /api/detect/jobs` (multipart `file`) → `GET /api/detect/jobs/{jobId}` | Same detection as a background job — use this from browsers, since CPU inference can exceed request timeouts (Safari ≈ 60 s) |
| `POST /api/detect` `{"caseId": ...}` | Detection on a server-side demo scene |
| `POST /api/attribution` (form: `slick` GeoJSON, optional `t0`, `hoursBack` 1–72, `nParticles` 100–10 000, `videos`, file `ais`) | Starts a hindcast + AIS attribution job, returns `jobId` and the stage list |
| `GET /api/attribution/{jobId}` | Job status, current stage index/name, and the result when done |
| `GET /api/attribution/{jobId}/video/{backward\|forward}` | Simulation video (MP4, or GIF without ffmpeg) |

A detection result contains the slick `geometry` (GeoJSON in EPSG:4326, area and perimeter in km, minimum-rotated-rectangle length/width, elongation and PCA axis bearing), `inference` confidence statistics, an `input` block with data-quality warnings, `scene` metadata including the acquisition time parsed from the GeoTIFF tags or filename, and three base64 PNGs (`sarPng`, `maskPng`, `probabilityPng`) for a before/after slider.

An attribution result contains the `config` used, `slick` descriptors, `forcing` sources, hourly `particles` (subsampled), `envelopes` and `driftTrack`, the per-hour `forcingSeries`, an `ais` summary, ranked `suspects` (MMSI, minimum distance to origin, closest hour, SOG/COG, component scores and `culpritScorePct`), their `tracks`, `videoUrls`, and `warnings`.

Example:

```bash
# 1. detect
curl -F file=@scene.tif http://localhost:8000/api/detect/upload > detect.json

# 2. attribute (pass the slick GeoJSON and the acquisition time)
curl -F "slick=$(jq -c .geometry.geojson detect.json)" \
     -F t0=2019-10-26T05:53:00Z \
     -F ais=@ais_region.parquet \
     http://localhost:8000/api/attribution
# → {"jobId": "...", "status": "queued", "stages": [...]}

curl http://localhost:8000/api/attribution/<jobId>
```

Jobs are held in memory: detection jobs expire after 1 h, attribution jobs and their videos after 3 h, and a restart clears them. Run a single instance (or pin polling to one container) so every poll reaches the process that owns the job.

## Deployment

**Docker / Hugging Face Spaces.** The `Dockerfile` installs CPU-only PyTorch and ffmpeg, runs as user 1000 and listens on `$PORT` (7860 by default, 8080 on Cloud Run). The `README.md` inside the backend folder carries the Space front-matter. Set the credentials as Space secrets.

**Modal (recommended, GPU optional).**

```bash
pip install modal && modal setup
# In the Modal dashboard create a secret named "oilspill-secrets" with
# CDS_API_KEY, CMEMS_USERNAME, CMEMS_PASSWORD
modal deploy modal_app.py                  # CPU, 4 cores
OILSPILL_GPU=T4 modal deploy modal_app.py  # T4 GPU
```

The app runs as a single container that stays warm for 10 minutes and scales to zero, which keeps in-memory jobs consistent across polls.

**Google Cloud Run.** From Cloud Shell inside the backend folder, export the three credentials and run `./deploy_cloudrun.sh`. It deploys `oilspill-api` with 2 vCPU / 4 GiB, no CPU throttling and `max-instances 1`, and prints the URL to put into the dashboard's `VITE_*` variables.

## Retraining the model

Open `Oil spill detection.ipynb` in Google Colab with a GPU runtime and a mounted Google Drive. The notebook downloads the Zenodo parts with `aria2` into `MyDrive/SAR_OilSpill_Zenodo`, extracts the `.7z` archives, stages the most oil-rich shards to local SSD as `.img.npy` / `.mask.npy` pairs, builds the balanced train/val cache and trains for 30 epochs, saving the best real-oil-IoU checkpoint to `MyDrive/SAR_OilSpill_Zenodo/checkpoints/best_oil_deeplabv3plus.pth`. Copy that file into the backend folder (or upload it to a Hugging Face model repo and set `OIL_MODEL_REPO`). The pre-processing in `app.py` is byte-for-byte the same as the notebook's dataset class, so a retrained checkpoint drops in without code changes as long as the architecture is unchanged.

## Running the attribution notebook standalone

`Oil spill analysis.ipynb` is the reference implementation of the pipeline and includes an interactive Folium map (slick, hourly envelopes, drift track, suspect tracks) and CSV/GeoJSON exports that the API does not produce. Edit the `CFG` dictionary in Cell 1(b) — paths, credentials and physics parameters live only there — upload `oil_mask.tif` and an AIS file to `/content/data/`, and run Cells 2 → 9 in order. Set `DEMO_MODE=True` to exercise the whole workflow on synthetic data without any credentials.

## Notes and limitations

ERA5 is published with roughly a five-day latency and the CMEMS analysis/forecast product covers only the last two years or so, so very recent or very old scenes may fall back to lower-resolution or synthetic forcing — check the `forcing` block and `warnings` in every result. A GeoTIFF without a CRS is given an assumed georeference (centre 18.30° E, 37.95° N, 0.0002° per pixel) and flagged `georefSource: SYNTHETIC_ASSUMED`; an upload without an acquisition time gets a synthetic T₀. Attribution scores are analytical decision support derived from model correlations and are not evidence of responsibility on their own.

## Acknowledgements

Drift physics follows the leeway and random-walk formulations used in OpenDrift (Dagestad et al., 2018) and Abascal et al. (2012); origin envelopes use Gaussian KDE after Parzen (1962) and Silverman (1986). Training data comes from the Sentinel-1 oil-spill benchmark published on Zenodo; met-ocean forcing from the Copernicus Climate Data Store and Copernicus Marine Service.
