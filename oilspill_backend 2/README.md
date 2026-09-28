---
title: OilSpill Intelligence API
emoji: 🛰️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# OilSpill Intelligence — Detection + Attribution API

DeepLabV3+ (ResNet-50) oil-slick segmentation for Sentinel-1 SAR GeoTIFFs (VV/VH in dB).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Status, whether the checkpoint loaded |
| `GET /api/model/metrics` | Validation scorecard |
| `POST /api/detect/upload` (multipart `file`) | Detect on an uploaded GeoTIFF / `.npy` patch |
| `POST /api/detect` `{"caseId": "OS-2026-001"}` | Detect on a demo scene in `scenes/` |
| `POST /api/attribution` (form: `slick` GeoJSON, `t0`, optional `ais` file) | Start reverse hindcast + AIS attribution job → `jobId` |
| `GET /api/attribution/{jobId}` | Job progress (7 stages) and result |
| `GET /api/attribution/{jobId}/video/{backward\|forward}` | Simulation video (MP4, or GIF if ffmpeg is missing) |

## Environment variables (set as secrets on the host — never commit them)

| Variable | Purpose |
|---|---|
| `CDS_API_KEY` | Copernicus CDS personal access token (ERA5 winds) |
| `CMEMS_USERNAME`, `CMEMS_PASSWORD` | Copernicus Marine account (surface currents) |
| `AIS_PATH` | Optional server-side AIS file (.parquet/.csv) used when none is uploaded |

Without these, attribution still runs but uses clearly-flagged SYNTHETIC forcing / demo AIS.

## Run locally

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # (Mac: plain `pip install torch torchvision`)
pip install -r requirements.txt
export CDS_API_KEY=... CMEMS_USERNAME=... CMEMS_PASSWORD=...
uvicorn app:app --port 8000
```

Interactive docs: `/docs`


## Deploy on Modal (GPU, recommended)

```bash
pip install modal
modal setup                      # opens the browser to log in (one time)
# Modal dashboard → Secrets → Create → Custom: name it  oilspill-secrets
#   CDS_API_KEY=...  CMEMS_USERNAME=...  CMEMS_PASSWORD=...
# put best_oil_deeplabv3plus.pth next to modal_app.py, then:
modal deploy modal_app.py        # prints https://<you>--oilspill-intelligence-api.modal.run
```
Use that URL for both `VITE_DETECTION_API_URL` and `VITE_ATTRIBUTION_API_URL`.

## Deploy on Google Cloud Run (CPU alternative)

From Google Cloud Shell in this folder: `export CDS_API_KEY=... CMEMS_USERNAME=... CMEMS_PASSWORD=...` then `./deploy_cloudrun.sh`.
