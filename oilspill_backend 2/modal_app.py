# ==============================================================================
# OilSpill Intelligence — Modal deployment (detection + attribution on a T4 GPU)
#
#   1. pip install modal && modal setup            (one time, logs you in)
#   2. Create the secret "oilspill-secrets" in the Modal dashboard with
#      CDS_API_KEY, CMEMS_USERNAME, CMEMS_PASSWORD
#   3. Put best_oil_deeplabv3plus.pth next to this file
#   4. modal deploy modal_app.py                    → CPU (no payment method needed)
#      OILSPILL_GPU=T4 modal deploy modal_app.py     → T4 GPU (needs a card on the Modal account)
# ==============================================================================
import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
APP_DIR = "/root/oilspill"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch", "torchvision",
        "fastapi", "uvicorn[standard]", "python-multipart",
        "segmentation-models-pytorch>=0.4", "rasterio", "shapely", "pyproj", "scipy",
        "opencv-python-headless", "pillow", "numpy", "huggingface_hub",
        "pandas", "xarray", "netcdf4", "contourpy", "duckdb", "matplotlib",
        "cdsapi>=0.7.2", "copernicusmarine>=2.0", "python-dotenv",
    )
    .env({"OIL_CHECKPOINT": f"{APP_DIR}/best_oil_deeplabv3plus.pth", "PYTHONUNBUFFERED": "1"})
    # local files last, so code changes redeploy in seconds
    .add_local_file(HERE / "app.py", f"{APP_DIR}/app.py")
    .add_local_file(HERE / "attribution.py", f"{APP_DIR}/attribution.py")
    .add_local_file(HERE / "best_oil_deeplabv3plus.pth", f"{APP_DIR}/best_oil_deeplabv3plus.pth")
)

app = modal.App("oilspill-intelligence")

GPU = os.environ.get("OILSPILL_GPU") or None   # e.g. "T4"; None = CPU only


@app.function(
    image=image,
    gpu=GPU,
    cpu=None if GPU else 4.0,          # CPU mode: 4 cores for faster detection
    memory=4096,
    secrets=[modal.Secret.from_name("oilspill-secrets")],
    # One container: detection/attribution jobs live in memory, so every poll must hit
    # the same container. Stays warm 10 min after the last request, then scales to zero.
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    import sys

    sys.path.insert(0, APP_DIR)
    from app import app as fastapi_app  # the same FastAPI server you run locally

    return fastapi_app
