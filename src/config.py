from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
HF_CACHE_DIR = BASE_DIR / "hf-cache"

MODEL_ID = "is36e/detr-resnet-50-sku110k"
CONFIDENCE_THRESHOLD = 0.6
MAX_IMAGE_EDGE = 1600
MAX_DETECTIONS = 250

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
