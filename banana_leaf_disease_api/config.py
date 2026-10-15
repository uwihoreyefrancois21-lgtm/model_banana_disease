import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(os.environ.get("BANANA_MODEL_PATH", BASE_DIR / "models" / "banana_leaf_mobilenetv2.keras"))
CLASS_NAMES_PATH = Path(
    os.environ.get("BANANA_CLASS_NAMES_PATH", BASE_DIR / "models" / "class_names.json")
)
IMG_SIZE = (224, 224)
USE_TTA = os.environ.get("BANANA_USE_TTA", "true").lower() in ("1", "true", "yes")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
