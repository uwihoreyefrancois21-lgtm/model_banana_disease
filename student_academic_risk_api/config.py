import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get("STUDENT_RISK_MODEL_PATH", BASE_DIR / "models" / "student_academicrisk.keras")
)
PREPROCESSOR_PATH = Path(
    os.environ.get(
        "STUDENT_RISK_PREPROCESSOR_PATH",
        BASE_DIR / "models" / "student_academicrisk_preprocessor.joblib",
    )
)
LABELS_PATH = Path(
    os.environ.get("STUDENT_RISK_LABELS_PATH", BASE_DIR / "models" / "student_academicrisk_labels.json")
)
METADATA_PATH = Path(
    os.environ.get("STUDENT_RISK_METADATA_PATH", BASE_DIR / "models" / "model_metadata.json")
)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5001"))


def load_metadata():
    if not METADATA_PATH.is_file():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)
