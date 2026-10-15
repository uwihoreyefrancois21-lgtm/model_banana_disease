import joblib
import numpy as np

from . import config
from .schema import ALLOWED_CATEGORICAL, CATEGORICAL_FEATURES, FEATURE_FIELDS, NUMERIC_FEATURES

_pipeline = None
_metadata = config.load_metadata()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not config.MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model artifact not found at {config.MODEL_PATH}. "
                "Run scripts/export_model.py once (see README) to create it from the notebook workflow."
            )
        _pipeline = joblib.load(config.MODEL_PATH)
    return _pipeline


def validate_payload(data):
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object"

    unknown = [key for key in data if key not in FEATURE_FIELDS]
    if unknown:
        return None, f"Unknown field(s): {', '.join(sorted(unknown))}"

    missing = [key for key in FEATURE_FIELDS if key not in data]
    if missing:
        return None, f"Missing required field(s): {', '.join(sorted(missing))}"

    cleaned = {}
    for key in NUMERIC_FEATURES:
        value = data[key]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            cleaned[key] = np.nan
            continue
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            return None, f"Field '{key}' must be numeric or null"

    for key in CATEGORICAL_FEATURES:
        value = data[key]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            cleaned[key] = np.nan
            continue
        if not isinstance(value, str):
            return None, f"Field '{key}' must be a string category or null"
        if value not in ALLOWED_CATEGORICAL[key]:
            allowed = ", ".join(ALLOWED_CATEGORICAL[key])
            return None, f"Invalid value for '{key}': {value}. Allowed: {allowed}"
        cleaned[key] = value

    return cleaned, None


def predict_from_features(data):
    cleaned, error = validate_payload(data)
    if error:
        return None, error, 400

    row = [cleaned[key] for key in FEATURE_FIELDS]
    features = np.array([row], dtype=object)
    pipeline = get_pipeline()
    predicted = pipeline.predict(features)[0]
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)[0]
        classes = list(pipeline.classes_)
        probabilities = {label: float(proba[i]) for i, label in enumerate(classes)}

    return {
        "predicted_class": str(predicted),
        "prediction": str(predicted),
        "probabilities": probabilities,
        "model_name": _metadata.get("model_name", "Logistic Regression"),
    }, None, 200
