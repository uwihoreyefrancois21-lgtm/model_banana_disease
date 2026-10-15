from flask import Blueprint, jsonify, request

from . import config
from .inference import get_pipeline, predict_from_features
from .schema import FEATURE_FIELDS, TARGET_CLASSES


def create_blueprint(url_prefix: str = "") -> Blueprint:
    bp = Blueprint("student_risk", __name__, url_prefix=url_prefix)

    def api_metadata():
        metadata = config.load_metadata()
        model_ready = config.MODEL_PATH.is_file()
        prefix = url_prefix or ""
        return {
            "service": "student-academic-risk-api",
            "status": "ok" if model_ready else "model_missing",
            "model_file": config.MODEL_PATH.name,
            "model_ready": model_ready,
            "best_model": metadata.get("model_name", "Logistic Regression"),
            "target_classes": TARGET_CLASSES,
            "feature_fields": FEATURE_FIELDS,
            "preprocessing": metadata.get(
                "preprocessing",
                "median/mode imputation, one-hot encoding, standardization (notebook pipeline)",
            ),
            "base_path": prefix or "/",
            "endpoints": {
                "health": f"GET {prefix}/health",
                "predict": f"POST {prefix}/predict (JSON body with feature fields)",
            },
        }

    @bp.route("/health", methods=["GET"])
    def health():
        if not config.MODEL_PATH.is_file():
            return jsonify({"status": "degraded", "reason": "model artifact missing"}), 503
        try:
            get_pipeline()
        except Exception as exc:
            return jsonify({"status": "degraded", "reason": str(exc)}), 503
        return jsonify({"status": "ok"})

    @bp.route("/", methods=["GET"])
    @bp.route("/api", methods=["GET"])
    def api_info():
        return jsonify(api_metadata())

    @bp.route("/predict", methods=["POST"])
    def predict():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        body = request.get_json(silent=True)
        result, error, status = predict_from_features(body)
        if error:
            return jsonify({"error": error}), status
        return jsonify(result)

    return bp
