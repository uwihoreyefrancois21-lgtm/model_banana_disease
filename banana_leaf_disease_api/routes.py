from flask import Blueprint, jsonify, request

from . import config
from .inference import CLASS_NAMES, MODEL_PATH, USE_TTA, run_prediction


def create_blueprint(url_prefix: str = "") -> Blueprint:
    bp = Blueprint("banana", __name__, url_prefix=url_prefix)

    def api_metadata():
        prefix = url_prefix or ""
        return {
            "service": "banana-leaf-disease-api",
            "status": "ok",
            "model": MODEL_PATH.name,
            "classes": CLASS_NAMES,
            "image_size": list(config.IMG_SIZE),
            "preprocessing": "MobileNetV2 preprocess_input on uint8 RGB 224x224",
            "tta": USE_TTA,
            "base_path": prefix or "/",
            "endpoints": {
                "health": f"GET {prefix}/health",
                "predict": f"POST {prefix}/predict (multipart form field: image)",
            },
        }

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @bp.route("/", methods=["GET"])
    @bp.route("/api", methods=["GET"])
    def api_info():
        return jsonify(api_metadata())

    @bp.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "Missing form field 'image'"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        file_bytes = file.read()
        try:
            payload = run_prediction(file_bytes)
        except (OSError, ValueError) as exc:
            return jsonify({"error": f"Could not decode image: {exc}"}), 400
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {exc}"}), 500

        return jsonify(payload)

    return bp
