import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "banana_leaf_mobilenetv2.keras"
CLASS_NAMES_PATH = BASE_DIR / "outputs" / "class_names.json"
IMG_SIZE = (224, 224)
USE_TTA = True

if CLASS_NAMES_PATH.exists():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["cordana", "healthy", "pestalotiopsis", "sigatoka"]

print(f"Loading model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"ERROR: model file not found at {MODEL_PATH}")
    sys.exit(1)

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Input shape: {model.input_shape}, Output shape: {model.output_shape}")
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)


def load_and_preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def predict_single(arr):
    batch = preprocess_input(tf.cast(np.expand_dims(arr, axis=0), tf.float32))
    probs = model.predict(batch, verbose=0)[0]
    return probs


def predict_tta(arr):
    views = (
        arr,
        arr[:, ::-1],
        arr[::-1, :],
        arr[::-1, ::-1],
    )
    batch = np.stack([v for v in views], axis=0)
    batch = preprocess_input(tf.cast(batch, tf.float32))
    all_probs = model.predict(batch, verbose=0)
    return np.mean(all_probs, axis=0)


def green_pixel_ratio(arr):
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    green_dominant = (g > r) & (g > b) & (g > 40.0)
    return float(np.mean(green_dominant))


def texture_stddev(arr):
    gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
    return float(np.std(gray))


def mean_adjacent_diff(arr):
    gray = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114).astype(np.float32)
    h_diff = np.abs(np.diff(gray, axis=1))
    v_diff = np.abs(np.diff(gray, axis=0))
    return float(0.5 * (np.mean(h_diff) + np.mean(v_diff)))


def is_banana_leaf(arr, probs):
    green_ratio = green_pixel_ratio(arr)
    stddev = texture_stddev(arr)
    adj_diff = mean_adjacent_diff(arr)
    n = len(probs)
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)) / np.log(n))
    max_prob = float(np.max(probs))

    reasons = []
    if green_ratio < 0.08:
        reasons.append(f"low green content ({green_ratio:.2f})")
    if stddev < 6.0:
        reasons.append(f"flat image (texture {stddev:.1f})")
    if adj_diff > 55.0:
        reasons.append(f"noisy image (smoothness {adj_diff:.1f})")
    if max_prob < 0.20:
        reasons.append(f"low confidence ({max_prob:.2f})")
    if entropy > 0.90:
        reasons.append(f"high uncertainty (entropy {entropy:.2f})")

    info = {"green_ratio": round(green_ratio, 3), "stddev": round(stddev, 2),
            "smoothness": round(adj_diff, 2), "entropy": round(entropy, 3),
            "max_prob": round(max_prob, 3)}
    if reasons:
        info["reason"] = "; ".join(reasons)
        return False, info
    return True, info


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": CLASS_NAMES,
        "endpoint": "POST /predict with form field 'image'",
        "tta": USE_TTA,
    })


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH.name,
        "classes": CLASS_NAMES,
        "endpoint": "POST /predict with form field 'image'",
        "tta": USE_TTA,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing form field 'image'"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        file_bytes = file.read()
        arr = load_and_preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    try:
        if USE_TTA:
            probs = predict_tta(arr)
        else:
            probs = predict_single(arr)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    probabilities = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    is_leaf, leaf_info = is_banana_leaf(arr, probs)
    if not is_leaf:
        zero_probs = {cls: 0.0 for cls in CLASS_NAMES}
        return jsonify({
            "predicted_class": "Not a banana leaf",
            "probabilities": zero_probs,
            "class_probabilities": zero_probs,
            "prediction": "Not a banana leaf",
            "not_a_leaf": True,
            "leaf_info": leaf_info,
        })

    return jsonify({
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "class_probabilities": probabilities,
        "prediction": predicted_class,
        "not_a_leaf": False,
        "leaf_info": leaf_info,
    })


if __name__ == "__main__":
    print("Starting Banana Leaf AI backend on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
