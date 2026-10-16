import io
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASS_NAMES_PATH, IMG_SIZE, MODEL_PATH as _KERAS_MODEL_PATH, USE_TTA

# The Keras model is converted to TFLite (same directory, same stem). The Lite
# interpreter keeps a small arena instead of the full Keras graph, which is what
# makes this fit a small host, and it runs the same weights ~2.5x faster.
MODEL_PATH = _KERAS_MODEL_PATH.with_suffix(".tflite")


def load_class_names():
    if CLASS_NAMES_PATH.is_file():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as handle:
            names = json.load(handle)
        if isinstance(names, list) and names:
            return names
    return ["cordana", "healthy", "pestalotiopsis", "sigatoka"]


def load_interpreter():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


CLASS_NAMES = load_class_names()
interpreter = load_interpreter()
_input = interpreter.get_input_details()[0]
_output = interpreter.get_output_details()[0]


def load_and_preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess(batch):
    """MobileNetV2 preprocess_input: uint8 RGB -> float32 in [-1, 1]."""
    return (batch.astype(np.float32) / 127.5) - 1.0


def forward(batch):
    """One forward pass for a single uint8 RGB image through the Lite graph."""
    interpreter.set_tensor(_input["index"], preprocess(batch[np.newaxis, ...]))
    interpreter.invoke()
    return interpreter.get_tensor(_output["index"])[0].copy()


def predict_single(arr):
    return forward(arr)


def predict_tta(arr):
    views = (
        arr,
        arr[:, ::-1],
        arr[::-1, :],
        arr[::-1, ::-1],
    )
    return np.mean([forward(view) for view in views], axis=0)


def warmup():
    """Run once at import so the first user request is not the slow one."""
    dummy = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    predict_single(dummy)
    if USE_TTA:
        predict_tta(dummy)


warmup()


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

    info = {
        "green_ratio": round(green_ratio, 3),
        "stddev": round(stddev, 2),
        "smoothness": round(adj_diff, 2),
        "entropy": round(entropy, 3),
        "max_prob": round(max_prob, 3),
    }
    if reasons:
        info["reason"] = "; ".join(reasons)
        return False, info
    return True, info


def run_prediction(file_bytes):
    arr = load_and_preprocess_image(file_bytes)
    probs = predict_tta(arr) if USE_TTA else predict_single(arr)

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    probabilities = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}

    is_leaf, leaf_info = is_banana_leaf(arr, probs)
    if not is_leaf:
        zero_probs = {cls: 0.0 for cls in CLASS_NAMES}
        return {
            "predicted_class": "Not a banana leaf",
            "probabilities": zero_probs,
            "class_probabilities": zero_probs,
            "prediction": "Not a banana leaf",
            "not_a_leaf": True,
            "leaf_info": leaf_info,
        }

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "class_probabilities": probabilities,
        "prediction": predicted_class,
        "not_a_leaf": False,
        "leaf_info": leaf_info,
    }
