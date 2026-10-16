# Banana Leaf Disease API — Documentation

## Model (deployed)

| Item | Value |
|------|--------|
| **Weights file** | `banana_leaf_disease_api/models/banana_leaf_mobilenetv2.tflite` |
| **Class labels** | `banana_leaf_disease_api/models/class_names.json` |
| **Architecture** | MobileNetV2 fine-tuned classifier (4-class softmax) |
| **Training source** | `banana_disease.ipynb` |
| **Input** | RGB image, resized to 224×224 |
| **Preprocessing** | `tensorflow.keras.applications.mobilenet_v2.preprocess_input` |
| **Classes** | `cordana`, `healthy`, `pestalotiopsis`, `sigatoka` |
| **Inference engine** | TensorFlow Lite (`tf.lite.Interpreter`) |

Optional env overrides:

| Variable | Default |
|----------|---------|
| `BANANA_MODEL_PATH` | `models/banana_leaf_mobilenetv2.keras` (base path; the `.tflite` is loaded from the same stem) |
| `BANANA_CLASS_NAMES_PATH` | `models/class_names.json` |
| `BANANA_USE_TTA` | `false` (average 4 flip/rotate views when enabled; 4× slower) |

---

## Base URLs

**Production (single site — recommended):**

- Base: `https://your-app.example.com/banana`
- Health: `GET https://your-app.example.com/banana/health`
- Predict: `POST https://your-app.example.com/banana/predict`

Start from repo root: `gunicorn -c gunicorn.conf.py host_app:app`

**Local standalone (optional):**

- Base: `http://127.0.0.1:5000`
- Run: `python banana_leaf_disease_api/app.py`

CORS is enabled for browser frontends on another origin.

---

## Endpoints

Paths below use `{base}` = `/banana` when hosted, or `` (empty) when running standalone on port 5000.

### `GET {base}/health`

**Response 200**

```json
{ "status": "ok" }
```

---

### `GET {base}/` and `GET {base}/api`

**Response 200**

```json
{
  "service": "banana-leaf-disease-api",
  "status": "ok",
  "model": "banana_leaf_mobilenetv2.tflite",
  "classes": ["cordana", "healthy", "pestalotiopsis", "sigatoka"],
  "image_size": [224, 224],
  "preprocessing": "MobileNetV2 preprocess_input on uint8 RGB 224x224",
  "tta": false,
  "base_path": "/banana",
  "endpoints": {
    "health": "GET /banana/health",
    "predict": "POST /banana/predict (multipart form field: image)"
  }
}
```

---

### `POST {base}/predict`

Runs inference with **`banana_leaf_mobilenetv2.tflite`**.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | yes | Leaf photo (JPEG, PNG, etc.) |

**Success 200 — valid leaf**

```json
{
  "predicted_class": "sigatoka",
  "prediction": "sigatoka",
  "probabilities": {
    "cordana": 0.02,
    "healthy": 0.05,
    "pestalotiopsis": 0.08,
    "sigatoka": 0.85
  },
  "class_probabilities": {
    "cordana": 0.02,
    "healthy": 0.05,
    "pestalotiopsis": 0.08,
    "sigatoka": 0.85
  },
  "not_a_leaf": false,
  "leaf_info": {
    "green_ratio": 0.42,
    "stddev": 18.5,
    "smoothness": 12.3,
    "entropy": 0.35,
    "max_prob": 0.85
  }
}
```

**Success 200 — not a leaf** (heuristic check; class probabilities zeroed)

```json
{
  "predicted_class": "Not a banana leaf",
  "prediction": "Not a banana leaf",
  "probabilities": {
    "cordana": 0.0,
    "healthy": 0.0,
    "pestalotiopsis": 0.0,
    "sigatoka": 0.0
  },
  "not_a_leaf": true,
  "leaf_info": {
    "green_ratio": 0.03,
    "reason": "low green content (0.03); flat image (texture 4.2)"
  }
}
```

**Errors**

| Status | Body | Cause |
|--------|------|--------|
| 400 | `{ "error": "Missing form field 'image'" }` | No file part |
| 400 | `{ "error": "Empty filename" }` | Empty upload |
| 400 | `{ "error": "Could not decode image: ..." }` | Invalid image |
| 500 | `{ "error": "Inference failed: ..." }` | Missing `.tflite` or runtime error |

---

## Frontend integration

```javascript
// Combined host (production)
const API_BASE = "https://your-app.example.com/banana";

async function classifyLeaf(file) {
  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);

  if (data.not_a_leaf) {
    return { ok: false, reason: data.leaf_info };
  }
  return { ok: true, class: data.predicted_class, probs: data.probabilities };
}
```

**cURL**

```bash
curl -X POST "https://your-app.example.com/banana/predict" \
  -F "image=@leaf.jpg"
```

---

## Deployment checklist

1. Push the repo; use root **`Procfile`**: `gunicorn -c gunicorn.conf.py host_app:app`
2. Ship **`banana_leaf_disease_api/models/banana_leaf_mobilenetv2.tflite`** (~10 MB)
3. Ship **`banana_leaf_disease_api/models/class_names.json`**
4. Install root **`requirements.txt`** (includes TensorFlow, Flask, gunicorn)
5. Frontend calls **`/banana/predict`**, not `/predict` at site root

**Verify model path on server:** `GET /banana/api` → `"model": "banana_leaf_mobilenetv2.tflite"`
