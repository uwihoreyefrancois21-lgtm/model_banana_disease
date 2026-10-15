# Banana Leaf Disease API — Documentation

Base URL examples: `http://localhost:5000` (local) or `https://your-banana-api.example.com` (hosted).

CORS is enabled for browser frontends on another origin.

---

## Endpoints

### `GET /health`

**Response 200**

```json
{ "status": "ok" }
```

---

### `GET /` and `GET /api`

Service metadata.

**Response 200**

```json
{
  "service": "banana-leaf-disease-api",
  "status": "ok",
  "model": "banana_leaf_mobilenetv2.keras",
  "classes": ["cordana", "healthy", "pestalotiopsis", "sigatoka"],
  "image_size": [224, 224],
  "preprocessing": "MobileNetV2 preprocess_input on uint8 RGB 224x224",
  "tta": true,
  "endpoints": {
    "health": "GET /health",
    "predict": "POST /predict (multipart form field: image)"
  }
}
```

---

### `POST /predict`

Classify a banana leaf image. Matches notebook training: resize to 224×224, RGB, `mobilenet_v2.preprocess_input`, softmax over four disease classes. When `tta` is enabled, predictions are averaged over the original image and three flipped views (same as production `app.py` logic).

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

**Success 200 — rejected as non-leaf** (heuristic gate; probabilities zeroed)

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
  "class_probabilities": { "...": 0.0 },
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
| 400 | `{ "error": "Could not decode image: ..." }` | Invalid/corrupt image |
| 500 | `{ "error": "Inference failed: ..." }` | Model/runtime failure |

---

## Frontend integration

Point your hosted `banana_leaf_ai_frontend.html` (or any UI) at this API base URL.

**JavaScript (fetch)**

```javascript
const API_BASE = "https://your-banana-api.example.com";

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
    console.log("Not a leaf:", data.leaf_info);
    return;
  }

  console.log("Class:", data.predicted_class, data.probabilities);
}
```

**cURL**

```bash
curl -X POST "https://your-banana-api.example.com/predict" \
  -F "image=@leaf.jpg"
```

**Notes**

- Do not send JSON for image upload; use `FormData` / multipart.
- Class order matches `class_names.json` and notebook label ids 0–3.
- Disable TTA on low-memory hosts: set env `BANANA_USE_TTA=false`.

---

## Deployment checklist

1. Push the **whole repository** (both API folders); deploy this service from `banana_leaf_disease_api/`.
2. Ensure `models/banana_leaf_mobilenetv2.keras` is present (Git LFS if needed).
3. Install `requirements.txt` on the host.
4. Start with `gunicorn app:app` (or platform-specific WSGI).
5. Set `PORT` / reverse proxy to HTTPS for browser frontends.
