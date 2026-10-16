# Deploy on Render (one repository, one port)

## Setup

1. Push this **entire repo** to GitHub.
2. Render → **New Web Service** → connect the repo.
3. Use these settings (or import `render.yaml`):

| Setting | Value |
|---------|--------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn -c gunicorn.conf.py app:app` |
| **Python** | 3.11 |

4. Ensure model files are in the repo:

   - `banana_leaf_disease_api/models/banana_leaf_mobilenetv2.tflite`
   - `banana_leaf_disease_api/models/class_names.json`
   - `student_academic_risk_api/models/student_academicrisk.keras`
   - `student_academic_risk_api/models/student_academicrisk_preprocessor.joblib`
   - `student_academic_risk_api/models/student_academicrisk_labels.json`

Render sets **`PORT`** automatically — do not hard-code 8080 in production.

## After deploy — one URL, two APIs

Replace `https://YOUR-SERVICE.onrender.com` with your Render URL:

| Model | Method | URL |
|-------|--------|-----|
| Banana | `GET` | `https://YOUR-SERVICE.onrender.com/banana/health` |
| Banana | `POST` | `https://YOUR-SERVICE.onrender.com/banana/predict` (multipart `image`) |
| Student | `GET` | `https://YOUR-SERVICE.onrender.com/student/health` |
| Student | `POST` | `https://YOUR-SERVICE.onrender.com/student/predict` (JSON body) |
| Index | `GET` | `https://YOUR-SERVICE.onrender.com/` |

## Local (same paths as Render)

```bash
python app.py
```

Default `http://127.0.0.1:8080` — use `/banana/...` and `/student/...` (not root `/predict`).

## Frontend

```javascript
const RENDER_BASE = "https://YOUR-SERVICE.onrender.com";

// Banana only
fetch(`${RENDER_BASE}/banana/predict`, { method: "POST", body: formData });

// Student only
fetch(`${RENDER_BASE}/student/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(studentFields),
});
```
