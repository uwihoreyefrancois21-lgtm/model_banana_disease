# ML Flask APIs (single repository)

Two independent backends in one GitHub repo — deploy separately or on different ports/processes.

| Service | Folder | Default port |
|---------|--------|----------------|
| Banana leaf disease | `banana_leaf_disease_api/` | 5000 |
| Student academic risk | `student_academic_risk_api/` | 5001 |

## Install once (root)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Same libraries for both apps: Flask, NumPy, scikit-learn, TensorFlow, etc. **Pandas is not required** for serving predictions.

## Run locally

**Option A — one URL (same as hosting):**

```bash
set PORT=8080
python app.py
```

- Banana: `http://127.0.0.1:8080/banana/predict`
- Student: `http://127.0.0.1:8080/student/predict`

**Option B — two ports (dev only):**

```bash
cd banana_leaf_disease_api && set PORT=5000 && python app.py
cd student_academic_risk_api && set PORT=5001 && python app.py
```

**Render (one repo, one Web Service, one `PORT`):** see [`RENDER.md`](RENDER.md).

```bash
gunicorn -c gunicorn.conf.py app:app
```

Frontend on Render (same host, separate paths):

- `https://YOUR-SERVICE.onrender.com/banana/predict`
- `https://YOUR-SERVICE.onrender.com/student/predict`

## Student model artifact (deployed backend uses `.keras`)

The **`/student`** API loads **`student_academic_risk_api/models/student_academicrisk.keras`** (+ preprocessor joblib). Include these files when you push or deploy:

- `student_academicrisk.keras`
- `student_academicrisk_preprocessor.joblib`
- `student_academicrisk_labels.json`
- `model_metadata.json`

Regenerate if needed:

```bash
python student_academic_risk_api/scripts/export_model.py --data dataset/student_academic_risk_dataset.csv
```

## API documentation

| Service | Document |
|---------|----------|
| Banana leaf disease | [`banana_leaf_disease_api/API_DOCUMENTATION.md`](banana_leaf_disease_api/API_DOCUMENTATION.md) |
| Student academic risk | [`student_academic_risk_api/API_DOCUMENTATION.md`](student_academic_risk_api/API_DOCUMENTATION.md) |

Hosted paths: `/banana/...` and `/student/...` (see each doc).

Root **`app.py`** is the combined entry (same as `host_app.py` for Gunicorn).
