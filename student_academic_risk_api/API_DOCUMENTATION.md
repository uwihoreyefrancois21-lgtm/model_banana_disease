# Student Academic Risk API — Documentation

## Model (deployed)

| Item | Value |
|------|--------|
| **Weights file** | `student_academic_risk_api/models/student_academicrisk.keras` |
| **Preprocessor** | `student_academic_risk_api/models/student_academicrisk_preprocessor.joblib` (required at runtime) |
| **Label order** | `student_academic_risk_api/models/student_academicrisk_labels.json` |
| **Metrics / metadata** | `student_academic_risk_api/models/model_metadata.json` |
| **Training source** | `student_academic_risk_kfold_models.ipynb` (same features & preprocessing as notebook) |
| **Architecture** | Keras softmax classifier (logistic regression equivalent on preprocessed features) |
| **Inference engine** | TensorFlow Keras + scikit-learn `ColumnTransformer` preprocessor |
| **Target classes** | `At_Risk`, `Not_At_Risk` |

**Held-out test performance** (in `model_metadata.json`):

| Metric | Deployed `.keras` | Notebook sklearn baseline |
|--------|-------------------|---------------------------|
| Accuracy | 68.25% | 68.42% |
| Macro F1 | 68.21% | 68.29% |

Optional env overrides:

| Variable | Default |
|----------|---------|
| `STUDENT_RISK_MODEL_PATH` | `models/student_academicrisk.keras` |
| `STUDENT_RISK_PREPROCESSOR_PATH` | `models/student_academicrisk_preprocessor.joblib` |
| `STUDENT_RISK_LABELS_PATH` | `models/student_academicrisk_labels.json` |
| `STUDENT_RISK_METADATA_PATH` | `models/model_metadata.json` |

Pre-deploy check:

```bash
python student_academic_risk_api/deploy_check.py
```

Regenerate artifacts (one time, needs CSV locally):

```bash
python student_academic_risk_api/scripts/export_model.py --data dataset/student_academic_risk_dataset.csv
```

---

## Base URLs

**Production (single site — recommended):**

- Base: `https://your-app.example.com/student`
- Health: `GET https://your-app.example.com/student/health`
- Predict: `POST https://your-app.example.com/student/predict`

Start from repo root: `gunicorn -c gunicorn.conf.py host_app:app`

**Local standalone (optional):**

- Base: `http://127.0.0.1:5001`
- Run: `python student_academic_risk_api/app.py`

CORS is enabled for browser clients.

---

## Endpoints

Paths use `{base}` = `/student` when hosted, or `` when standalone on port 5001.

### `GET {base}/health`

**200** when `student_academicrisk.keras` and preprocessor load:

```json
{
  "status": "ok",
  "inference_engine": "tensorflow.keras",
  "artifacts": {
    "keras_model": "student_academicrisk.keras",
    "keras_model_present": true,
    "preprocessor_present": true,
    "labels_present": true
  }
}
```

**503** if deploy files missing:

```json
{
  "status": "degraded",
  "reason": "Deploy student_academicrisk.keras and student_academicrisk_preprocessor.joblib",
  "artifacts": {
    "keras_model": "student_academicrisk.keras",
    "keras_model_present": false,
    "preprocessor_present": false,
    "labels_present": false
  }
}
```

---

### `GET {base}/` and `GET {base}/api`

**Response 200** (when model files present)

```json
{
  "service": "student-academic-risk-api",
  "status": "ok",
  "model_file": "student_academicrisk.keras",
  "model_ready": true,
  "notebook_reference": "student_academic_risk_kfold_models.ipynb",
  "notebook_best_model": "Logistic Regression",
  "deployed_model": "Keras Logistic Regression (deployed .keras)",
  "model_performance": {
    "summary": "Held-out test accuracy 68.25% (F1 macro 68.21%)",
    "deployed_keras_test": {
      "model_name": "Keras Logistic Regression (deployed .keras)",
      "accuracy_display": "68.25%",
      "f1_macro_display": "68.21%"
    },
    "notebook_logistic_regression_test": {
      "accuracy_display": "68.42%",
      "f1_macro_display": "68.29%"
    }
  },
  "target_classes": ["At_Risk", "Not_At_Risk"],
  "feature_fields": [
    "Age", "Gender", "Program", "Study_Hours_Per_Week",
    "Attendance_Percent", "Previous_Average_Score", "Assignment_Score",
    "Internet_Access", "Part_Time_Work", "Financial_Stress",
    "Commute_Minutes", "Missed_Deadlines"
  ],
  "preprocessing": "median/mode imputation, StandardScaler, OneHotEncoder(handle_unknown='ignore')",
  "base_path": "/student",
  "endpoints": {
    "health": "GET /student/health",
    "predict": "POST /student/predict (JSON body with feature fields)"
  }
}
```

---

### `POST {base}/predict`

Runs inference with **`student_academicrisk.keras`** (after preprocessor transform).

**Content-Type:** `application/json`

All fields **required**. Use `null` for missing numeric/categorical values (imputed in pipeline).

| Field | Type | Allowed values |
|-------|------|----------------|
| `Age` | number | — |
| `Gender` | string | `Female`, `Male` |
| `Program` | string | `Business`, `Education`, `Engineering`, `Hospitality`, `IT` |
| `Study_Hours_Per_Week` | number or null | — |
| `Attendance_Percent` | number | 0–100 |
| `Previous_Average_Score` | number or null | — |
| `Assignment_Score` | number | — |
| `Internet_Access` | string | `Yes`, `No` |
| `Part_Time_Work` | string | `Yes`, `No` |
| `Financial_Stress` | string | `Low`, `Medium`, `High` |
| `Commute_Minutes` | number | — |
| `Missed_Deadlines` | number | — |

`Student_ID` is **not** accepted (identifier only in training data).

**Example request**

```json
{
  "Age": 23,
  "Gender": "Male",
  "Program": "IT",
  "Study_Hours_Per_Week": 4.8,
  "Attendance_Percent": 80.5,
  "Previous_Average_Score": 70.7,
  "Assignment_Score": 36.0,
  "Internet_Access": "Yes",
  "Part_Time_Work": "No",
  "Financial_Stress": "Low",
  "Commute_Minutes": 49.2,
  "Missed_Deadlines": 3
}
```

**Success 200**

```json
{
  "predicted_class": "At_Risk",
  "prediction": "At_Risk",
  "probabilities": {
    "At_Risk": 0.537,
    "Not_At_Risk": 0.463
  },
  "model_file": "student_academicrisk.keras",
  "model_name": "Keras Logistic Regression (deployed .keras)",
  "model_accuracy_display": "68.25%",
  "model_f1_macro_display": "68.21%"
}
```

**Errors**

| Status | Example | Cause |
|--------|---------|--------|
| 400 | `{ "error": "Content-Type must be application/json" }` | Wrong content type |
| 400 | `{ "error": "Missing required field(s): ..." }` | Incomplete JSON |
| 400 | `{ "error": "Invalid value for 'Program': ..." }` | Bad category |
| 503 | Health degraded | `.keras` or preprocessor not on server |

---

## Frontend integration

```javascript
// Combined host (production)
const API_BASE = "https://your-app.example.com/student";

async function predictRisk(student) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(student),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);

  return {
    label: data.predicted_class,
    probs: data.probabilities,
    model: data.model_file,
  };
}
```

**cURL**

```bash
curl -X POST "https://your-app.example.com/student/predict" \
  -H "Content-Type: application/json" \
  -d "{\"Age\":23,\"Gender\":\"Male\",\"Program\":\"IT\",\"Study_Hours_Per_Week\":4.8,\"Attendance_Percent\":80.5,\"Previous_Average_Score\":70.7,\"Assignment_Score\":36.0,\"Internet_Access\":\"Yes\",\"Part_Time_Work\":\"No\",\"Financial_Stress\":\"Low\",\"Commute_Minutes\":49.2,\"Missed_Deadlines\":3}"
```

---

## Deployment checklist

1. Push repo; use root **`Procfile`**: `gunicorn -c gunicorn.conf.py host_app:app`
2. Upload all four files under `student_academic_risk_api/models/`:
   - **`student_academicrisk.keras`**
   - **`student_academicrisk_preprocessor.joblib`**
   - **`student_academicrisk_labels.json`**
   - **`model_metadata.json`**
3. Install root **`requirements.txt`** (TensorFlow + scikit-learn + Flask)
4. Run **`python student_academic_risk_api/deploy_check.py`** on the host
5. Frontend calls **`/student/predict`**

**Verify:** `GET /student/api` → `"model_file": "student_academicrisk.keras"`, `"model_ready": true`

**Note:** Dataset is synthetic/teaching data; use outputs for demo/education, not real institutional decisions.
