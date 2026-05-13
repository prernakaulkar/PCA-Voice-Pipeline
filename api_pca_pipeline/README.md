# PCA Pipeline API

Accepts **6373 raw acoustic features** (from the Voice Feature Extraction API)
and returns **24 PCA components** after applying articulation filtering,
feature engineering selection, StandardScaler, and PCA.

---

## Endpoint

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check — shows input/output config |
| POST | `/process` | Process features and return PCA components |

---

## How to Run

**1. Create your `.env` file:**
```
PCA_PROCESS_KEY=your_strong_api_key_here
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8014 --reload
```

**4. Run the test script:**
```bash
python run.py
```

---

## How to Use

```python
import requests

url = "http://127.0.0.1:8014/process"
headers = {"x-api-key": "your_api_key"}

payload = {
    "features": {
        "F0final_sma_amean": 187.432,
        "jitterLocal_sma_amean": 0.00312,
        "...": "all 6373 features from Voice Extraction API"
    }
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

## Input

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| features | dict | JSON body | All 6373 OpenSMILE feature names and values |
| x-api-key | string | header | API authentication key |

## Output

```json
{
  "status": "success",
  "input_features": 6373,
  "selected_features": 120,
  "pca_components": 24,
  "variance_kept": 95.01,
  "components": {
    "PC1": 2.413241,
    "PC2": -1.072341,
    "...": "24 components total"
  }
}
```

---

## Pipeline Inside This API

```
6373 raw features
      ↓ articulation + feature engineering selection
120 features
      ↓ StandardScaler
120 normalised features
      ↓ PCA
24 components (95% variance)
```

---

## File Structure

```
api_pca_pipeline/
├── app.py           — main FastAPI application
├── run.py           — test runner script
├── payload.json     — sample input with feature values
├── output.json      — sample API response
├── example.env      — environment variable template
├── requirements.txt — Python dependencies
└── README.md        — this file
```
