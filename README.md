# PCA Voice Pipeline — Mental Health Voice Prediction System

A voice-based mental health prediction pipeline that extracts acoustic features from audio and processes them through an articulation filter and PCA for mental health condition analysis.

---

## System Overview

```
Audio Input (.wav)
      ↓
API 1 — Voice Feature Extraction  (port 5800)
      ↓ 6373 OpenSMILE features
API 2 — PCA Pipeline              (port 5900)
      ↓ 24 PCA components
Ready for ML prediction
```

---

## APIs

| API | Port | Input | Output |
|-----|------|-------|--------|
| Voice Feature Extraction | 5800 | .wav audio file | 6373 acoustic features |
| PCA Pipeline | 5900 | 6373 features JSON | 24 PCA components |

---

## Project Structure

```
PCA-Voice-Pipeline/
├── api_voice_extract/       — Voice Feature Extraction API
│   ├── app.py               — FastAPI application
│   ├── run.py               — Test runner
│   ├── output.json          — Sample response
│   ├── example.env          — Environment variable template
│   ├── requirements.txt     — Dependencies
│   ├── Dockerfile           — Docker container config
│   └── README.md            — API documentation
├── api_pca_pipeline/        — PCA Pipeline API
│   ├── app.py               — FastAPI application
│   ├── run.py               — Test runner (chains both APIs)
│   ├── payload.json         — Sample input
│   ├── output.json          — Sample response
│   ├── example.env          — Environment variable template
│   ├── requirements.txt     — Dependencies
│   ├── Dockerfile           — Docker container config
│   └── README.md            — API documentation
├── src/                     — Pipeline training scripts
│   ├── config.py
│   ├── step1_extract.py
│   ├── step2_articulation.py
│   ├── step3_variance.py
│   ├── step3b_feature_engineering.py
│   ├── step4_smote.py
│   └── step5_pca.py
├── docker-compose.yml       — Run both APIs together
├── run_pipeline.py          — Master pipeline runner
├── requirements.txt         — Root dependencies
└── .gitignore
```

---

## Quick Start

### Run with Docker (Recommended)

```bash
git clone https://github.com/prernakaulkar/PCA-Voice-Pipeline.git
cd PCA-Voice-Pipeline
docker-compose up --build
```

APIs will be available at:
- API 1: `http://localhost:5800`
- API 2: `http://localhost:5900`

### Run Locally

**1. Create `.env` files:**
```
api_voice_extract/.env  → VOICE_EXTRACT_KEY=your_key
api_pca_pipeline/.env   → PCA_PROCESS_KEY=your_key
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start API 1:**
```bash
cd api_voice_extract
uvicorn app:app --host 0.0.0.0 --port 5800 --reload
```

**4. Start API 2:**
```bash
cd api_pca_pipeline
uvicorn app:app --host 0.0.0.0 --port 5900 --reload
```

**5. Test both APIs:**
```bash
cd api_pca_pipeline
python run.py
```

---

## Pipeline Stages

| Stage | Input | Output |
|-------|-------|--------|
| 1 — OpenSMILE extraction | 714 audio files | 6373 features |
| 2 — Articulation filter | 6373 features | 2848 features |
| 3 — Variance threshold | 2848 features | 1736 features |
| 3b — Feature engineering | 1736 features | 120 features |
| 4 — SMOTE oversampling | 714 samples | 10002 samples |
| 5 — PCA | 120 features | 24 components |

---

## Mental Health Conditions

```
Normal
Depression
Anxiety
Stress
Bipolar
Suicidal
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| OpenSMILE ComParE 2016 | Acoustic feature extraction |
| scikit-learn | Feature selection, PCA, ML |
| imbalanced-learn | SMOTE oversampling |
| FastAPI | REST API framework |
| Docker | Containerisation |
| Python 3.11 | Runtime |

---

## Environment Variables

| Variable | API | Description |
|----------|-----|-------------|
| `VOICE_EXTRACT_KEY` | API 1 | Authentication key for feature extraction |
| `PCA_PROCESS_KEY` | API 2 | Authentication key for PCA pipeline |

## Testing the APIs

### Sample Audio Files
Test audio files are available in the `test_samples/` folder.
Use any `.wav` file from there to test both APIs.

---

### Step 1 — Start both APIs
```bash
docker-compose up --build -d
```

### Step 2 — Test API 1 (Feature Extraction)
```python
import requests

url = "http://YOUR_VPS_IP:5800/extract"
headers = {"x-api-key": "your_voice_extract_key"}

with open("test_samples/P008_2025-10-01_AIIMSN_800.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("sample.wav", f, "audio/wav")},
        headers=headers
    )

print(response.json())
```

### Step 3 — Test API 2 (PCA Pipeline)
```python
import requests

# Pass features from API 1 response into API 2
features = response.json()["features"]

url = "http://YOUR_VPS_IP:5900/process"
headers = {"x-api-key": "your_pca_process_key"}

response2 = requests.post(
    url,
    json={"features": features},
    headers=headers
)

print(response2.json())
```

### Expected Output — API 1
```json
{
  "status": "success",
  "filename": "sample.wav",
  "feature_count": 6373,
  "features": { "F0final_sma_amean": 187.4, "..." : "..." }
}
```

### Expected Output — API 2
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