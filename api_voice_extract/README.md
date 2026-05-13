# Voice Feature Extraction API

Extracts **6373 OpenSMILE ComParE 2016** acoustic features from a `.wav` audio file.

---

## Endpoint

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/extract` | Extract features from audio file |

---

## How to Run

**1. Create your `.env` file:**
```
VOICE_EXTRACT_KEY=your_strong_api_key_here
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8013 --reload
```

**4. Run the test script:**
```bash
python run.py
```

---

## How to Use

```python
import requests

url = "http://127.0.0.1:8013/extract"
headers = {"x-api-key": "your_api_key"}

with open("audio.wav", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("audio.wav", f, "audio/wav")},
        headers=headers
    )

print(response.json())
```

---

## Input

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| file | .wav file | form-data | Audio file to extract features from |
| x-api-key | string | header | API authentication key |

## Output

```json
{
  "status": "success",
  "filename": "audio.wav",
  "feature_count": 6373,
  "features": {
    "F0final_sma_amean": 187.432,
    "jitterLocal_sma_amean": 0.00312,
    "...": "6373 features total"
  }
}
```

---

## File Structure

```
api_voice_extract/
├── app.py           — main FastAPI application
├── run.py           — test runner script
├── output.json      — sample API response
├── example.env      — environment variable template
├── requirements.txt — Python dependencies
└── README.md        — this file
```
