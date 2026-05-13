"""
app.py — Voice Feature Extraction API
──────────────────────────────────────
Accepts a .wav audio file and returns 6373 OpenSMILE
ComParE 2016 acoustic features as JSON.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8013 --reload

Endpoint:
    POST /extract
"""

import os
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import opensmile

load_dotenv()
API_KEY = os.getenv("VOICE_EXTRACT_KEY")

app = FastAPI(
    title="Voice Feature Extraction API",
    description="Extracts 6373 OpenSMILE ComParE 2016 features from a .wav audio file.",
    version="1.0.0"
)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def verify_key(x_api_key: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized — invalid API key.")


@app.get("/")
def home():
    return {"message": "Voice Feature Extraction API is running."}


@app.post("/extract")
async def extract_features(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    """
    Extract 6373 OpenSMILE ComParE 2016 features from an uploaded .wav file.

    Headers:
        x-api-key: your API key

    Body:
        file: .wav audio file (multipart/form-data)

    Returns:
        JSON with filename, feature_count, and features dict
    """
    verify_key(x_api_key)

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are accepted.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        features    = smile.process_file(tmp_path)
        feature_dict = {k: round(float(v), 6) for k, v in features.iloc[0].to_dict().items()}

        return JSONResponse({
            "status"       : "success",
            "filename"     : file.filename,
            "feature_count": len(feature_dict),
            "features"     : feature_dict
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
