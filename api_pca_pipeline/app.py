"""
app.py — PCA Pipeline API
──────────────────────────
Accepts 6373 raw acoustic features (from Voice Feature Extraction API)
and returns 24 PCA components after applying:
  1. Articulation filter
  2. Feature engineering selection
  3. StandardScaler
  4. PCA (95% variance)

Run:
    uvicorn app:app --host 0.0.0.0 --port 5900 --reload

Endpoint:
    POST /process
"""

import os
import numpy as np
import joblib
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()
API_KEY   = os.getenv("PCA_PROCESS_KEY")
MODELS_DIR = Path(__file__).parent.parent / "models"

# Load all saved pipeline models at startup
articulation_cols   = joblib.load(MODELS_DIR / "articulation_cols.pkl")
feature_eng_cols    = joblib.load(MODELS_DIR / "feature_engineering_cols.pkl")
scaler              = joblib.load(MODELS_DIR / "scaler.pkl")
pca                 = joblib.load(MODELS_DIR / "pca.pkl")
label_encoder       = joblib.load(MODELS_DIR / "label_encoder.pkl")

app = FastAPI(
    title="PCA Pipeline API",
    description="Processes 6373 raw features through articulation filter, feature engineering, scaler and PCA. Returns 24 PCA components.",
    version="1.0.0"
)


class FeaturesInput(BaseModel):
    features: Dict[str, float]


def verify_key(x_api_key: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized — invalid API key.")


@app.get("/")
def home():
    return {
        "message"         : "PCA Pipeline API is running.",
        "input_features"  : 6373,
        "output_components": int(pca.n_components_)
    }


@app.post("/process")
def process_features(
    payload: FeaturesInput,
    x_api_key: str = Header(...)
):
    """
    Process 6373 raw features through the full pipeline and return 24 PCA components.

    Headers:
        x-api-key: your API key

    Body:
        { "features": { "feature_name": value, ... } }

    Returns:
        JSON with pca_components count and components dict
    """
    verify_key(x_api_key)

    try:
        features = payload.features

        if len(features) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Expected ~6373 features, got {len(features)}. Pass full OpenSMILE output."
            )

        # Step 1 — Articulation filter + feature engineering selection
        # feature_eng_cols contains the exact 120 column names selected during training
        eng_values = [features.get(col, 0.0) for col in feature_eng_cols]
        missing    = [col for col in feature_eng_cols if col not in features]

        if missing:
            return JSONResponse({
                "status" : "warning",
                "message": f"{len(missing)} expected features were missing — filled with 0.0",
                "missing_count": len(missing)
            }, status_code=206)

        # Step 2 — StandardScaler
        scaled = scaler.transform([eng_values])

        # Step 3 — PCA
        pca_out    = pca.transform(scaled)[0]
        components = {f"PC{i+1}": round(float(v), 6) for i, v in enumerate(pca_out)}

        return JSONResponse({
            "status"          : "success",
            "input_features"  : len(features),
            "selected_features": len(feature_eng_cols),
            "pca_components"  : len(pca_out),
            "variance_kept"   : round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
            "components"      : components
        })

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
