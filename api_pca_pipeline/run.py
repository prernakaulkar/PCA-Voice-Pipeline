"""
run.py — Test runner for PCA Pipeline API
──────────────────────────────────────────
Automatically chains both APIs:
  1. Sends audio file to API 1 (Voice Feature Extraction) → gets 6373 features
  2. Sends those features to API 2 (PCA Pipeline) → gets 24 PCA components

Usage:
    python run.py

Requirements:
    - API 1 must be running on port 8013
    - API 2 must be running on port 8014
    - .env file must exist with PCA_PROCESS_KEY set
    - Place a sample .wav file named sample_audio.wav in api_voice_extract folder
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────
API1_URL       = "http://127.0.0.1:5800/extract"
API2_URL       = "http://127.0.0.1:5900/process"
API1_KEY       = "scs@VoiceExtract#2026$secure"
API2_KEY       = os.getenv("PCA_PROCESS_KEY")
WAV_FILE       = r"D:\PCA_Voicepipeline\api_voice_extract\sample_audio.wav"

# ── Step 1 — Call API 1 to get 6373 features ─────────────────
print("=" * 55)
print("  STEP 1 — Voice Feature Extraction (API 1)")
print("=" * 55)

if not os.path.exists(WAV_FILE):
    print(f"[ERROR] Audio file not found: {WAV_FILE}")
    print("Please place a .wav file named 'sample_audio.wav'")
    print("inside the api_voice_extract folder.")
    exit(1)

print(f"  Sending : {WAV_FILE}")

with open(WAV_FILE, "rb") as f:
    response1 = requests.post(
        API1_URL,
        files={"file": ("sample_audio.wav", f, "audio/wav")},
        headers={"x-api-key": API1_KEY}
    )

if response1.status_code != 200:
    print(f"[ERROR] API 1 failed: {response1.status_code}")
    print(response1.json())
    exit(1)

result1   = response1.json()
features  = result1.get("features", {})

print(f"  Status         : {result1.get('status')}")
print(f"  Features count : {result1.get('feature_count')}")

# ── Step 2 — Call API 2 to get 24 PCA components ─────────────
print("\n" + "=" * 55)
print("  STEP 2 — PCA Pipeline (API 2)")
print("=" * 55)

payload = {"features": features}

response2 = requests.post(
    API2_URL,
    json=payload,
    headers={"x-api-key": API2_KEY}
)

result2 = response2.json()

print(f"  Status Code      : {response2.status_code}")
print(f"  Status           : {result2.get('status')}")
print(f"  Input features   : {result2.get('input_features')}")
print(f"  Selected features: {result2.get('selected_features')}")
print(f"  PCA components   : {result2.get('pca_components')}")
print(f"  Variance kept    : {result2.get('variance_kept')}%")

print("\n  PCA Components:")
for key, val in result2.get("components", {}).items():
    print(f"    {key}: {val}")

# ── Save output ───────────────────────────────────────────────
with open("output.json", "w") as f:
    json.dump(result2, f, indent=2)

print("\n  Full output saved to output.json")
print("=" * 55)
