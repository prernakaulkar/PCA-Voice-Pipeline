"""
run.py — Test runner for Voice Feature Extraction API
──────────────────────────────────────────────────────
Reads a .wav file, calls POST /extract, and prints the response.

Usage:
    python run.py

Requirements:
    - API must be running: uvicorn app:app --port 5800
    - .env file must exist with VOICE_EXTRACT_KEY set
    - Place a sample .wav file named sample_audio.wav in this folder
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL  = "http://127.0.0.1:5800/extract"
API_KEY  = os.getenv("VOICE_EXTRACT_KEY")
WAV_FILE = "sample_audio.wav"

if not os.path.exists(WAV_FILE):
    print(f"[ERROR] Audio file not found: {WAV_FILE}")
    print("Please place a .wav file named 'sample_audio.wav' in this folder.")
    exit(1)

print(f"Sending {WAV_FILE} to {API_URL} ...")

with open(WAV_FILE, "rb") as f:
    response = requests.post(
        API_URL,
        files={"file": (WAV_FILE, f, "audio/wav")},
        headers={"x-api-key": API_KEY}
    )

result = response.json()

print(f"\nStatus Code : {response.status_code}")
print(f"Status      : {result.get('status')}")
print(f"Filename    : {result.get('filename')}")
print(f"Features    : {result.get('feature_count')}")
print("\nFirst 5 features:")
features = result.get("features", {})
for i, (k, v) in enumerate(features.items()):
    if i >= 5:
        break
    print(f"  {k}: {v}")

with open("output.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nFull output saved to output.json")
