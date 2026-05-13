"""
step1_extract.py
────────────────
Extracts 6373 OpenSMILE ComParE 2016 features from every .wav file
in data/raw/[condition]/ folders.

Output: output/step1_features_raw.csv
        Rows    = total audio files
        Columns = filename, label, feat_0 ... feat_6372
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import opensmile
import pandas as pd
from tqdm import tqdm
from config import RAW_DIR, OUTPUT_DIR, CONDITIONS

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    all_rows = []
    total_ok  = 0
    total_err = 0

    for condition in CONDITIONS:
        folder = RAW_DIR / condition
        if not folder.exists():
            print(f"  [SKIP] {condition} folder not found")
            continue

        wav_files = sorted(folder.glob("*.wav"))
        if not wav_files:
            print(f"  [SKIP] {condition} — no .wav files found")
            continue

        print(f"\n  [{condition}]  {len(wav_files)} files")

        for wav_path in tqdm(wav_files, desc=f"  {condition}", ncols=70):
            try:
                features = smile.process_file(str(wav_path))
                row = features.iloc[0].to_dict()
                row["filename"] = wav_path.name
                row["label"]    = condition
                all_rows.append(row)
                total_ok += 1
            except Exception as e:
                print(f"\n  [ERROR] {wav_path.name}: {e}")
                total_err += 1

    if not all_rows:
        print("\n  No features extracted. Check your audio files.")
        return

    df = pd.DataFrame(all_rows)

    # move filename and label to front
    cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    df = df[cols]

    out_path = OUTPUT_DIR / "step1_features_raw.csv"
    df.to_csv(out_path, index=False)

    print(f"""
============================================================
  STEP 1 COMPLETE
============================================================
  Files processed  : {total_ok}
  Errors           : {total_err}
  Output shape     : {df.shape}
  Features per file: {df.shape[1] - 2}
  Saved to         : {out_path}
============================================================""")

if __name__ == "__main__":
    run()
