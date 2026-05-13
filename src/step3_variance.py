"""
step3_variance.py
─────────────────
Removes near-constant features using VarianceThreshold.
Features that barely change across all 714 samples carry no signal.

Input  : output/step2_features_articulation.csv
Output : output/step3_features_variance.csv
Saves  : models/variance_threshold.pkl
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold
from config import OUTPUT_DIR, MODELS_DIR, VARIANCE_THRESHOLD


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step2_features_articulation.csv"
    if not in_path.exists():
        print(f"  [ERROR] {in_path} not found. Run step2 first.")
        return

    print(f"  Loading: {in_path}")
    df = pd.read_csv(in_path)

    meta_cols    = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X            = df[feature_cols].values

    print(f"  Features before threshold : {len(feature_cols)}")

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(X)

    kept_mask  = selector.get_support()
    keep_cols  = [c for c, k in zip(feature_cols, kept_mask) if k]
    removed    = len(feature_cols) - len(keep_cols)

    print(f"  Removed (near-constant)   : {removed}")
    print(f"  Features after threshold  : {len(keep_cols)}")

    pkl_path = MODELS_DIR / "variance_threshold.pkl"
    joblib.dump(selector, pkl_path)
    print(f"  Model saved to            : {pkl_path}")

    out_df   = df[meta_cols + keep_cols]
    out_path = OUTPUT_DIR / "step3_features_variance.csv"
    out_df.to_csv(out_path, index=False)

    print(f"""
============================================================
  STEP 3 COMPLETE — Variance threshold
============================================================
  Input features   : {len(feature_cols)}
  Kept features    : {len(keep_cols)}
  Removed          : {removed}
  Output shape     : {out_df.shape}
  Saved to         : {out_path}
  Model saved      : {pkl_path}
============================================================""")


if __name__ == "__main__":
    run()
