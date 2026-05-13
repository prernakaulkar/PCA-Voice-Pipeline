"""
step4_smote.py
──────────────
Balances all 6 conditions to 1667 samples each using SMOTE.
Now reads from step3b (feature engineered) output.

Input  : output/step3b_features_engineered.csv
Output : output/step4_features_smote.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from config import OUTPUT_DIR, CONDITIONS, SMOTE_TARGET, SMOTE_K_NEIGHBORS


def run():
    in_path = OUTPUT_DIR / "step3b_features_engineered.csv"
    if not in_path.exists():
        print(f"  [ERROR] {in_path} not found. Run step3b first.")
        return

    print(f"  Loading: {in_path}")
    df = pd.read_csv(in_path)

    meta_cols    = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values
    y = df["label"].values

    print(f"\n  Input shape : {df.shape}")
    print(f"  Class distribution BEFORE SMOTE:")
    for cond in CONDITIONS:
        count = (y == cond).sum()
        bar   = "█" * (count // 5)
        print(f"    {cond:<12} {count:>4}  {bar}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    sampling_strategy = {
        le.transform([c])[0]: SMOTE_TARGET
        for c in CONDITIONS
        if c in le.classes_
    }

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=42,
    )

    print(f"\n  Running SMOTE (target: {SMOTE_TARGET} per class)...")
    X_res, y_res = smote.fit_resample(X, y_enc)
    y_labels     = le.inverse_transform(y_res)

    out_df = pd.DataFrame(X_res, columns=feature_cols)
    out_df.insert(0, "label", y_labels)

    out_path = OUTPUT_DIR / "step4_features_smote.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\n  Class distribution AFTER SMOTE:")
    for cond in CONDITIONS:
        count = (y_labels == cond).sum()
        bar   = "█" * (count // 50)
        print(f"    {cond:<12} {count:>4}  {bar}")

    print(f"""
============================================================
  STEP 4 COMPLETE — SMOTE oversampling
============================================================
  Real samples     : {len(df)}
  Synthetic added  : {len(out_df) - len(df)}
  Total samples    : {len(out_df)}
  Features         : {len(feature_cols)}
  Output shape     : {out_df.shape}
  Saved to         : {out_path}
============================================================""")


if __name__ == "__main__":
    run()
