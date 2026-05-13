"""
step2_articulation.py
─────────────────────
Filters 6373 raw features down to ~2373 articulatory features.

Keeps  : features from vocal tract, larynx, lungs
Removes: RASTA auditory filters, psychoacoustic, rolloff variants,
         chroma, tonnetz, and person-specific absolute mean features

Output : output/step2_features_articulation.csv
Saves  : models/articulation_cols.pkl  (list of kept column names)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import joblib
from config import OUTPUT_DIR, MODELS_DIR

# ── LLD groups to REMOVE entirely (non-articulatory) ────────
REMOVE_LLD_PATTERNS = [
    "audSpec_Rfilt",          # RASTA auditory filterbank (psychoacoustic)
    "audspecRasta",           # RASTA norm
    "pcm_fftMag_spectralRollOff",   # rolloff variants
    "pcm_fftMag_spectralKurtosis",  # statistical shape, not articulatory
    "pcm_fftMag_spectralSkewness",
    "pcm_fftMag_spectralVariance",
    "pcm_fftMag_psySharpness",      # psychoacoustic sharpness
]

# ── Person-specific functionals to remove ───────────────────
# These encode WHO is speaking (gender, age, vocal cord size)
# not WHAT condition they have
PERSON_SPECIFIC_SUFFIX = "_amean"
PERSON_SPECIFIC_LLDS   = [
    "F0final_sma",
    "jitterLocal_sma",
    "jitterDDP_sma",
    "shimmerLocal_sma",
    "logHNR_sma",
    "pcm_RMSenergy_sma",
    "mfcc_sma[1]",
    "mfcc_sma[2]",
    "mfcc_sma[3]",
    "mfcc_sma[4]",
    "mfcc_sma[5]",
    "mfcc_sma[6]",
    "mfcc_sma[7]",
    "mfcc_sma[8]",
    "mfcc_sma[9]",
    "mfcc_sma[10]",
    "mfcc_sma[11]",
    "mfcc_sma[12]",
    "mfcc_sma[13]",
    "mfcc_sma[14]",
]


def is_person_specific(col: str) -> bool:
    if not col.endswith(PERSON_SPECIFIC_SUFFIX):
        return False
    for lld in PERSON_SPECIFIC_LLDS:
        if lld in col:
            return True
    return False


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step1_features_raw.csv"
    if not in_path.exists():
        print(f"  [ERROR] {in_path} not found. Run step1 first.")
        return

    print(f"  Loading: {in_path}")
    df = pd.read_csv(in_path)

    meta_cols    = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"  Features before filter : {len(feature_cols)}")

    # remove non-articulatory LLD groups
    removed_pattern = [
        c for c in feature_cols
        if any(p in c for p in REMOVE_LLD_PATTERNS)
    ]

    # remove person-specific absolute means
    removed_person = [
        c for c in feature_cols
        if c not in removed_pattern and is_person_specific(c)
    ]

    remove_set  = set(removed_pattern + removed_person)
    keep_cols   = [c for c in feature_cols if c not in remove_set]

    print(f"  Removed (non-articulatory): {len(removed_pattern)}")
    print(f"  Removed (person-specific) : {len(removed_person)}")
    print(f"  Features after filter     : {len(keep_cols)}")

    # save the kept column names for use in API later
    pkl_path = MODELS_DIR / "articulation_cols.pkl"
    joblib.dump(keep_cols, pkl_path)
    print(f"  Saved column list to      : {pkl_path}")

    out_df   = df[meta_cols + keep_cols]
    out_path = OUTPUT_DIR / "step2_features_articulation.csv"
    out_df.to_csv(out_path, index=False)

    print(f"""
============================================================
  STEP 2 COMPLETE — Articulation filter
============================================================
  Input features   : {len(feature_cols)}
  Kept features    : {len(keep_cols)}
  Output shape     : {out_df.shape}
  Saved to         : {out_path}
  Model saved      : {pkl_path}
============================================================""")


if __name__ == "__main__":
    run()
