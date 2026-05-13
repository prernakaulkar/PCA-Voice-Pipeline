"""
step3b_feature_engineering.py
──────────────────────────────
Reduces 1736 features down to ~120 using 3 techniques:

  A) Correlation filter     → removes duplicate features (corr > 0.95)
  B) SelectKBest ANOVA      → keeps top 200 by F-score
  C) Random Forest importance → keeps top 120 by importance

Target: ~120 features → PCA → ~50 components

Input  : output/step3_features_variance.csv
Output : output/step3b_features_engineered.csv
Saves  : models/feature_engineering_cols.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from config import OUTPUT_DIR, MODELS_DIR, CORRELATION_THRESHOLD, ANOVA_TOP_K, RF_TOP_N


def remove_correlated(df, feature_cols, threshold):
    """Remove features with correlation above threshold."""
    print(f"\n  [A] Correlation filter (threshold={threshold})")
    X = df[feature_cols]
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in feature_cols if c not in to_drop]
    print(f"     Removed (correlated) : {len(to_drop)}")
    print(f"     Remaining            : {len(kept)}")
    return kept


def anova_selection(df, feature_cols, y, k):
    """Keep top K features by ANOVA F-score."""
    print(f"\n  [B] SelectKBest ANOVA (top {k})")
    k = min(k, len(feature_cols))
    X = df[feature_cols].values
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    kept = [c for c, m in zip(feature_cols, mask) if m]
    print(f"     Remaining            : {len(kept)}")
    return kept


def rf_selection(df, feature_cols, y, n):
    """Keep top N features by Random Forest importance."""
    print(f"\n  [C] Random Forest importance (top {n})")
    n = min(n, len(feature_cols))
    X = df[feature_cols].values
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    kept = [feature_cols[i] for i in sorted(indices)]
    print(f"     Remaining            : {len(kept)}")
    return kept


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step3_features_variance.csv"
    if not in_path.exists():
        print(f"  [ERROR] {in_path} not found. Run step3 first.")
        return

    print(f"  Loading: {in_path}")
    df = pd.read_csv(in_path)

    meta_cols    = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Input features : {len(feature_cols)}")

    le = LabelEncoder()
    y  = le.fit_transform(df["label"].values)

    # A — Correlation filter
    feature_cols = remove_correlated(df, feature_cols, CORRELATION_THRESHOLD)

    # B — ANOVA SelectKBest
    feature_cols = anova_selection(df, feature_cols, y, ANOVA_TOP_K)

    # C — Random Forest importance
    feature_cols = rf_selection(df, feature_cols, y, RF_TOP_N)

    # Save selected column names
    pkl_path = MODELS_DIR / "feature_engineering_cols.pkl"
    joblib.dump(feature_cols, pkl_path)

    # Save output CSV
    keep_meta = [c for c in meta_cols if c in df.columns]
    out_df    = df[keep_meta + feature_cols]
    out_path  = OUTPUT_DIR / "step3b_features_engineered.csv"
    out_df.to_csv(out_path, index=False)

    print(f"""
============================================================
  STEP 3b COMPLETE — Feature Engineering
============================================================
  Input features   : {len([c for c in df.columns if c not in meta_cols])}
  After corr filter: ~{len(feature_cols) + RF_TOP_N - RF_TOP_N} (see above)
  After ANOVA      : {ANOVA_TOP_K}
  Final features   : {len(feature_cols)}
  Output shape     : {out_df.shape}
  Saved to         : {out_path}
  Model saved      : {pkl_path}
  Expected PCA out : ~50 components
============================================================""")


if __name__ == "__main__":
    run()
