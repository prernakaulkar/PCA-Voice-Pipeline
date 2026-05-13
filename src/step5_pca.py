"""
step5_pca.py
────────────
Normalises features with StandardScaler then compresses to
PCA components at 95% explained variance.

Input  : output/step4_features_smote.csv
Output : output/step5_features_pca.csv
         output/pca_variance_plot.png
Saves  : models/scaler.pkl
         models/pca.pkl
         models/label_encoder.pkl
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from config import OUTPUT_DIR, MODELS_DIR, PCA_VARIANCE


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    in_path = OUTPUT_DIR / "step4_features_smote.csv"
    if not in_path.exists():
        print(f"  [ERROR] {in_path} not found. Run step4 first.")
        return

    print(f"  Loading: {in_path}")
    df = pd.read_csv(in_path)

    feature_cols = [c for c in df.columns if c != "label"]
    X            = df[feature_cols].values
    y            = df["label"].values

    print(f"  Input shape : {df.shape}")

    # ── Label encoder ────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    print(f"  Classes     : {list(le.classes_)}")

    # ── StandardScaler ───────────────────────────────────────
    print("  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"  Scaler saved to  : {MODELS_DIR / 'scaler.pkl'}")

    # ── PCA ──────────────────────────────────────────────────
    print(f"  Fitting PCA (variance={PCA_VARIANCE})...")
    pca = PCA(n_components=PCA_VARIANCE, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, MODELS_DIR / "pca.pkl")
    print(f"  PCA components  : {pca.n_components_}")
    print(f"  Variance kept   : {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA saved to    : {MODELS_DIR / 'pca.pkl'}")

    # ── Variance plot ─────────────────────────────────────────
    cumvar   = np.cumsum(pca.explained_variance_ratio_) * 100
    plot_path = OUTPUT_DIR / "pca_variance_plot.png"
    plt.figure(figsize=(10, 4))
    plt.plot(cumvar, color="#1D9E75", linewidth=2)
    plt.axhline(y=95, color="#D85A30", linestyle="--", label="95% threshold")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("PCA — cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Variance plot   : {plot_path}")

    # ── Save PCA CSV ─────────────────────────────────────────
    pca_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    out_df   = pd.DataFrame(X_pca, columns=pca_cols)
    out_df.insert(0, "label", y)
    out_path = OUTPUT_DIR / "step5_features_pca.csv"
    out_df.to_csv(out_path, index=False)

    print(f"""
============================================================
  STEP 5 COMPLETE — StandardScaler + PCA
============================================================
  Input  shape     : {X.shape}
  PCA components   : {pca.n_components_}
  Output shape     : {out_df.shape}
  Variance kept    : {pca.explained_variance_ratio_.sum()*100:.2f}%
  Saved to         : {out_path}
  Models saved:
    scaler.pkl
    pca.pkl
    label_encoder.pkl
============================================================""")


if __name__ == "__main__":
    run()
