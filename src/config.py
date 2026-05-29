from pathlib import Path

# ── Base directory ──────────────────────────────────────────
BASE_DIR = Path(r"D:\PCA_Voicepipeline")

# ── Data paths ──────────────────────────────────────────────
RAW_DIR        = BASE_DIR / "data" / "raw"
PATIENTS_DIR   = BASE_DIR / "data" / "patients"

# ── Output + model paths ────────────────────────────────────
OUTPUT_DIR     = BASE_DIR / "output"
MODELS_DIR     = BASE_DIR / "models"

# ── Conditions ──────────────────────────────────────────────
CONDITIONS = ["Normal", "Depression", "Anxiety", "Stress", "Bipolar", "Suicidal"]

# ── Audio ───────────────────────────────────────────────────
TARGET_SR = 16000

# ── Variance threshold ──────────────────────────────────────
VARIANCE_THRESHOLD = 0.01

# ── Feature Engineering (step3b) ────────────────────────────
CORRELATION_THRESHOLD = 0.95   # remove features correlated above this
ANOVA_TOP_K           = 200    # keep top 200 by ANOVA F-score
RF_TOP_N              = 120   # keep top 120 by RF importance

# ── SMOTE ───────────────────────────────────────────────────
SMOTE_TARGET      = 1667   # per condition → 1667 × 6 = 10002 total
SMOTE_K_NEIGHBORS = 5

# ── PCA ─────────────────────────────────────────────────────
PCA_VARIANCE = 0.95            # keep 95% variance → expect ~50 components
