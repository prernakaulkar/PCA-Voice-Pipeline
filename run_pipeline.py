"""
run_pipeline.py
───────────────
Master runner for all 5 pipeline stages.

Usage:
  python run_pipeline.py              # runs all 5 stages
  python run_pipeline.py --stage 1    # runs only stage 1
  python run_pipeline.py --stage 1 2  # runs stages 1 and 2
"""

import sys
import time
import argparse
import importlib.util
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

STAGES = {
    1: ("Feature extraction   →  step1_features_raw.csv",         "step1_extract"),
    2: ("Articulation filter  →  step2_features_articulation.csv", "step2_articulation"),
    3: ("Variance threshold   →  step3_features_variance.csv",     "step3_variance"),
    4: ("SMOTE oversampling   →  step4_features_smote.csv",        "step4_smote"),
    5: ("StandardScaler + PCA →  step5_features_pca.csv",          "step5_pca"),
}

HEADER = """
════════════════════════════════════════════════════════════
  PCA Voice Pipeline — Mental Health Audio Processing
════════════════════════════════════════════════════════════
  Stage 1  Feature extraction   (OpenSMILE)
  Stage 2  Articulation filter  (6373 → ~2373)
  Stage 3  Variance threshold   (~2373 → ~1500)
  Stage 4  SMOTE oversampling   (714 → 10002)
  Stage 5  Scaler + PCA         (~1500 → 60-100 components)
════════════════════════════════════════════════════════════
"""


def load_and_run(module_name):
    module_path = SRC_DIR / f"{module_name}.py"
    spec   = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.run()


def run_stage(n):
    label, module_name = STAGES[n]
    bar = "▓" * 60
    print(f"\n{bar}")
    print(f"  RUNNING STAGE {n} — {label}")
    print(f"{bar}")
    t0 = time.time()
    load_and_run(module_name)
    elapsed = time.time() - t0
    print(f"\n  Stage {n} completed in {elapsed:.1f} seconds\n")


def main():
    print(HEADER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", nargs="+", type=int,
                        choices=list(STAGES.keys()),
                        help="Which stage(s) to run. Default = all.")
    args = parser.parse_args()

    stages_to_run = args.stage if args.stage else list(STAGES.keys())

    t_total = time.time()
    for s in sorted(stages_to_run):
        run_stage(s)

    total_time = time.time() - t_total
    print(f"""
════════════════════════════════════════════════════════════
  ALL DONE  —  total time: {total_time:.1f}s
════════════════════════════════════════════════════════════""")


if __name__ == "__main__":
    main()
