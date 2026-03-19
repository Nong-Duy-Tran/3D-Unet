#!/usr/bin/env python
"""
Run CV5 training sequentially for all folds (Linux/Unix version)
Similar to train_cv5_fold.sh but implemented in Python
"""

import subprocess
import sys
from pathlib import Path

# Get the path to train.py
SCRIPT_DIR = Path(__file__).parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent    # project root
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "2d" / "train.py"

# Use current Python executable
PYTHON = sys.executable

print(f"[INFO] Python: {PYTHON}")
print(f"[INFO] Train script: {TRAIN_SCRIPT}")

if not TRAIN_SCRIPT.exists():
    print(f"[ERROR] Train script not found: {TRAIN_SCRIPT}")
    sys.exit(1)

NUM_FOLDS = 5

for fold in range(NUM_FOLDS):
    print(f"\n{'=' * 60}")
    print(f"Training fold {fold}")
    print(f"{'=' * 60}")
    
    cmd = [PYTHON, str(TRAIN_SCRIPT), "--fold", str(fold)]
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[WARNING] Fold {fold} exited with code {result.returncode}")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Training stopped by user at fold {fold}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Fold {fold} failed: {e}")
        sys.exit(1)
    
    print(f"[DONE] Fold {fold} completed")

print(f"\n{'=' * 60}")
print("All folds completed!")
print(f"{'=' * 60}")
