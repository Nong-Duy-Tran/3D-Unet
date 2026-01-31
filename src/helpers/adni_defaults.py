"""Defaults for ADNI NIfTI processing."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CSVS = [
    REPO_ROOT / "data/ADNI/abb_9_13_2025.csv",
    REPO_ROOT / "data/ADNI/bbc_9_18_2025.csv",
    REPO_ROOT / "data/ADNI/ecc_9_20_2025.csv",
]

DEFAULT_SOURCE_ROOT_MAP = {
    "abb": REPO_ROOT / "data/ADNI/abb/ADNI",
    "bbc": REPO_ROOT / "data/ADNI/bbc/ADNI",
    "ecc": REPO_ROOT / "data/ADNI/ecc/ADNI",
}

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/ADNI"

DEFAULT_SPLIT_RATIOS = "train=0.8,val=0.1,test=0.1"
DEFAULT_SOURCE_LABEL_MAP = "abb=2,bbc=1,ecc=0"
