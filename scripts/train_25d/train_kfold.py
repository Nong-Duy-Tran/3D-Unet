from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts._kfold_common import add_kfold_args, run_kfold


def main() -> int:
    parser = add_kfold_args(
        argparse.ArgumentParser(),
        default_config_name="train_25d_oasis",
        description="Run 2.5D k-fold training sequentially.",
    )
    args = parser.parse_args()
    return run_kfold(args, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
