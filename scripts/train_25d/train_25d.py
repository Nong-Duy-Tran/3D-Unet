from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.project.training.runner import run_training


@hydra.main(config_path="../../configs/baseline", config_name="train_25d_oasis", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
