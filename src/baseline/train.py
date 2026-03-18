from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
from omegaconf import DictConfig

from src.project.training.runner import run_training


@hydra.main(config_path="../../configs/baseline", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
