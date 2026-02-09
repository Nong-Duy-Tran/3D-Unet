from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _load_config(repo_root: Path, config_name: str):
    cfg_path = repo_root / "configs" / "baseline" / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run k-fold training sequentially.")
    parser.add_argument("--config-name", default="train_2d_oasis", help="Config name under configs/baseline/")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to run")
    parser.add_argument("--start-fold", type=int, default=0, help="Start fold index (inclusive)")
    parser.add_argument("--end-fold", type=int, default=None, help="End fold index (exclusive)")
    parser.add_argument(
        "--wandb-prefix",
        type=str,
        default=None,
        help="Optional wandb run name prefix (e.g., oasis_k5)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override(s), e.g. training.max_epochs=64",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(repo_root, args.config_name)

    base_ckpt = getattr(getattr(cfg, "checkpoint", None), "dir", None) or "checkpoints/kfold"
    base_out = getattr(cfg, "output_dir", None) or f"outputs/{args.config_name}"
    base_hydra = f"outputs/{args.config_name}/hydra"

    end_fold = args.end_fold if args.end_fold is not None else args.folds

    for fold in range(args.start_fold, end_fold):
        ckpt_dir = f"{base_ckpt}/fold_{fold}"
        out_dir = f"{base_out}/fold_{fold}"
        hydra_dir = f"{base_hydra}/fold_{fold}"

        overrides = [
            f"data.fold_index={fold}",
            f"checkpoint.dir={ckpt_dir}",
            f"output_dir={out_dir}",
            f"hydra.run.dir={hydra_dir}",
        ]
        if args.wandb_prefix:
            overrides.append(f"logging.wandb.name={args.wandb_prefix}_fold_{fold}")
        overrides.extend(args.override)

        cmd = [sys.executable, "-m", "src.baseline.train", "--config-name", args.config_name, *overrides]
        print("Running:", " ".join(cmd))
        if not args.dry_run:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            result = subprocess.run(cmd, check=False, cwd=repo_root, env=env)
            if result.returncode != 0:
                return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
