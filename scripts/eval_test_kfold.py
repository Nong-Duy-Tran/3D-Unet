from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.baseline.data import get_dataloaders
from src.project.models import create_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoint of each fold on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="checkpoints/25d/resnet18_selfattn_pos_50sl_5folds",
        help="Checkpoint run directory containing fold_* subdirectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for test batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional override for dataloader workers.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help='Comma-separated fold indices (e.g. "0,1,2") or "all".',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to save metrics. Default: <run_dir>/test_eval",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_fold_indices(spec: str, run_dir: Path) -> list[int]:
    if spec.strip().lower() == "all":
        indices: list[int] = []
        for fold_dir in sorted(run_dir.glob("fold_*")):
            try:
                indices.append(int(fold_dir.name.split("_")[-1]))
            except ValueError:
                continue
        return indices
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def find_best_checkpoint(fold_dir: Path) -> Path:
    best = sorted(fold_dir.glob("baseline-epoch=*.ckpt"))
    if not best:
        raise FileNotFoundError(f"No best checkpoint found under {fold_dir}")
    return best[0]


def load_checkpoint_model(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(obj["hyper_parameters"])
    model = create_model_from_config(cfg)
    state_dict = {
        key.removeprefix("model."): value
        for key, value in obj["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, cfg


def infer_test_dir(cfg) -> str:
    val_dir = Path(str(cfg.data.val_dir))
    if val_dir.name != "val":
        raise ValueError(f"Expected val_dir to end with 'val', got: {val_dir}")
    return str(val_dir.parent / "test")


def build_test_loader(cfg, batch_size: int | None, num_workers: int | None):
    test_dir = infer_test_dir(cfg)
    _, test_loader = get_dataloaders(
        train_dir=test_dir,
        val_dir=test_dir,
        batch_size=batch_size or int(cfg.data.batch_size),
        num_workers=num_workers if num_workers is not None else int(cfg.data.num_workers),
        target_shape=tuple(getattr(cfg.data, "target_shape", (64, 64, 64))),
        weighted_sampler=False,
        use_2d=bool(getattr(cfg.data, "use_2d", False)),
        num_slices=int(getattr(cfg.data, "num_slices", 8)),
        slice_axis=int(getattr(cfg.data, "slice_axis", 0)),
        resize_2d=getattr(cfg.data, "resize_2d", None),
        slice_strategy_train=getattr(cfg.data, "slice_strategy_val", "uniform"),
        slice_strategy_val=getattr(cfg.data, "slice_strategy_val", "uniform"),
        imagenet_norm=bool(getattr(cfg.data, "imagenet_norm", False)),
        randaugment=False,
        randaugment_ops=int(getattr(cfg.data, "randaugment_ops", 2)),
        randaugment_mag=int(getattr(cfg.data, "randaugment_mag", 9)),
        rgb_mode=bool(getattr(cfg.data, "rgb_mode", False)),
        normalize=bool(getattr(cfg.data, "normalize", True)),
        data_format=str(getattr(cfg.data, "format", "nifti")),
        classes=list(getattr(cfg.data, "classes", [])) or None,
        class_map=getattr(cfg.data, "class_map", None),
        jpg_view=str(getattr(cfg.data, "jpg_view", "axial")),
        jpg_mode=str(getattr(cfg.data, "jpg_mode", "stack")),
        window_size=int(getattr(cfg.data, "window_size", 5)),
        image_size=getattr(cfg.data, "image_size", None),
        train_subject_ids=None,
        val_subject_ids=None,
    )
    return test_loader, test_dir


def evaluate_model(model, loader, device: torch.device) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for test evaluation.") from exc

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()

    metrics: dict[str, Any] = {}
    metrics["num_samples"] = int(len(y_true))
    metrics["acc"] = float(accuracy_score(y_true, preds))
    metrics["precision_macro"] = float(precision_score(y_true, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, preds, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, preds, average="macro", zero_division=0))

    if probs.shape[1] == 2 and len(set(y_true.tolist())) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    elif probs.shape[1] > 2 and len(set(y_true.tolist())) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    else:
        metrics["auc"] = None

    cm = confusion_matrix(y_true, preds, labels=list(range(probs.shape[1])))
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_folds": len(results)}
    for key in ("acc", "precision_macro", "recall_macro", "f1_macro", "auc"):
        values = [float(item[key]) for item in results if item.get(key) is not None]
        if not values:
            summary[key] = None
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    if results and all(item.get("confusion_matrix") is not None for item in results):
        cms = [np.asarray(item["confusion_matrix"], dtype=np.int64) for item in results]
        summary["confusion_matrix_sum"] = np.sum(cms, axis=0).tolist()
    return summary


def main() -> int:
    args = parse_args()
    run_dir = (repo_root / args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = resolve_device(args.device)
    fold_indices = parse_fold_indices(args.folds, run_dir)
    if not fold_indices:
        raise ValueError(f"No folds resolved from --folds={args.folds}")

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "test_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for fold_idx in fold_indices:
        fold_dir = run_dir / f"fold_{fold_idx}"
        ckpt_path = find_best_checkpoint(fold_dir)
        model, cfg = load_checkpoint_model(ckpt_path, device)
        test_loader, test_dir = build_test_loader(cfg, args.batch_size, args.num_workers)
        metrics = evaluate_model(model, test_loader, device)
        metrics["fold"] = fold_idx
        metrics["checkpoint"] = str(ckpt_path.relative_to(repo_root))
        metrics["test_dir"] = test_dir
        results.append(metrics)

        print("=" * 70)
        print(f"fold_{fold_idx}")
        print(f"checkpoint : {metrics['checkpoint']}")
        print(f"test_dir    : {metrics['test_dir']}")
        print(f"samples     : {metrics['num_samples']}")
        print(f"acc         : {metrics['acc']:.4f}")
        print(f"precision   : {metrics['precision_macro']:.4f}")
        print(f"recall      : {metrics['recall_macro']:.4f}")
        print(f"f1          : {metrics['f1_macro']:.4f}")
        print(f"auc         : {metrics['auc'] if metrics['auc'] is None else f'{metrics['auc']:.4f}'}")
        print(f"cm          : {metrics['confusion_matrix']}")

    summary = aggregate_results(results)
    payload = {"run_dir": str(run_dir.relative_to(repo_root)), "results": results, "summary": summary}

    json_path = output_dir / "test_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_dir / "test_metrics.csv"
    fieldnames = [
        "fold",
        "checkpoint",
        "test_dir",
        "num_samples",
        "acc",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc",
        "cm_00",
        "cm_01",
        "cm_10",
        "cm_11",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            cm = np.asarray(item["confusion_matrix"], dtype=np.int64)
            row = {
                "fold": item["fold"],
                "checkpoint": item["checkpoint"],
                "test_dir": item["test_dir"],
                "num_samples": item["num_samples"],
                "acc": item["acc"],
                "precision_macro": item["precision_macro"],
                "recall_macro": item["recall_macro"],
                "f1_macro": item["f1_macro"],
                "auc": item["auc"],
                "cm_00": int(cm[0, 0]) if cm.shape[0] > 0 and cm.shape[1] > 0 else "",
                "cm_01": int(cm[0, 1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else "",
                "cm_10": int(cm[1, 0]) if cm.shape[0] > 1 and cm.shape[1] > 0 else "",
                "cm_11": int(cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else "",
            }
            writer.writerow(row)

    print("=" * 70)
    print("Saved test evaluation")
    print(f"JSON : {json_path}")
    print(f"CSV  : {csv_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
