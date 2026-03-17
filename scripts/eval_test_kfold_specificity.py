from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.project.data import build_test_dataloader, infer_classes_from_data
from src.project.models import create_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoint of each fold on test split and include specificity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
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
        help="Optional directory to save metrics. Default: <run_dir>/test_eval_specificity",
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
    best = list(fold_dir.glob("baseline-epoch=*.ckpt"))
    if not best:
        raise FileNotFoundError(f"No best checkpoint found under {fold_dir}")
    if len(best) == 1:
        return best[0]
    return max(best, key=lambda p: p.stat().st_mtime)


def load_checkpoint_model(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(obj["hyper_parameters"])

    inferred_classes = infer_classes_from_data(cfg, repo_root)
    with open_dict(cfg):
        cfg.data.classes = inferred_classes
        cfg.model.num_classes = len(inferred_classes)

    model = create_model_from_config(cfg)
    state_dict = {
        key.removeprefix("model."): value
        for key, value in obj["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, cfg


def specificity_from_confusion_matrix(cm: np.ndarray) -> float:
    total = float(cm.sum())
    if total <= 0:
        return 0.0

    specificities: list[float] = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        tn = total - tp - fp - fn
        denom = tn + fp
        specificities.append((tn / denom) if denom > 0 else 0.0)
    return float(sum(specificities) / len(specificities))


def evaluate_model(model, loader, device: torch.device) -> dict[str, Any]:
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-learn is required for test evaluation.") from exc

    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    loss_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            batch_size = int(y.shape[0])
            loss_sum += float(loss.item()) * batch_size
            num_samples += batch_size

            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())

    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()

    cm = confusion_matrix(y_true, preds, labels=list(range(probs.shape[1])))

    metrics: dict[str, Any] = {}
    metrics["num_samples"] = int(len(y_true))
    metrics["loss"] = float(loss_sum / max(num_samples, 1))
    metrics["acc"] = float(accuracy_score(y_true, preds))
    metrics["precision_macro"] = float(precision_score(y_true, preds, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, preds, average="macro", zero_division=0))
    metrics["specificity_macro"] = float(specificity_from_confusion_matrix(cm))
    metrics["f1_macro"] = float(f1_score(y_true, preds, average="macro", zero_division=0))

    if probs.shape[1] == 2 and len(set(y_true.tolist())) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    elif probs.shape[1] > 2 and len(set(y_true.tolist())) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    else:
        metrics["auc"] = None

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_folds": len(results)}
    for key in ("loss", "acc", "precision_macro", "recall_macro", "specificity_macro", "f1_macro", "auc"):
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

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "test_eval_specificity")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for fold_idx in fold_indices:
        fold_dir = run_dir / f"fold_{fold_idx}"
        ckpt_path = find_best_checkpoint(fold_dir)
        model, cfg = load_checkpoint_model(ckpt_path, device)

        if args.batch_size is not None:
            with open_dict(cfg):
                cfg.data.batch_size = args.batch_size
        if args.num_workers is not None:
            with open_dict(cfg):
                cfg.data.num_workers = args.num_workers

        test_loader = build_test_dataloader(cfg, repo_root)
        metrics = evaluate_model(model, test_loader, device)
        metrics["fold"] = fold_idx
        metrics["checkpoint"] = str(ckpt_path.relative_to(repo_root))
        metrics["test_dir"] = str((repo_root / cfg.data.test_dir).resolve().relative_to(repo_root))
        results.append(metrics)

        print("=" * 70)
        print(f"fold_{fold_idx}")
        print(f"checkpoint   : {metrics['checkpoint']}")
        print(f"test_dir     : {metrics['test_dir']}")
        print(f"samples      : {metrics['num_samples']}")
        print(f"loss         : {metrics['loss']:.4f}")
        print(f"acc          : {metrics['acc']:.4f}")
        print(f"precision    : {metrics['precision_macro']:.4f}")
        print(f"recall       : {metrics['recall_macro']:.4f}")
        print(f"specificity  : {metrics['specificity_macro']:.4f}")
        print(f"f1           : {metrics['f1_macro']:.4f}")
        print(f"auc          : {metrics['auc'] if metrics['auc'] is None else f'{metrics['auc']:.4f}'}")
        print(f"cm           : {metrics['confusion_matrix']}")

    summary = aggregate_results(results)
    payload = {"run_dir": str(run_dir.relative_to(repo_root)), "results": results, "summary": summary}

    json_path = output_dir / "test_metrics_specificity.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_dir / "test_metrics_specificity.csv"
    fieldnames = [
        "fold",
        "checkpoint",
        "test_dir",
        "num_samples",
        "loss",
        "acc",
        "precision_macro",
        "recall_macro",
        "specificity_macro",
        "f1_macro",
        "auc",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for key in ("loss", "acc", "precision_macro", "recall_macro", "specificity_macro", "f1_macro", "auc"):
        value = summary.get(key)
        if value is None:
            print(f"{key:16s}: n/a")
        else:
            print(f"{key:16s}: {value['mean']:.4f} ± {value['std']:.4f}")
    if "confusion_matrix_sum" in summary:
        print(f"confusion_matrix_sum: {summary['confusion_matrix_sum']}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
