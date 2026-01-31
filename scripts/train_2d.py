from __future__ import annotations

import random
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import lightning.pytorch as pl
import timm

from src.baseline.models.deit2d_classifier import DeiT2DClassifier
from src.baseline.models.simple_unet2d_classifier import SimpleUNet2DClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class MRIVolumeStackDataset(Dataset):
    def __init__(
        self,
        subjects: list[tuple[str, list[str], int]],
        num_slices: int,
        strategy: str,
        transform: T.Compose | None = None,
    ):
        self.subjects = subjects
        self.num_slices = num_slices
        self.strategy = strategy
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        _, paths, label = self.subjects[idx]
        if self.strategy == "random":
            chosen = random.sample(paths, k=min(self.num_slices, len(paths)))
        elif self.strategy == "center":
            chosen = _center_slices(paths, self.num_slices)
        else:
            chosen = _uniform_slices(paths, self.num_slices)

        slice_tensors = []
        for p in chosen:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            slice_tensors.append(img)
        x = torch.stack(slice_tensors, dim=0)
        return x, label


_SUBJECT_RE = re.compile(r"(.*)_(ax|sag|cor)_\d+\.jpg$", re.IGNORECASE)


def _subject_id_from_path(path: Path) -> str:
    m = _SUBJECT_RE.match(path.name)
    if m:
        return m.group(1)
    return path.name.rsplit("_s", 1)[0]


def _uniform_slices(paths: list[str], num_slices: int) -> list[str]:
    if len(paths) <= num_slices:
        return paths
    idxs = np.linspace(0, len(paths) - 1, num_slices)
    return [paths[int(round(i))] for i in idxs]


def _center_slices(paths: list[str], num_slices: int) -> list[str]:
    if len(paths) <= num_slices:
        return paths
    center = len(paths) // 2
    half = num_slices // 2
    start = max(0, center - half)
    end = min(len(paths), start + num_slices)
    return paths[start:end]


def get_subject_split_stacks(root_dir: str, classes: list[str], test_size: float = 0.2):
    train_subjects, test_subjects = [], []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = Path(root_dir) / cls_name
        all_files = list(cls_dir.glob("*.jpg"))
        if not all_files:
            continue

        subjects: dict[str, list[Path]] = {}
        for f in all_files:
            subject_id = _subject_id_from_path(f)
            subjects.setdefault(subject_id, []).append(f)

        sub_ids = list(subjects.keys())
        tr_ids, te_ids = train_test_split(
            sub_ids,
            test_size=test_size,
            random_state=42,
            stratify=[cls_name] * len(sub_ids),
        )

        for s_id in tr_ids:
            paths = sorted(str(p) for p in subjects[s_id])
            train_subjects.append((s_id, paths, cls_idx))
        for s_id in te_ids:
            paths = sorted(str(p) for p in subjects[s_id])
            test_subjects.append((s_id, paths, cls_idx))

    return train_subjects, test_subjects


def _resolve_ckpt_prefix(cfg: DictConfig) -> str:
    if cfg.model.name == "simpleunet2d":
        return "simpleunet2d"
    return "deit2d"


def _resolve_ckpt_dir(cfg: DictConfig) -> Path:
    base = Path(cfg.checkpoint.dir)
    return base / _resolve_ckpt_prefix(cfg)


class MetricsHistory(pl.Callback):
    def __init__(self) -> None:
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.val_auc = []

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D401
        metrics = trainer.callback_metrics
        if "train/loss" in metrics:
            self.train_loss.append(float(metrics["train/loss"]))
        if "train/acc" in metrics:
            self.train_acc.append(float(metrics["train/acc"]))

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: D401
        metrics = trainer.callback_metrics
        if "val/loss" in metrics:
            self.val_loss.append(float(metrics["val/loss"]))
        if "val/acc" in metrics:
            self.val_acc.append(float(metrics["val/acc"]))
        if "val/auc_epoch" in metrics:
            self.val_auc.append(float(metrics["val/auc_epoch"]))


@torch.no_grad()
def evaluate_subjects(model, loader, device, class_names):
    model.eval()
    all_probs = []
    all_labels = []
    for x, y in tqdm(loader, desc="[Subject Eval]"):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=1)

    if probs.shape[1] > 2:
        auc_score = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    else:
        auc_score = roc_auc_score(labels, probs[:, 1])

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names)
    return probs, labels, preds, auc_score, cm, report


def save_plots(output_dir, history: MetricsHistory, labels, preds, probs, class_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if history.train_loss:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.train_loss, label="Train Loss")
        if history.val_loss:
            plt.plot(history.val_loss, label="Val Loss")
        plt.title("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.subplot(1, 2, 2)
        plt.plot(history.train_acc, label="Train Acc")
        if history.val_acc:
            plt.plot(history.val_acc, label="Val Acc")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_acc.png", dpi=150)
        plt.close()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    if probs.shape[1] > 2:
        y_bin = label_binarize(labels, classes=list(range(len(class_names))))
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(figsize=(5, 4))
        for i, name in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC (OvR)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close()
    else:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close()


@hydra.main(config_path="../configs/baseline", config_name="train_2d_jpg", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    train_transform = T.Compose([
        T.Resize((cfg.data.image_size, cfg.data.image_size)),
        T.RandAugment(num_ops=cfg.data.randaugment_ops, magnitude=cfg.data.randaugment_mag),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((cfg.data.image_size, cfg.data.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_subjects, val_subjects = get_subject_split_stacks(
        cfg.data.root_dir,
        list(cfg.data.classes),
        cfg.data.test_size,
    )
    train_ds = MRIVolumeStackDataset(
        train_subjects,
        num_slices=cfg.data.num_slices,
        strategy=cfg.data.strategy,
        transform=train_transform,
    )
    val_ds = MRIVolumeStackDataset(
        val_subjects,
        num_slices=cfg.data.num_slices,
        strategy=cfg.data.val_strategy,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    steps_per_epoch = max(1, len(train_loader))
    lit_model = _LightningModule(cfg, steps_per_epoch)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(_resolve_ckpt_dir(cfg)),
            filename=f"{_resolve_ckpt_prefix(cfg)}-{{epoch:03d}}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
        ),
    ]
    history_cb = MetricsHistory()
    callbacks.append(history_cb)

    logger = None
    if cfg.logging.wandb.enabled:
        logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.logging.wandb.name,
            tags=list(cfg.logging.wandb.tags),
        )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if cfg.data.format == "jpg":
        ckpt_path = None
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_path = cb.best_model_path
                break
        if ckpt_path:
            lit_model = _LightningModule.load_from_checkpoint(ckpt_path, cfg=cfg, steps_per_epoch=steps_per_epoch)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit_model.to(device)
        class_names = list(cfg.data.classes)
        probs, labels, preds, auc_score, cm, report = evaluate_subjects(
            lit_model, val_loader, device, class_names
        )
        output_dir = getattr(cfg, "output_dir", "outputs/train_2d")
        save_plots(output_dir, history_cb, labels, preds, probs, class_names)
        report_path = Path(output_dir) / "classification_report.txt"
        report_path.write_text(report)


class _LightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, steps_per_epoch: int):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        if cfg.data.format == "jpg":
            if cfg.model.name == "simpleunet2d":
                self.model = SimpleUNet2DClassifier(
                    in_channels=3,
                    num_classes=cfg.model.num_classes,
                    base_features=cfg.model.base_features,
                    head_hidden=cfg.model.head_hidden,
                    dropout=cfg.model.dropout,
                    head_dropout=cfg.model.head_dropout,
                )
            else:
                self.model = DeiT2DClassifier(
                    timm_name=cfg.model.name,
                    pretrained=cfg.model.pretrained,
                    image_size=cfg.data.image_size,
                    in_channels=3,
                    num_classes=cfg.model.num_classes,
                    drop_path_rate=cfg.model.drop_path_rate,
                    attn_drop_rate=cfg.model.attn_drop_rate,
                    dropout=0.1,
                )
        else:
            self.model = timm.create_model(
                cfg.model.name,
                pretrained=cfg.model.pretrained,
                num_classes=cfg.model.num_classes,
                in_chans=3,
                drop_path_rate=cfg.model.drop_path_rate,
                attn_drop_rate=cfg.model.attn_drop_rate,
            )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        self.val_probs = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)
        self.val_probs.append(probs.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if not self.val_probs:
            return
        probs = torch.cat(self.val_probs).numpy()
        labels = torch.cat(self.val_labels).numpy()
        try:
            if probs.shape[1] > 2:
                auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            else:
                auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.0
        self.log("val/auc_epoch", auc, on_epoch=True, prog_bar=False)
        self.val_probs = []
        self.val_labels = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.training.lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.cfg.training.max_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == "__main__":
    main()
