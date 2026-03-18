"""
Evaluation script for 2D CNN + Attention Alzheimer's classification
Uses Backbone + MultiAttention (models.py) + OASIS2DDataset (dataset.py)
"""
import os
import sys
import math
import random
import argparse
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

sys.path.insert(0, os.path.dirname(__file__))
from models import MyDenseNetMultiAttention, MyMobileNetMultiAttention
from dataset import OASIS2DDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIEW_SLICE_MAP   = {'all': 240, 'axial': 80, 'coronal': 80, 'sagittal': 80}
VIEW_IDX        = {'all': slice(None), 'axial': slice(None), 'coronal': slice(None), 'sagittal': slice(None)}
VIEW_SLICE_LBLS = {
    'all':      [f'Ax{i}' for i in range(80)] + [f'Co{i}' for i in range(80)] + [f'Sa{i}' for i in range(80)],
    'axial':    [f'Ax{i}' for i in range(80)],
    'coronal':  [f'Co{i}' for i in range(80)],
    'sagittal': [f'Sa{i}' for i in range(80)],
}
OUTCOME_DIRS = {
    (1, 1): 'TP_Alzheimer_correct',
    (1, 0): 'FN_Alzheimer_pred_Normal',
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_attention_grid(images_np, attn_scores, subject_id, label, pred,
                        save_path, view='all', ncols=8, colormap='Reds'):
    """
    Overlay per-slice attention weights on grayscale MRI slices and save as a grid.

    images_np  : (S, H, W) numpy array  – raw MRI pixel values
    attn_scores: (S,)      numpy array  – per-slice attention weight
    """
    S = images_np.shape[0]
    nrows = math.ceil(S / ncols)
    slice_labels = VIEW_SLICE_LBLS.get(view, [str(i) for i in range(S)])

    # Normalize attention to [0, 1]
    a_min, a_max = attn_scores.min(), attn_scores.max()
    attn_norm = (attn_scores - a_min) / (a_max - a_min + 1e-8)

    cmap_heat = plt.get_cmap(colormap)  # e.g. 'Reds', 'plasma', 'magma'

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.8, nrows * 2.6))
    plt.subplots_adjust(hspace=0.55, wspace=0.08)
    axes = np.array(axes).flatten()

    for i in range(S):
        ax = axes[i]
        img = images_np[i].astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rgb = np.stack([img_norm] * 3, axis=-1)              # (H, W, 3) grayscale→RGB

        heat_rgb = np.array(cmap_heat(attn_norm[i])[:3])    # (3,) colormap color
        alpha = attn_norm[i] * 0.55                          # max 55% overlay
        overlay = (1 - alpha) * rgb + alpha * heat_rgb[np.newaxis, np.newaxis, :]
        overlay = np.clip(overlay, 0, 1)

        ax.imshow(overlay)
        ax.set_title(f'{slice_labels[i]}\n{attn_norm[i]:.2f}', fontsize=5)
        ax.axis('off')

        # Red border for top-attended slices
        if attn_norm[i] > 0.75:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
                spine.set_visible(True)

    for i in range(S, len(axes)):
        axes[i].set_visible(False)

    true_str = 'Alzheimer' if label == 1 else 'Normal'
    pred_str = 'Alzheimer' if pred  == 1 else 'Normal'
    correct  = '✓ CORRECT' if label == pred else '✗ WRONG'
    color    = 'green'     if label == pred else 'red'
    fig.suptitle(f'{subject_id}  |  True: {true_str}  |  Pred: {pred_str}  |  {correct}',
                 fontsize=9, fontweight='bold', color=color, y=1.005)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap save helpers
# ---------------------------------------------------------------------------

def _slice_view_info(global_idx, view):
    """
    Resolve the anatomical view and local slice index for a given global slice index.

    For 'all' view (80 slices): axial [0-15], coronal [16-47], sagittal [48-79].
    For single-view inputs the view and local index are identical to the global index.

    Returns:
        tuple(str, int): (view_name, local_index_within_view)
    """
    if view == 'all':
        if global_idx < 80:
            return 'axial', int(global_idx)
        elif global_idx < 160:
            return 'coronal', int(global_idx - 80)
        else:
            return 'sagittal', int(global_idx - 160)
    return view, int(global_idx)


def save_val_sample_heatmaps(dataset, labels, preds, attn_array, subject_ids,
                              save_dir, view='all', n_samples=3, seed=0):
    """
    Save attention-grid overlays and info.json for a random sample of patients
    from the validation split.

    Args:
        dataset    : OASIS2DDataset instance.
        labels     : list[int] – ground-truth labels.
        preds      : list[int] – predicted labels.
        attn_array : np.ndarray of shape (N, S) – per-slice attention weights.
        subject_ids: list[str] – subject ID strings.
        save_dir   : Path – root output directory for this fold.
        view       : str  – one of 'all', 'axial', 'coronal', 'sagittal'.
        n_samples  : int  – number of random patients to visualise.
        seed       : int  – RNG seed for reproducible sampling.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=min(n_samples, len(labels)), replace=False)
    heatmap_dir = save_dir / 'sample_heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        sid = subject_ids[idx]
        images_np = dataset.images[idx][VIEW_IDX.get(view, slice(None))]
        make_attention_grid(images_np, attn_array[idx], sid,
                            int(labels[idx]), int(preds[idx]),
                            heatmap_dir / f'{sid}_attn.png', view=view)
        plot_patient_attention_distribution(
            attn_array[idx],
            heatmap_dir / f'{sid}_attention_distribution.png',
            view=view,
            subject_id=sid,
            label=int(labels[idx]),
            pred=int(preds[idx]),
            all_attn_array=attn_array,
            all_labels=labels,
        )

        # Build top-10 slice list with view and local-index annotations
        top10_global = np.argsort(attn_array[idx])[::-1][:10].tolist()
        top10_info = [
            {'global_index': g, 'view': _slice_view_info(g, view)[0],
             'local_index': _slice_view_info(g, view)[1]}
            for g in top10_global
        ]
        with open(heatmap_dir / f'{sid}_info.json', 'w') as f:
            json.dump({
                'subject_id': sid,
                'true_label': int(labels[idx]),
                'predicted':  int(preds[idx]),
                'top10_slices': top10_info,
            }, f, indent=2)
    print(f'  Saved {len(indices)} sample heatmaps \u2192 {heatmap_dir}')


def save_test_heatmaps(dataset, labels, preds, attn_array, subject_ids,
                       save_dir, view='all'):
    """
    Save per-patient attention-grid overlays and info.json files organised
    into TP / TN / FP / FN outcome folders (test split).

    Args:
        dataset    : OASIS2DDataset instance.
        labels     : list[int] – ground-truth labels.
        preds      : list[int] – predicted labels.
        attn_array : np.ndarray of shape (N, S) – per-slice attention weights.
        subject_ids: list[str] – subject ID strings.
        save_dir   : Path – root output directory for this fold.
        view       : str  – one of 'all', 'axial', 'coronal', 'sagittal'.
    """
    per_patient_dir = save_dir / 'per_patient'
    counts = {k: 0 for k in OUTCOME_DIRS}

    for idx in tqdm(range(len(labels)), desc='Saving per-patient heatmaps'):
        key = (int(labels[idx]), int(preds[idx]))
        # Skip if outcome not in OUTCOME_DIRS (only TP and FN, skip TN and FP)
        if key not in OUTCOME_DIRS:
            continue
        sid = subject_ids[idx]
        patient_dir = per_patient_dir / OUTCOME_DIRS[key] / sid
        patient_dir.mkdir(parents=True, exist_ok=True)

        images_np = dataset.images[idx][VIEW_IDX.get(view, slice(None))]
        make_attention_grid(images_np, attn_array[idx], sid,
                            int(labels[idx]), int(preds[idx]),
                            patient_dir / 'attention_grid.png', view=view)
        plot_patient_attention_distribution(
            attn_array[idx],
            patient_dir / 'attention_distribution.png',
            view=view,
            subject_id=sid,
            label=int(labels[idx]),
            pred=int(preds[idx]),
            all_attn_array=attn_array,
            all_labels=labels,
        )

        # Build top-10 slice list with view and local-index annotations
        top10_global = np.argsort(attn_array[idx])[::-1][:10].tolist()
        top10_info = [
            {'global_index': g, 'view': _slice_view_info(g, view)[0],
             'local_index': _slice_view_info(g, view)[1]}
            for g in top10_global
        ]
        with open(patient_dir / 'info.json', 'w') as f:
            json.dump({
                'subject_id': sid,
                'true_label': int(labels[idx]),
                'predicted':  int(preds[idx]),
                'outcome':    OUTCOME_DIRS[key],
                'top10_slices': top10_info,
            }, f, indent=2)
        counts[key] += 1

    print(f'\n  Per-patient heatmaps → {per_patient_dir}')
    for key, name in OUTCOME_DIRS.items():
        print(f'    {name}: {counts[key]}')


def plot_attention_distribution(attn_array, labels, save_path, view='all'):
    """
    Plot mean attention scores for Alzheimer patients, broken down by anatomical axis.

    For view='all' (80 slices: 16 axial + 32 coronal + 32 sagittal), renders three
    subplots side by side, one per axis. For single-view inputs, renders one subplot.
    Shaded band shows ±1 standard deviation across Alzheimer patients.

    Args:
        attn_array : (N, S) float array – per-patient per-slice attention weights.
        labels     : (N,) int array    – ground-truth labels (1 = Alzheimer).
        save_path  : Path-like         – output PNG path.
        view       : str               – one of 'all', 'axial', 'coronal', 'sagittal'.
    """
    labels_arr = np.array(labels)
    alz_mask = labels_arr == 1
    if not alz_mask.any():
        return  # No Alzheimer patients in this split — skip.

    alz_attn = attn_array[alz_mask]  # (N_alz, S)

    if view == 'all':
        axes_info = [
            ('Axial',    alz_attn[:, 0:80]),
            ('Coronal',  alz_attn[:, 80:160]),
            ('Sagittal', alz_attn[:, 160:240]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    else:
        axes_info = [(view.capitalize(), alz_attn)]
        fig, ax_single = plt.subplots(1, 1, figsize=(8, 4))
        axes = [ax_single]

    for ax, (axis_name, axis_attn) in zip(axes, axes_info):
        S = axis_attn.shape[1]
        x = np.arange(S)
        mean_attn = axis_attn.mean(axis=0)
        std_attn  = axis_attn.std(axis=0)

        ax.plot(x, mean_attn, color='tomato', linewidth=1.5, label='Mean (Alzheimer)')
        ax.fill_between(x, mean_attn - std_attn, mean_attn + std_attn,
                        color='tomato', alpha=0.2, label='±1 Std Dev')

        # Mark the peak attention slice
        peak = int(np.argmax(mean_attn))
        ax.axvline(peak, color='tomato', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.annotate(f'peak={peak}', xy=(peak, mean_attn[peak]),
                    xytext=(peak + 0.5, mean_attn[peak]),
                    fontsize=7, color='tomato')

        ax.set_xlabel('Slice Index', fontsize=9)
        ax.set_ylabel('Attention Score', fontsize=9)
        ax.set_title(f'{axis_name} View  ({S} slices)', fontsize=10)
        ax.set_xlim(0, S - 1)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Alzheimer — Attention Score by Anatomical Axis', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_patient_attention_distribution(attn_scores, save_path, view='all',
                                        subject_id=None, label=None, pred=None,
                                        all_attn_array=None, all_labels=None):
    """
    Plot per-patient attention distribution and save to PNG.
    For 'all' view, split into axial/coronal/sagittal subplots.
    
    If all_attn_array and all_labels are provided, overlay population mean ± std 
    (from Alzheimer patients) as a line chart for comparison.
    """
    attn_scores = np.asarray(attn_scores)
    title = 'Per-patient Attention Distribution'
    if subject_id is not None:
        title = f'{subject_id} - {title}'
    if label is not None and pred is not None:
        true_str = 'Alzheimer' if int(label) == 1 else 'Normal'
        pred_str = 'Alzheimer' if int(pred) == 1 else 'Normal'
        title += f' | True: {true_str}, Pred: {pred_str}'

    if view == 'all':
        axes_info = [
            ('Axial', attn_scores[0:80]),
            ('Coronal', attn_scores[80:160]),
            ('Sagittal', attn_scores[160:240]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    else:
        axes_info = [(view.capitalize(), attn_scores)]
        fig, ax_single = plt.subplots(1, 1, figsize=(8, 4))
        axes = [ax_single]

    # Compute population mean ± std if provided
    pop_mean_dict = {}
    pop_std_dict = {}
    if all_attn_array is not None and all_labels is not None:
        all_labels_arr = np.array(all_labels)
        alz_mask = all_labels_arr == 1
        if alz_mask.any():
            alz_attn = all_attn_array[alz_mask]
            if view == 'all':
                pop_mean_dict['Axial'] = alz_attn[:, 0:80].mean(axis=0)
                pop_std_dict['Axial'] = alz_attn[:, 0:80].std(axis=0)
                pop_mean_dict['Coronal'] = alz_attn[:, 80:160].mean(axis=0)
                pop_std_dict['Coronal'] = alz_attn[:, 80:160].std(axis=0)
                pop_mean_dict['Sagittal'] = alz_attn[:, 160:240].mean(axis=0)
                pop_std_dict['Sagittal'] = alz_attn[:, 160:240].std(axis=0)
            else:
                pop_mean_dict[view.capitalize()] = alz_attn.mean(axis=0)
                pop_std_dict[view.capitalize()] = alz_attn.std(axis=0)

    for ax, (axis_name, axis_scores) in zip(axes, axes_info):
        x = np.arange(len(axis_scores))
        
        # Overlay population mean ± std if available
        if axis_name in pop_mean_dict:
            pop_mean = pop_mean_dict[axis_name]
            pop_std = pop_std_dict[axis_name]
            ax.plot(x, pop_mean, color='steelblue', linewidth=2, marker='o', 
                   markersize=4, label='Alzheimer Mean', zorder=5)
            ax.fill_between(x, pop_mean - pop_std, pop_mean + pop_std,
                           color='steelblue', alpha=0.2, label='±1 Std Dev (Population)')
        
        peak = int(np.argmax(axis_scores))
        ax.axvline(peak, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_title(f'{axis_name} View', fontsize=10)
        ax.set_xlabel('Slice Index', fontsize=9)
        ax.set_ylabel('Attention Score', fontsize=9)
        ax.set_xlim(-0.5, len(axis_scores) - 0.5)
        ax.grid(axis='y', alpha=0.3)
        if axis_name in pop_mean_dict:
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def prepare_output_root(save_dir):
    """Clear old evaluation results and recreate output root for a fresh run."""
    if save_dir.exists():
        shutil.rmtree(save_dir)
        print(f"[INFO] Removed old evaluation directory: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)


def plot_prob_distribution(labels, probs, save_path):
    labels = np.array(labels)
    probs = np.array(probs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    normal_probs = probs[labels == 0]
    alzheimer_probs = probs[labels == 1]

    axes[0].hist(normal_probs, bins=20, alpha=0.7, label='Normal', color='steelblue')
    axes[0].hist(alzheimer_probs, bins=20, alpha=0.7, label='Alzheimer', color='tomato')
    axes[0].set_xlabel('Probability (Alzheimer class)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Prediction Probability Distribution')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].boxplot([normal_probs, alzheimer_probs], labels=['Normal', 'Alzheimer'])
    axes[1].set_ylabel('Probability (Alzheimer class)')
    axes[1].set_title('Prediction Probability by True Class')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, dataloader, device):
    """Run inference and return metrics + raw arrays."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    all_attn        = []  # per-slice attention scores, list of (S,) arrays
    all_subject_ids = []  # subject ID strings

    with torch.no_grad():
        for images, labels, subject_ids in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs, attn_weights = model(images)
            # attn_weights: (B, S, S) already averaged over heads — mean over query dim → (B, S)
            slice_attn = attn_weights.mean(dim=1)  # (B, S)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_attn.extend(slice_attn.cpu().numpy())
            all_subject_ids.extend(list(subject_ids))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy':    float(accuracy_score(all_labels, all_preds)),
        'precision':   float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall':      float(recall_score(all_labels, all_preds, zero_division=0)),
        'specificity': float(recall_score(all_labels, all_preds, pos_label=0, zero_division=0)),
        'f1':          float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc':         float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.0,
        'confusion_matrix': cm.tolist(),
        'true_negatives':   int(tn),
        'false_positives':  int(fp),
        'false_negatives':  int(fn),
        'true_positives':   int(tp),
        'classification_report': classification_report(
            all_labels, all_preds,
            target_names=['Normal', 'Alzheimer'],
            output_dict=True
        ),
    }

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"AUC-ROC:     {metrics['auc']:.4f}")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Alzheimer']))

    return metrics, all_labels, all_preds, all_probs, np.array(all_attn), all_subject_ids


def evaluate_fold(args, fold):
    """Evaluate a single fold."""
    # Auto-compute num_slices from view
    num_slices = args.num_slices if args.num_slices is not None else VIEW_SLICE_MAP[args.view]

    print(f"\n{'='*70}")
    print(f"Evaluating Fold {fold} | View: {args.view.upper()} ({num_slices} slices) | Split: {args.split}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & loader
    dataset = OASIS2DDataset(
        data_dir=args.data_dir,
        fold=fold,
        split=args.split,
        view=args.view,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"Batches: {len(loader)}")

    # Locate checkpoint
    ckpt_path = Path(args.checkpoint_dir) / f'fold_{fold}' / 'best_model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build model
    model = MyMobileNetMultiAttention(
        num_classes=2,
        num_slices=num_slices,
        embed_dim=args.embed_dim,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}  |  saved metric: {ckpt.get('metric', '?'):.4f}")

    # Run evaluation
    metrics, labels, preds, probs, attn_array, subject_ids = evaluate_model(model, loader, device)
    metrics['fold'] = fold
    metrics['view'] = args.view
    metrics['num_slices'] = num_slices
    metrics['split'] = args.split

    # Save results
    save_dir = Path(args.save_dir) / f'fold_{fold}'
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nGenerating plots...")
    plot_confusion_matrix(labels, preds, ['Normal', 'Alzheimer'], save_dir / 'confusion_matrix.png')
    plot_roc_curve(labels, probs, save_dir / 'roc_curve.png')
    plot_prob_distribution(labels, probs, save_dir / 'probability_distribution.png')
    plot_attention_distribution(attn_array, labels, save_dir / 'attention_distribution.png', view=args.view)
    np.save(save_dir / 'attn_weights.npy', attn_array)
    np.save(save_dir / 'subject_ids.npy', np.array(subject_ids))

    # Per-patient heatmap overlays
    print("\nGenerating attention overlay images...")
    if args.split == 'test':
        # Test split: generate full per-patient folders (TP / TN / FP / FN)
        save_test_heatmaps(dataset, labels, preds, attn_array, subject_ids,
                           save_dir, view=args.view)
    else:
        # Val / Train split: generate a small random sample to avoid flooding disk
        save_val_sample_heatmaps(dataset, labels, preds, attn_array, subject_ids,
                                 save_dir, view=args.view, n_samples=3)

    print(f"Results saved to: {save_dir}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model for Alzheimer classification')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_jpeg',
                        help='Root directory containing axial/, coronal/, sagittal/ JPEG folders')
    parser.add_argument('--fold', type=int, default=-1,
                        help='Fold to evaluate (0-4), or -1 for all folds')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--view', type=str, default='sagittal',
                        choices=['all', 'axial', 'coronal', 'sagittal'],
                        help='MRI view: all (80 slices), axial (16), coronal (32), sagittal (32)')
    parser.add_argument('--num_slices', type=int, default=None,
                        help='Override number of slices (auto-computed from --view if not set)')

    # Model
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension (must match training config)')

    # Checkpoint & output
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/2d_cv5',
                        help='Root directory of fold checkpoints (contains fold_X/best_model.pth)')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results/2d_cv5',
                        help='Directory to save evaluation outputs')

    # Runtime
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)
    prepare_output_root(Path(args.save_dir))

    print("=" * 70)
    print("2D CNN + Attention — Evaluation")
    print("=" * 70)
    print("\nConfiguration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if args.fold == -1:
        # Evaluate all folds and summarise
        all_metrics = []
        for fold in range(5):
            m = evaluate_fold(args, fold)
            all_metrics.append(m)

        print("\n" + "=" * 70)
        print("5-Fold Cross-Validation Results")
        print("=" * 70)
        for m in all_metrics:
            print(f"Fold {m['fold']}: Acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}  "
                  f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  Spec={m['specificity']:.4f}")

        avg_acc = np.mean([m['accuracy'] for m in all_metrics])
        std_acc = np.std([m['accuracy'] for m in all_metrics])
        avg_auc = np.mean([m['auc'] for m in all_metrics])
        std_auc = np.std([m['auc'] for m in all_metrics])
        avg_f1  = np.mean([m['f1']  for m in all_metrics])
        std_f1  = np.std([m['f1']  for m in all_metrics])

        print(f"\nAverage: Acc={avg_acc:.4f}±{std_acc:.4f}  "
              f"AUC={avg_auc:.4f}±{std_auc:.4f}  "
              f"F1={avg_f1:.4f}±{std_f1:.4f}")

        summary = {
            'view': args.view,
            'split': args.split,
            'fold_results': all_metrics,
            'average_acc': float(avg_acc), 'std_acc': float(std_acc),
            'average_auc': float(avg_auc), 'std_auc': float(std_auc),
            'average_f1':  float(avg_f1),  'std_f1':  float(std_f1),
        }
        summary_dir = Path(args.save_dir)
        with open(summary_dir / 'cv5_eval_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_dir / 'cv5_eval_summary.json'}")
    else:
        evaluate_fold(args, args.fold)
