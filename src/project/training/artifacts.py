from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig


def save_model_artifacts(
    lightning_module: Any,
    sample_batch: Any,
    cfg: DictConfig,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model = getattr(lightning_module, "model", lightning_module)
    sample_input = _extract_input_tensor(sample_batch)

    summary_path = artifact_dir / "model_architecture.txt"
    png_path = artifact_dir / "model_architecture.png"

    summary_text, flow = _build_summary(model, sample_input, cfg)
    summary_path.write_text(summary_text, encoding="utf-8")

    try:
        _save_flow_diagram_png(flow, png_path)
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to save model diagram PNG to {png_path}: {exc}")

    print(f"Saved model architecture summary: {summary_path}")
    if png_path.exists():
        print(f"Saved model architecture diagram: {png_path}")


def _extract_input_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    elif isinstance(batch, dict):
        if "image" in batch:
            x = batch["image"]
        elif "x" in batch:
            x = batch["x"]
        else:
            raise ValueError("Unable to infer input tensor from batch dict.")
    else:
        x = batch

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected tensor input batch, got {type(x).__name__}")
    if x.shape[0] > 1:
        x = x[:1]
    return x.detach().cpu()


def _shape_str(x: torch.Tensor) -> str:
    return "(" + ", ".join(str(int(dim)) for dim in x.shape) + ")"


def _module_desc(module: Any) -> str:
    if module is None:
        return "n/a"
    return type(module).__name__


def _is_identity(module: Any) -> bool:
    return isinstance(module, torch.nn.Identity)


def _trace_slice_sequence_model(model: Any, sample_input: torch.Tensor) -> list[dict[str, str]]:
    x = sample_input
    flow: list[dict[str, str]] = [
        {
            "name": "Input MRI 2.5D",
            "detail": "x: (B, N, C, H, W)",
            "shape": _shape_str(x),
        }
    ]

    if x.dim() == 4:
        x = x.unsqueeze(1)
        flow.append(
            {
                "name": "Unsqueeze Sequence Dim",
                "detail": "Convert 4D input to 5D expected by 2.5D model",
                "shape": _shape_str(x),
            }
        )

    b, n, c, h, w = x.shape
    x = x.view(b * n, c, h, w)
    flow.append(
        {
            "name": "Reshape",
            "detail": "Merge batch and slice dimensions before 2D encoder",
            "shape": _shape_str(x),
        }
    )

    feats = model.encoder(x)
    flow.append(
        {
            "name": "Backbone Encoder",
            "detail": _module_desc(model.encoder),
            "shape": _shape_str(feats),
        }
    )

    feats = feats.view(b, n, -1)
    flow.append(
        {
            "name": "Reshape Back",
            "detail": "Restore sequence dimension per subject",
            "shape": _shape_str(feats),
        }
    )

    if hasattr(model, "slice_proj"):
        feats = model.slice_proj(feats)
        flow.append(
            {
                "name": "Slice Projection",
                "detail": _module_desc(model.slice_proj),
                "shape": _shape_str(feats),
            }
        )

    if hasattr(model, "slice_pos"):
        feats = model.slice_pos(feats)
        mode = getattr(model.slice_pos, "mode", "unknown")
        flow.append(
            {
                "name": "Slice Positional Encoding",
                "detail": f"{_module_desc(model.slice_pos)} mode={mode}",
                "shape": _shape_str(feats),
            }
        )

    if hasattr(model, "slice_sequence"):
        feats = model.slice_sequence(feats)
        mode = getattr(model.slice_sequence, "mode", "unknown")
        encoder_name = _module_desc(getattr(model.slice_sequence, "encoder", None))
        flow.append(
            {
                "name": "Slice Sequence Encoder",
                "detail": f"{_module_desc(model.slice_sequence)} mode={mode}, encoder={encoder_name}",
                "shape": _shape_str(feats),
            }
        )

    if hasattr(model, "token_mlp"):
        feats = model.token_mlp(feats)
        token_block = getattr(model.token_mlp, "block", None)
        token_desc = _module_desc(token_block)
        if _is_identity(token_block):
            token_desc = f"{token_desc} (disabled)"
        flow.append(
            {
                "name": "Token MLP",
                "detail": token_desc,
                "shape": _shape_str(feats),
            }
        )

    pooled = model.pool(feats)
    flow.append(
        {
            "name": "Slice Attention Pool",
            "detail": _module_desc(model.pool),
            "shape": _shape_str(pooled),
        }
    )

    logits = model.classifier(pooled)
    flow.append(
        {
            "name": "Classifier Head",
            "detail": _module_desc(model.classifier),
            "shape": _shape_str(logits),
        }
    )
    return flow


def _trace_generic_model(model: Any, sample_input: torch.Tensor) -> list[dict[str, str]]:
    x = sample_input
    with torch.no_grad():
        logits = model(x)
    return [
        {
            "name": "Input",
            "detail": "Sample input batch",
            "shape": _shape_str(x),
        },
        {
            "name": "Model Forward",
            "detail": _module_desc(model),
            "shape": _shape_str(logits),
        },
    ]


def _build_summary(model: Any, sample_input: torch.Tensor, cfg: DictConfig) -> tuple[str, list[dict[str, str]]]:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "encoder") and hasattr(model, "pool") and hasattr(model, "classifier"):
                flow = _trace_slice_sequence_model(model, sample_input)
            else:
                flow = _trace_generic_model(model, sample_input)
    finally:
        model.train(was_training)

    lines: list[str] = [
        "Model Architecture Summary",
        "=" * 80,
        f"model_class: {_module_desc(model)}",
        f"config_model_name: {getattr(cfg.model, 'name', 'unknown')}",
        f"sample_input_shape: {_shape_str(sample_input)}",
        "",
        "Shape Flow",
        "-" * 80,
    ]

    for idx, item in enumerate(flow, start=1):
        lines.append(f"{idx}. {item['name']}")
        lines.append(f"   detail: {item['detail']}")
        lines.append(f"   shape : {item['shape']}")

    lines.extend(
        [
            "",
            "Model Repr",
            "-" * 80,
            repr(model),
            "",
        ]
    )
    return "\n".join(lines), flow


def _save_flow_diagram_png(flow: list[dict[str, str]], output_path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is unavailable") from exc

    box_height = 1.2
    gap = 0.45
    width = 10.0
    total_height = len(flow) * (box_height + gap) + 0.8

    fig, ax = plt.subplots(figsize=(12, max(6, total_height * 0.7)))
    ax.set_xlim(0, width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    y = total_height - box_height - 0.4
    x = 0.8
    box_width = width - 1.6

    for idx, item in enumerate(flow):
        patch = FancyBboxPatch(
            (x, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#1f2937",
            facecolor="#e5eef9",
        )
        ax.add_patch(patch)
        title = f"{idx + 1}. {item['name']}"
        detail = f"{item['detail']}\nshape: {item['shape']}"
        ax.text(x + 0.25, y + 0.78, title, fontsize=11, fontweight="bold", va="center", ha="left")
        ax.text(x + 0.25, y + 0.38, detail, fontsize=9.5, va="center", ha="left")

        if idx < len(flow) - 1:
            ax.annotate(
                "",
                xy=(width / 2.0, y - gap + 0.08),
                xytext=(width / 2.0, y),
                arrowprops=dict(arrowstyle="->", linewidth=1.6, color="#1f2937"),
            )
        y -= box_height + gap

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
