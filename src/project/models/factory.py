from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from src.baseline.models import get_model, get_model_config


def build_model_kwargs(cfg: DictConfig) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "in_channels": cfg.model.in_channels,
        "num_classes": cfg.model.num_classes,
    }
    vgg_model_names = {"vgg2d", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"}

    if cfg.model.name == "simple":
        model_kwargs["base_features"] = cfg.model.base_features
        return model_kwargs
    if cfg.model.name == "deit2d":
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(cfg.data.target_shape[1])
        in_channels = cfg.model.in_channels
        if getattr(cfg.data, "rgb_mode", False):
            in_channels = 3
        model_kwargs.update(
            {
                "timm_name": cfg.model.timm_name,
                "pretrained": cfg.model.pretrained,
                "image_size": image_size,
                "in_channels": in_channels,
                "drop_path_rate": cfg.model.drop_path_rate,
                "attn_drop_rate": cfg.model.attn_drop_rate,
                "dropout": cfg.model.dropout,
                "slice_embed_dim": getattr(cfg.model, "slice_embed_dim", None),
                "slice_pos_encoding": getattr(cfg.model, "slice_pos_encoding", "none"),
                "slice_pos_max_len": int(
                    getattr(cfg.model, "slice_pos_max_len", getattr(cfg.data, "num_slices", 512))
                ),
                "slice_pos_dropout": float(getattr(cfg.model, "slice_pos_dropout", 0.0)),
                "slice_sequence_encoder": getattr(cfg.model, "slice_sequence_encoder", "none"),
                "slice_num_heads": int(getattr(cfg.model, "slice_num_heads", 8)),
                "slice_transformer_depth": int(getattr(cfg.model, "slice_transformer_depth", 1)),
                "slice_transformer_mlp_ratio": float(
                    getattr(cfg.model, "slice_transformer_mlp_ratio", 4.0)
                ),
                "slice_transformer_dropout": float(
                    getattr(cfg.model, "slice_transformer_dropout", 0.0)
                ),
                "slice_transformer_attn_dropout": float(
                    getattr(cfg.model, "slice_transformer_attn_dropout", 0.0)
                ),
                "attn_init": bool(getattr(cfg.model, "attn_init", False)),
                "slice_attn_hidden_dim": getattr(cfg.model, "slice_attn_hidden_dim", None),
                "slice_attn_dropout": float(getattr(cfg.model, "slice_attn_dropout", 0.0)),
                "slice_attn_activation": getattr(cfg.model, "slice_attn_activation", "tanh"),
                "slice_attn_use_layernorm": bool(getattr(cfg.model, "slice_attn_use_layernorm", False)),
                "slice_attn_mode": getattr(cfg.model, "slice_attn_mode", "basic"),
            }
        )
        return model_kwargs
    if cfg.model.name == "timm25d":
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(getattr(cfg.data, "image_size", 224))
        in_channels = getattr(cfg.model, "in_channels", None)
        if in_channels is None:
            in_channels = int(getattr(cfg.data, "window_size", 5))
        model_kwargs.update(
            {
                "timm_name": cfg.model.timm_name,
                "pretrained": cfg.model.pretrained,
                "image_size": image_size,
                "in_channels": in_channels,
                "drop_path_rate": cfg.model.drop_path_rate,
                "attn_drop_rate": cfg.model.attn_drop_rate,
                "dropout": cfg.model.dropout,
                "slice_embed_dim": getattr(cfg.model, "slice_embed_dim", None),
                "slice_pos_encoding": getattr(cfg.model, "slice_pos_encoding", "none"),
                "slice_pos_max_len": int(
                    getattr(cfg.model, "slice_pos_max_len", getattr(cfg.data, "num_slices", 512))
                ),
                "slice_pos_dropout": float(getattr(cfg.model, "slice_pos_dropout", 0.0)),
                "slice_sequence_encoder": getattr(cfg.model, "slice_sequence_encoder", "none"),
                "slice_num_heads": int(getattr(cfg.model, "slice_num_heads", 8)),
                "slice_transformer_depth": int(getattr(cfg.model, "slice_transformer_depth", 1)),
                "slice_transformer_mlp_ratio": float(
                    getattr(cfg.model, "slice_transformer_mlp_ratio", 4.0)
                ),
                "slice_transformer_dropout": float(
                    getattr(cfg.model, "slice_transformer_dropout", 0.0)
                ),
                "slice_transformer_attn_dropout": float(
                    getattr(cfg.model, "slice_transformer_attn_dropout", 0.0)
                ),
                "attn_init": bool(getattr(cfg.model, "attn_init", False)),
                "slice_attn_hidden_dim": getattr(cfg.model, "slice_attn_hidden_dim", None),
                "slice_attn_dropout": float(getattr(cfg.model, "slice_attn_dropout", 0.0)),
                "slice_attn_activation": getattr(cfg.model, "slice_attn_activation", "tanh"),
                "slice_attn_use_layernorm": bool(getattr(cfg.model, "slice_attn_use_layernorm", False)),
                "slice_attn_mode": getattr(cfg.model, "slice_attn_mode", "basic"),
                "token_mlp_hidden_dim": getattr(cfg.model, "token_mlp_hidden_dim", None),
                "token_mlp_dropout": float(getattr(cfg.model, "token_mlp_dropout", 0.0)),
                "token_mlp_activation": getattr(cfg.model, "token_mlp_activation", "gelu"),
                "token_mlp_use_layernorm": bool(getattr(cfg.model, "token_mlp_use_layernorm", False)),
                "classifier_hidden_dim": getattr(cfg.model, "classifier_hidden_dim", None),
                "classifier_use_layernorm": bool(getattr(cfg.model, "classifier_use_layernorm", False)),
            }
        )
        return model_kwargs
    if cfg.model.name == "vgg25d":
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(getattr(cfg.data, "image_size", 224))
        in_channels = getattr(cfg.model, "in_channels", None)
        if in_channels is None:
            in_channels = int(getattr(cfg.data, "window_size", 5))
        model_kwargs.update(
            {
                "vgg_name": getattr(cfg.model, "vgg_name", "vgg16_bn"),
                "pretrained": bool(getattr(cfg.model, "pretrained", False)),
                "image_size": image_size,
                "in_channels": in_channels,
                "dropout": cfg.model.dropout,
                "slice_embed_dim": getattr(cfg.model, "slice_embed_dim", None),
                "slice_pos_encoding": getattr(cfg.model, "slice_pos_encoding", "none"),
                "slice_pos_max_len": int(
                    getattr(cfg.model, "slice_pos_max_len", getattr(cfg.data, "num_slices", 512))
                ),
                "slice_pos_dropout": float(getattr(cfg.model, "slice_pos_dropout", 0.0)),
                "slice_sequence_encoder": getattr(cfg.model, "slice_sequence_encoder", "none"),
                "slice_num_heads": int(getattr(cfg.model, "slice_num_heads", 8)),
                "slice_transformer_depth": int(getattr(cfg.model, "slice_transformer_depth", 1)),
                "slice_transformer_mlp_ratio": float(
                    getattr(cfg.model, "slice_transformer_mlp_ratio", 4.0)
                ),
                "slice_transformer_dropout": float(
                    getattr(cfg.model, "slice_transformer_dropout", 0.0)
                ),
                "slice_transformer_attn_dropout": float(
                    getattr(cfg.model, "slice_transformer_attn_dropout", 0.0)
                ),
                "attn_init": bool(getattr(cfg.model, "attn_init", False)),
                "slice_attn_hidden_dim": getattr(cfg.model, "slice_attn_hidden_dim", None),
                "slice_attn_dropout": float(getattr(cfg.model, "slice_attn_dropout", 0.0)),
                "slice_attn_activation": getattr(cfg.model, "slice_attn_activation", "tanh"),
                "slice_attn_use_layernorm": bool(getattr(cfg.model, "slice_attn_use_layernorm", False)),
                "slice_attn_mode": getattr(cfg.model, "slice_attn_mode", "basic"),
            }
        )
        return model_kwargs
    if cfg.model.name in vgg_model_names:
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(cfg.data.target_shape[1])
        in_channels = cfg.model.in_channels
        if getattr(cfg.data, "rgb_mode", False):
            in_channels = 3
        default_vgg_name = cfg.model.name if cfg.model.name != "vgg2d" else "vgg16_bn"
        model_kwargs.update(
            {
                "vgg_name": getattr(cfg.model, "vgg_name", default_vgg_name),
                "pretrained": cfg.model.pretrained,
                "image_size": image_size,
                "in_channels": in_channels,
                "dropout": cfg.model.dropout,
                "slice_attn_hidden_dim": getattr(cfg.model, "slice_attn_hidden_dim", None),
                "slice_attn_dropout": float(getattr(cfg.model, "slice_attn_dropout", 0.0)),
                "slice_attn_activation": getattr(cfg.model, "slice_attn_activation", "tanh"),
                "slice_attn_use_layernorm": bool(getattr(cfg.model, "slice_attn_use_layernorm", False)),
                "slice_attn_mode": getattr(cfg.model, "slice_attn_mode", "basic"),
            }
        )
        return model_kwargs
    if cfg.model.name == "simpleunet2d":
        in_channels = cfg.model.in_channels
        if getattr(cfg.data, "rgb_mode", False):
            in_channels = 3
        model_kwargs.update(
            {
                "in_channels": in_channels,
                "base_features": cfg.model.base_features,
                "head_hidden": cfg.model.head_hidden,
                "dropout": cfg.model.dropout,
                "head_dropout": cfg.model.head_dropout,
            }
        )
        return model_kwargs
    if cfg.model.name == "vit2d":
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(cfg.data.target_shape[1])
        model_kwargs.update(
            {
                "image_size": image_size,
                "patch_size": cfg.model.patch_size,
                "embed_dim": cfg.model.embed_dim,
                "depth": cfg.model.depth,
                "num_heads": cfg.model.num_heads,
                "mlp_dim": cfg.model.mlp_dim,
                "dropout": cfg.model.dropout,
            }
        )
        return model_kwargs
    if cfg.model.name == "swin2d":
        image_size = getattr(cfg.model, "image_size", None)
        if image_size is None:
            image_size = int(cfg.data.target_shape[1])
        model_kwargs.update(
            {
                "image_size": image_size,
                "patch_size": cfg.model.patch_size,
                "embed_dim": cfg.model.embed_dim,
                "depth": cfg.model.depth,
                "num_heads": cfg.model.num_heads,
                "window_size": cfg.model.window_size,
                "mlp_dim": cfg.model.mlp_dim,
                "dropout": cfg.model.dropout,
            }
        )
        return model_kwargs
    if cfg.model.name == "swin3d":
        model_kwargs.update(
            {
                "patch_size": cfg.model.patch_size,
                "embed_dim": cfg.model.embed_dim,
                "depth": cfg.model.depth,
                "num_heads": cfg.model.num_heads,
                "window_size": cfg.model.window_size,
                "mlp_dim": cfg.model.mlp_dim,
                "dropout": cfg.model.dropout,
            }
        )
        return model_kwargs

    f_maps = getattr(cfg.model, "f_maps", None)
    if f_maps is None:
        f_maps = getattr(cfg.model, "base_features", 64)
    model_kwargs["f_maps"] = f_maps

    num_levels = getattr(cfg.model, "num_levels", None)
    if num_levels is not None:
        model_kwargs["num_levels"] = num_levels

    return model_kwargs


def create_model_from_config(cfg: DictConfig):
    return get_model(model_name=cfg.model.name, **build_model_kwargs(cfg))
