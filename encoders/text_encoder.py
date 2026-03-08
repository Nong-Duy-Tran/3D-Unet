from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from encoders.template import (
    SLICE_PROMPT,
    VOLUME_PROMPT,
    build_slice_messages,
    build_text_messages,
    build_volume_messages,
    to_rgb_like_image,
)

DEFAULT_MODEL_NAME = "google/medgemma-1.5-4b-it"


def _default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _resolve_hf_token(token: str | None) -> str | None:
    if token:
        return token

    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        token_path = Path(hf_home) / "token"
        if token_path.exists():
            return token_path.read_text(encoding="utf-8").strip()

    return None


def _resolve_cache_dir(cache_dir: str | os.PathLike[str] | None) -> str | None:
    if cache_dir is not None:
        path = Path(cache_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        path = Path(hf_hub_cache).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        path = (Path(hf_home).expanduser().resolve() / "hub")
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    return None


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return masked.sum(dim=1) / denom


class TextEncoder:
    """
    MedGemma-backed multimodal encoder for prompt + image generation and embeddings.

    The same model is used for:
    1. deterministic caption generation from brain sMRI slices or selected slice sets
    2. pooled multimodal embeddings from the prepared chat inputs
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        torch_dtype: torch.dtype | None = None,
        device_map: str | dict[str, int | str] = "auto",
        token: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.torch_dtype = torch_dtype or _default_dtype()
        self.token = _resolve_hf_token(token)
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            token=self.token,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            token=self.token,
            cache_dir=self.cache_dir,
        )
        self.model.eval()
        self.device = getattr(self.model, "device", torch.device("cpu"))

    def _prepare_inputs(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool = True,
    ):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _move_batch_to_device(inputs, self.device, self.torch_dtype)
        return inputs

    def generate_from_messages(
        self,
        messages: list[dict[str, object]],
        max_new_tokens: int = 256,
    ) -> str:
        inputs = self._prepare_inputs(messages, add_generation_prompt=True)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True).strip()

    def encode_from_messages(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool = True,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(messages, add_generation_prompt=add_generation_prompt)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_state = outputs.hidden_states[-1]
            pooled = _mean_pool(hidden_state, inputs["attention_mask"])
            return F.normalize(pooled, dim=-1)

    def generate_slice_caption(
        self,
        slice_image: Image.Image | torch.Tensor,
        prompt: str = SLICE_PROMPT,
        max_new_tokens: int = 256,
    ) -> str:
        return self.generate_from_messages(
            build_slice_messages(slice_image=slice_image, prompt=prompt),
            max_new_tokens=max_new_tokens,
        )

    def generate_volume_summary(
        self,
        slice_images: Sequence[Image.Image | torch.Tensor],
        prompt: str = VOLUME_PROMPT,
        max_new_tokens: int = 256,
    ) -> str:
        return self.generate_from_messages(
            build_volume_messages(slice_images=slice_images, prompt=prompt),
            max_new_tokens=max_new_tokens,
        )

    def embed_texts(
        self,
        texts: str | Sequence[str],
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [
            self.encode_from_messages(build_text_messages(prompt=text), add_generation_prompt=False)[0]
            for text in texts
        ]
        return torch.stack(embeddings, dim=0)

    def encode_slice(
        self,
        slice_image: Image.Image | torch.Tensor,
        prompt: str = SLICE_PROMPT,
    ) -> torch.Tensor:
        return self.encode_from_messages(
            build_slice_messages(slice_image=slice_image, prompt=prompt),
            add_generation_prompt=True,
        )

    def encode_volume(
        self,
        slice_images: Sequence[Image.Image | torch.Tensor],
        prompt: str = VOLUME_PROMPT,
    ) -> torch.Tensor:
        return self.encode_from_messages(
            build_volume_messages(slice_images=slice_images, prompt=prompt),
            add_generation_prompt=True,
        )

    def encode_selected_slices(
        self,
        slices: Sequence[Image.Image | torch.Tensor],
        max_new_tokens: int = 256,
    ) -> dict[str, object]:
        rgb_like_slices = [to_rgb_like_image(slice_image) for slice_image in slices]
        slice_captions = [
            self.generate_slice_caption(slice_image, max_new_tokens=max_new_tokens)
            for slice_image in rgb_like_slices
        ]
        volume_summary = self.generate_volume_summary(
            rgb_like_slices,
            max_new_tokens=max_new_tokens,
        )

        return {
            "slice_captions": slice_captions,
            "volume_summary": volume_summary,
            "slice_input_embeddings": torch.cat(
                [self.encode_slice(slice_image) for slice_image in rgb_like_slices],
                dim=0,
            ),
            "volume_input_embedding": self.encode_volume(rgb_like_slices)[0],
            "slice_caption_embeddings": self.embed_texts(slice_captions),
            "volume_caption_embedding": self.embed_texts(volume_summary)[0],
        }
