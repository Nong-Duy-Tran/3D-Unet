from __future__ import annotations

from typing import Final

VALID_LABEL_MODES: Final[tuple[str, ...]] = (
    "cdr4",
    "normal_vs_nonnormal",
    "ad_vs_nonad",
    "cn_vs_ad_drop_05",
)


def _normalize_cdr(cdr: float) -> float:
    return round(float(cdr), 3)


def get_class_names(label_mode: str) -> list[str]:
    mode = str(label_mode)
    if mode == "cdr4":
        return ["cdr_0", "cdr_0p5", "cdr_1", "cdr_2"]
    if mode == "normal_vs_nonnormal":
        return ["normal", "nonnormal"]
    if mode == "ad_vs_nonad":
        return ["nonad", "ad"]
    if mode == "cn_vs_ad_drop_05":
        return ["cn", "ad"]
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def build_label(cdr: float, label_mode: str) -> tuple[int, str] | None:
    """
    Build training label and folder-safe class name from raw OASIS CDR.

    Returns:
      (label_idx, class_name), or None if this sample should be dropped
      under the requested label mode.
    """
    cdr_value = _normalize_cdr(cdr)
    mode = str(label_mode)

    if mode == "cdr4":
        mapping = {
            0.0: (0, "cdr_0"),
            0.5: (1, "cdr_0p5"),
            1.0: (2, "cdr_1"),
            2.0: (3, "cdr_2"),
        }
        return mapping.get(cdr_value)

    if mode == "normal_vs_nonnormal":
        if cdr_value == 0.0:
            return 0, "normal"
        if cdr_value in (0.5, 1.0, 2.0):
            return 1, "nonnormal"
        return None

    if mode == "ad_vs_nonad":
        if cdr_value in (0.0, 0.5):
            return 0, "nonad"
        if cdr_value in (1.0, 2.0):
            return 1, "ad"
        return None

    if mode == "cn_vs_ad_drop_05":
        if cdr_value == 0.5:
            return None
        if cdr_value == 0.0:
            return 0, "cn"
        if cdr_value in (1.0, 2.0):
            return 1, "ad"
        return None

    raise ValueError(f"Unsupported label_mode: {label_mode}")