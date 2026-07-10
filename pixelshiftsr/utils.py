from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class Shift:
    """A 2D sub-pixel shift.

    dx and dy are stored as floating point values. Their unit is controlled by
    the caller: either LR-pixel units or HR-pixel units.
    """
    dx: float
    dy: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.dx, self.dy)


def parse_shifts(text: str | None, scale: int = 2) -> list[Shift]:
    """Parse shifts from ``"0,0;0.5,0;0,0.5;0.5,0.5"``.

    If text is None or empty, return the canonical 2x2 half-pixel shifts in LR
    pixel units.
    """
    if text is None or str(text).strip() == "":
        return [Shift(0.0, 0.0), Shift(0.5, 0.0), Shift(0.0, 0.5), Shift(0.5, 0.5)]
    shifts: list[Shift] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid shift item: {item!r}. Expected dx,dy")
        shifts.append(Shift(float(parts[0]), float(parts[1])))
    if not shifts:
        raise ValueError("No valid shifts parsed.")
    return shifts


def convert_shifts_to_hr(shifts: Sequence[Shift], scale: int, shift_unit: str) -> list[Shift]:
    """Convert shifts to HR-pixel units.

    shift_unit='lr': dx=0.5 means half a LR pixel, i.e. dx*scale HR pixels.
    shift_unit='hr': dx=0.5 means directly 0.5 HR pixels. This matches the
    uploaded prototype script.
    """
    shift_unit = shift_unit.lower()
    if shift_unit not in {"lr", "hr"}:
        raise ValueError("shift_unit must be 'lr' or 'hr'.")
    factor = scale if shift_unit == "lr" else 1.0
    return [Shift(s.dx * factor, s.dy * factor) for s in shifts]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint/float image to float32 in [0, 1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    out = img.astype(np.float32)
    if out.size > 0 and out.max() > 1.5:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def from_float01(img: np.ndarray, dtype: str = "uint8") -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    if dtype == "uint16":
        return np.round(img * 65535.0).astype(np.uint16)
    if dtype == "float32":
        return img.astype(np.float32)
    return np.round(img * 255.0).astype(np.uint8)


def image_to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = to_float01(img)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] == 1:
        return arr[..., 0]
    # RGB luminance
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def natural_sort_key(path: str | Path):
    import re
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
