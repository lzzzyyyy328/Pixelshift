from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import cv2
import numpy as np

from .utils import ensure_dir, natural_sort_key, to_float01, from_float01

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def read_image(path: str | Path, as_float: bool = True) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        img = np.load(path)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        # OpenCV uses BGR/BGRA; convert to RGB/RGBA for consistent processing.
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return to_float01(img) if as_float else img


def save_image(path: str | Path, image: np.ndarray, dtype: str = "uint8") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".npy":
        np.save(path, image)
        return
    out = from_float01(image, dtype=dtype) if np.issubdtype(image.dtype, np.floating) else image
    if out.ndim == 3 and out.shape[-1] == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    elif out.ndim == 3 and out.shape[-1] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files, key=natural_sort_key)


def read_image_sequence(folder: str | Path, as_float: bool = True) -> list[np.ndarray]:
    files = list_images(folder)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return [read_image(p, as_float=as_float) for p in files]


def save_image_sequence(folder: str | Path, frames: Sequence[np.ndarray], prefix: str = "lr", dtype: str = "uint8") -> list[Path]:
    folder = ensure_dir(folder)
    paths: list[Path] = []
    for i, frame in enumerate(frames):
        p = folder / f"{prefix}_{i:02d}.png"
        save_image(p, frame, dtype=dtype)
        paths.append(p)
    return paths
