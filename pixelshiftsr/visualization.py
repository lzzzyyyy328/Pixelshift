from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import save_image
from .utils import ensure_dir, to_float01


def make_comparison(gt: np.ndarray | None, lr0: np.ndarray, sr: np.ndarray, out_path: str | Path, scale: int = 2) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lr_up = cv2.resize(to_float01(lr0), (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)

    if gt is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(np.clip(lr_up, 0, 1), cmap="gray" if lr_up.ndim == 2 else None)
        axes[0].set_title("LR frame 0 upsampled")
        axes[1].imshow(np.clip(sr, 0, 1), cmap="gray" if sr.ndim == 2 else None)
        axes[1].set_title("Pixel-shift SR")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.clip(to_float01(gt), 0, 1), cmap="gray" if gt.ndim == 2 else None)
        axes[0].set_title("HR ground truth")
        axes[1].imshow(np.clip(lr_up, 0, 1), cmap="gray" if lr_up.ndim == 2 else None)
        axes[1].set_title("LR frame 0 upsampled")
        axes[2].imshow(np.clip(sr, 0, 1), cmap="gray" if sr.ndim == 2 else None)
        axes[2].set_title("Pixel-shift SR")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_error_map(gt: np.ndarray, sr: np.ndarray, out_path: str | Path) -> None:
    gt = to_float01(gt)
    sr = to_float01(sr)
    err = np.abs(gt - sr)
    if err.ndim == 3:
        err = err.mean(axis=-1)
    err = err / (err.max() + 1e-8)
    save_image(out_path, err, dtype="uint8")
