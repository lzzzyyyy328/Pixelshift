from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import save_image
from .utils import ensure_dir, to_float01


def make_comparison(
    gt: np.ndarray | None,
    lr0: np.ndarray,
    sr: np.ndarray,
    out_path: str | Path,
    scale: int = 2,
    baseline: np.ndarray | None = None,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lr_up = cv2.resize(to_float01(lr0), (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)

    panels: list[tuple[str, np.ndarray]] = [("LR frame 0 (nearest)", lr_up)]
    if baseline is not None:
        panels.append(("Bicubic baseline", to_float01(baseline)))
    panels.append(("Pixel-shift SR", to_float01(sr)))
    if gt is not None:
        panels.append(("HR ground truth", to_float01(gt)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(np.clip(image, 0, 1), cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
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


def make_convergence_plot(losses: list[float], out_path: str | Path) -> None:
    if not losses:
        return
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    values = np.asarray(losses, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(np.arange(values.size), values, color="#176B87", linewidth=1.8)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Robust reprojection loss")
    axis.set_title("IBP convergence")
    axis.grid(True, alpha=0.25)
    if np.all(values > 0):
        axis.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_benchmark_comparison(
    images: dict[str, np.ndarray],
    out_path: str | Path,
    gt: np.ndarray | None = None,
) -> None:
    panels = list(images.items())
    if gt is not None:
        panels.append(("ground_truth", to_float01(gt)))
    if not panels:
        return
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    axes = np.atleast_1d(axes)
    for axis, (name, image) in zip(axes, panels):
        image = to_float01(image)
        axis.imshow(np.clip(image, 0, 1), cmap="gray" if image.ndim == 2 else None)
        axis.set_title(name.replace("_", " ").title())
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
