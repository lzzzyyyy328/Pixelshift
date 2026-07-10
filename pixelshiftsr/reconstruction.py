from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np
from tqdm import tqdm

from .degradation import degrade_one, gaussian_blur
from .utils import Shift, convert_shifts_to_hr, to_float01


@dataclass
class ReconstructionConfig:
    scale: int = 2
    shifts: Sequence[Shift] | None = None
    shift_unit: str = "lr"
    mode: str = "ibp"  # bicubic, interlace, fuse, ibp
    iterations: int = 80
    step_size: float = 0.20
    tv_weight: float = 0.0
    laplacian_weight: float = 0.0
    blur_type: str = "defocus"
    blur_kernel: int = 3
    blur_sigma: float = 0.0
    blur_stage: str = "after_downsample"
    border_mode: str = "replicate"
    interpolation: str = "area"
    clip: bool = True


def _upsample(frame: np.ndarray, scale: int, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame.astype(np.float32), (w * scale, h * scale), interpolation=interpolation)


def bicubic_baseline(frames: Sequence[np.ndarray], scale: int = 2) -> np.ndarray:
    ups = [_upsample(to_float01(f), scale, cv2.INTER_CUBIC) for f in frames]
    return np.clip(np.mean(ups, axis=0), 0.0, 1.0).astype(np.float32)


def interlace_reconstruct(frames: Sequence[np.ndarray], shifts_hr: Sequence[Shift], scale: int = 2) -> np.ndarray:
    """Known-shift direct grid interlacing.

    Best for ideal noiseless pixel-shift data where shifts fall exactly on the
    HR grid: e.g. HR shifts (0,0), (1,0), (0,1), (1,1) for scale=2.
    """
    frames = [to_float01(f) for f in frames]
    lr_h, lr_w = frames[0].shape[:2]
    out_shape = (lr_h * scale, lr_w * scale) + (() if frames[0].ndim == 2 else (frames[0].shape[-1],))
    hr = np.zeros(out_shape, dtype=np.float32)
    weight = np.zeros(out_shape, dtype=np.float32)

    for frame, sh in zip(frames, shifts_hr):
        # Convert shift to integer HR-grid offsets. We use modulo to keep offsets
        # within [0, scale-1].
        ox = int(round(sh.dx)) % scale
        oy = int(round(sh.dy)) % scale
        hr[oy::scale, ox::scale, ...] += frame
        weight[oy::scale, ox::scale, ...] += 1.0

    missing = weight <= 0
    if np.any(missing):
        fallback = bicubic_baseline(frames, scale=scale)
        hr[missing] = fallback[missing]
        weight[missing] = 1.0
    return np.clip(hr / np.maximum(weight, 1e-6), 0.0, 1.0).astype(np.float32)


def fuse_reconstruct(frames: Sequence[np.ndarray], shifts_hr: Sequence[Shift], scale: int = 2) -> np.ndarray:
    """Sub-pixel shift-aware fusion on the HR grid.

    Each LR frame is upsampled, then shifted back on the HR grid and averaged.
    This is more robust than direct interlacing when shifts are not exactly grid-aligned.
    """
    acc = None
    weight = None
    for frame, sh in zip(frames, shifts_hr):
        up = _upsample(to_float01(frame), scale, cv2.INTER_CUBIC)
        h, w = up.shape[:2]
        # Inverse warp because degradation shifted HR by +sh before sampling.
        M = np.array([[1.0, 0.0, -sh.dx], [0.0, 1.0, -sh.dy]], dtype=np.float32)
        aligned = cv2.warpAffine(up.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.ones(up.shape[:2], dtype=np.float32)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if aligned.ndim == 3:
            mask = mask[..., None]
        if acc is None:
            acc = np.zeros_like(aligned, dtype=np.float32)
            weight = np.zeros_like(aligned, dtype=np.float32)
        acc += aligned * mask
        weight += mask
    return np.clip(acc / np.maximum(weight, 1e-6), 0.0, 1.0).astype(np.float32)


def _backproject_error(error_lr: np.ndarray, shift_hr: Shift, scale: int) -> np.ndarray:
    """Approximate adjoint of the forward model: LR error -> HR correction."""
    err_hr = _upsample(error_lr, scale, cv2.INTER_LINEAR)
    h, w = err_hr.shape[:2]
    M = np.array([[1.0, 0.0, -shift_hr.dx], [0.0, 1.0, -shift_hr.dy]], dtype=np.float32)
    return cv2.warpAffine(err_hr.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _laplacian_smooth_step(x: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return x
    lap = cv2.Laplacian(x.astype(np.float32), cv2.CV_32F, ksize=3)
    return x - weight * lap


def _tv_denoise_step(x: np.ndarray, weight: float) -> np.ndarray:
    """Lightweight anisotropic TV-like smoothing step.

    This is deliberately simple and dependency-light. For high-quality TV, use
    skimage.restoration.denoise_tv_chambolle externally.
    """
    if weight <= 0:
        return x
    # Edge-preserving-ish: use bilateral on 8-bit proxy and blend back.
    x8 = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(x8, d=5, sigmaColor=20, sigmaSpace=3).astype(np.float32) / 255.0
    return (1.0 - weight) * x + weight * filtered


def ibp_reconstruct(frames: Sequence[np.ndarray], shifts_hr: Sequence[Shift], cfg: ReconstructionConfig) -> tuple[np.ndarray, list[float]]:
    """Iterative back-projection reconstruction.

    It minimizes a practical data-consistency objective: when the recovered HR
    image is degraded with the same forward model, it should reproduce all LR
    frames.
    """
    frames = [to_float01(f) for f in frames]
    if len(frames) != len(shifts_hr):
        raise ValueError("Number of frames and shifts must match.")

    # Use direct interlace if possible; otherwise robust fusion.
    x = fuse_reconstruct(frames, shifts_hr, scale=cfg.scale)
    losses: list[float] = []

    iterator = tqdm(range(cfg.iterations), desc="IBP", leave=False)
    for _ in iterator:
        correction = np.zeros_like(x, dtype=np.float32)
        loss = 0.0
        for y, sh in zip(frames, shifts_hr):
            pred = degrade_one(
                x,
                sh,
                scale=cfg.scale,
                blur_type=cfg.blur_type,
                blur_kernel=cfg.blur_kernel,
                blur_sigma=cfg.blur_sigma,
                blur_stage=cfg.blur_stage,
                border_mode=cfg.border_mode,
                interpolation=cfg.interpolation,
            )
            err = y.astype(np.float32) - pred.astype(np.float32)
            loss += float(np.mean(err * err))
            correction += _backproject_error(err, sh, cfg.scale)
        correction /= max(len(frames), 1)
        x = x + cfg.step_size * correction
        x = _laplacian_smooth_step(x, cfg.laplacian_weight)
        x = _tv_denoise_step(x, cfg.tv_weight)
        if cfg.clip:
            x = np.clip(x, 0.0, 1.0)
        losses.append(loss / max(len(frames), 1))
        iterator.set_postfix(loss=f"{losses[-1]:.6f}")
    return np.clip(x, 0.0, 1.0).astype(np.float32), losses


def reconstruct(frames: Sequence[np.ndarray], cfg: ReconstructionConfig) -> tuple[np.ndarray, dict]:
    shifts = list(cfg.shifts) if cfg.shifts is not None else [Shift(0.0, 0.0), Shift(0.5, 0.0), Shift(0.0, 0.5), Shift(0.5, 0.5)]
    shifts_hr = convert_shifts_to_hr(shifts, cfg.scale, cfg.shift_unit)
    mode = cfg.mode.lower()
    info: dict = {"mode": mode, "scale": cfg.scale, "shifts_hr": [s.as_tuple() for s in shifts_hr]}

    if mode == "bicubic":
        return bicubic_baseline(frames, cfg.scale), info
    if mode == "interlace":
        return interlace_reconstruct(frames, shifts_hr, cfg.scale), info
    if mode == "fuse":
        return fuse_reconstruct(frames, shifts_hr, cfg.scale), info
    if mode in {"ibp", "inverse", "backprojection"}:
        sr, losses = ibp_reconstruct(frames, shifts_hr, cfg)
        info["losses"] = losses
        return sr, info
    raise ValueError(f"Unknown reconstruction mode: {cfg.mode}")
