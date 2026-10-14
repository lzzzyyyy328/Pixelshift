from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from tqdm import tqdm

from .degradation import backproject_one, degrade_one
from .utils import Shift, convert_shifts_to_hr, to_float01


@dataclass
class ReconstructionConfig:
    scale: int = 2
    shifts: Sequence[Shift] | None = None
    shift_unit: str = "lr"
    frame_weights: Sequence[float] | None = None
    mode: str = "ibp"
    iterations: int = 80
    step_size: float = 0.20
    robust_loss: str = "huber"
    huber_delta: float = 0.03
    tolerance: float = 1e-5
    patience: int = 8
    min_iterations: int = 10
    tv_weight: float = 0.0
    laplacian_weight: float = 0.0
    blur_type: str = "defocus"
    blur_kernel: int = 3
    blur_sigma: float = 0.0
    blur_stage: str = "before_downsample"
    border_mode: str = "replicate"
    interpolation: str = "area"
    clip: bool = True
    show_progress: bool = True


def _validate_frames(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        raise ValueError("At least one LR frame is required.")
    converted = [to_float01(frame) for frame in frames]
    shape = converted[0].shape
    if len(shape) not in {2, 3}:
        raise ValueError(f"Unsupported LR frame shape: {shape}")
    for index, frame in enumerate(converted[1:], start=1):
        if frame.shape != shape:
            raise ValueError(
                f"Frame {index} has shape {frame.shape}; expected {shape}."
            )
    return converted


def _frame_weights(count: int, weights: Sequence[float] | None) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float32)
    out = np.asarray(weights, dtype=np.float32)
    if out.shape != (count,):
        raise ValueError(f"Expected {count} frame weights, received {out.size}.")
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError("Frame weights must be finite and non-negative.")
    if float(out.sum()) <= 0:
        raise ValueError("At least one frame weight must be positive.")
    return out


def _upsample(
    frame: np.ndarray,
    scale: int,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(
        frame.astype(np.float32),
        (w * scale, h * scale),
        interpolation=interpolation,
    )


def bicubic_baseline(
    frames: Sequence[np.ndarray],
    scale: int = 2,
    frame_weights: Sequence[float] | None = None,
) -> np.ndarray:
    frames_float = _validate_frames(frames)
    weights = _frame_weights(len(frames_float), frame_weights)
    acc = np.zeros_like(_upsample(frames_float[0], scale), dtype=np.float32)
    for frame, weight in zip(frames_float, weights):
        acc += float(weight) * _upsample(frame, scale, cv2.INTER_CUBIC)
    return np.clip(acc / float(weights.sum()), 0.0, 1.0).astype(np.float32)


def interlace_reconstruct(
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    scale: int = 2,
    frame_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Directly place ideal grid-aligned observations on the HR lattice."""
    frames_float = _validate_frames(frames)
    if len(frames_float) != len(shifts_hr):
        raise ValueError("Number of frames and shifts must match.")
    weights = _frame_weights(len(frames_float), frame_weights)
    lr_h, lr_w = frames_float[0].shape[:2]
    suffix = () if frames_float[0].ndim == 2 else (frames_float[0].shape[-1],)
    out_shape = (lr_h * scale, lr_w * scale) + suffix
    estimate = np.zeros(out_shape, dtype=np.float32)
    coverage = np.zeros(out_shape, dtype=np.float32)

    for frame, shift, frame_weight in zip(frames_float, shifts_hr, weights):
        offset_x = int(round(shift.dx)) % scale
        offset_y = int(round(shift.dy)) % scale
        estimate[offset_y::scale, offset_x::scale, ...] += float(frame_weight) * frame
        coverage[offset_y::scale, offset_x::scale, ...] += float(frame_weight)

    missing = coverage <= 1e-8
    if np.any(missing):
        fallback = bicubic_baseline(frames_float, scale, weights)
        estimate[missing] = fallback[missing]
        coverage[missing] = 1.0
    return np.clip(estimate / np.maximum(coverage, 1e-8), 0.0, 1.0).astype(np.float32)


def fuse_reconstruct(
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    scale: int = 2,
    frame_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Align upsampled observations on the HR grid and combine them by weight."""
    frames_float = _validate_frames(frames)
    if len(frames_float) != len(shifts_hr):
        raise ValueError("Number of frames and shifts must match.")
    weights = _frame_weights(len(frames_float), frame_weights)
    accumulator: np.ndarray | None = None
    coverage: np.ndarray | None = None

    for frame, shift, frame_weight in zip(frames_float, shifts_hr, weights):
        upsampled = _upsample(frame, scale, cv2.INTER_CUBIC)
        h, w = upsampled.shape[:2]
        matrix = np.array(
            [[1.0, 0.0, -shift.dx], [0.0, 1.0, -shift.dy]],
            dtype=np.float32,
        )
        aligned = cv2.warpAffine(
            upsampled,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        valid = cv2.warpAffine(
            np.ones((h, w), dtype=np.float32),
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if aligned.ndim == 3:
            valid = valid[..., None]
        if accumulator is None:
            accumulator = np.zeros_like(aligned, dtype=np.float32)
            coverage = np.zeros_like(aligned, dtype=np.float32)
        accumulator += float(frame_weight) * aligned * valid
        coverage += float(frame_weight) * valid

    assert accumulator is not None and coverage is not None
    return np.clip(
        accumulator / np.maximum(coverage, 1e-8),
        0.0,
        1.0,
    ).astype(np.float32)


def _robust_residual(
    residual: np.ndarray,
    loss_name: str,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    name = loss_name.lower()
    if name == "l2":
        return residual, 0.5 * residual * residual
    if name != "huber":
        raise ValueError("robust_loss must be 'l2' or 'huber'.")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive.")
    absolute = np.abs(residual)
    quadratic = absolute <= huber_delta
    influence = np.where(quadratic, residual, huber_delta * np.sign(residual))
    loss = np.where(
        quadratic,
        0.5 * residual * residual,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return influence.astype(np.float32), loss.astype(np.float32)


def _laplacian_smooth_step(image: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return image
    laplacian = cv2.Laplacian(
        image.astype(np.float32),
        cv2.CV_32F,
        ksize=3,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return image + weight * laplacian


def _tv_denoise_step(image: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return image
    channel_axis = -1 if image.ndim == 3 else None
    return denoise_tv_chambolle(
        image,
        weight=weight,
        channel_axis=channel_axis,
    ).astype(np.float32)


def _project(
    image: np.ndarray,
    shift_hr: Shift,
    cfg: ReconstructionConfig,
) -> np.ndarray:
    return degrade_one(
        image,
        shift_hr,
        scale=cfg.scale,
        blur_type=cfg.blur_type,
        blur_kernel=cfg.blur_kernel,
        blur_sigma=cfg.blur_sigma,
        blur_stage=cfg.blur_stage,
        border_mode=cfg.border_mode,
        interpolation=cfg.interpolation,
    )


def _data_step(
    image: np.ndarray,
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    weights: np.ndarray,
    cfg: ReconstructionConfig,
    with_correction: bool,
) -> tuple[float, list[float], np.ndarray | None, np.ndarray | None]:
    correction = np.zeros_like(image, dtype=np.float32) if with_correction else None
    sensitivity = np.zeros_like(image, dtype=np.float32) if with_correction else None
    weighted_loss = 0.0
    per_frame_mse: list[float] = []

    for frame, shift, frame_weight in zip(frames, shifts_hr, weights):
        prediction = _project(image, shift, cfg)
        residual = frame.astype(np.float32) - prediction.astype(np.float32)
        influence, robust_loss = _robust_residual(
            residual,
            cfg.robust_loss,
            cfg.huber_delta,
        )
        per_frame_mse.append(float(np.mean(residual * residual)))
        weighted_loss += float(frame_weight) * float(np.mean(robust_loss))

        if with_correction:
            assert correction is not None and sensitivity is not None
            correction += float(frame_weight) * backproject_one(
                influence,
                shift,
                image.shape,
                scale=cfg.scale,
                blur_type=cfg.blur_type,
                blur_kernel=cfg.blur_kernel,
                blur_sigma=cfg.blur_sigma,
                blur_stage=cfg.blur_stage,
                interpolation=cfg.interpolation,
            )
            sensitivity += float(frame_weight) * backproject_one(
                np.ones_like(frame, dtype=np.float32),
                shift,
                image.shape,
                scale=cfg.scale,
                blur_type=cfg.blur_type,
                blur_kernel=cfg.blur_kernel,
                blur_sigma=cfg.blur_sigma,
                blur_stage=cfg.blur_stage,
                interpolation=cfg.interpolation,
            )

    return weighted_loss / float(weights.sum()), per_frame_mse, correction, sensitivity


def ibp_reconstruct(
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    cfg: ReconstructionConfig,
) -> tuple[np.ndarray, dict]:
    """Run confidence-weighted robust iterative back-projection."""
    frames_float = _validate_frames(frames)
    if len(frames_float) != len(shifts_hr):
        raise ValueError("Number of frames and shifts must match.")
    if cfg.iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if cfg.step_size <= 0:
        raise ValueError("step_size must be positive.")
    if cfg.tolerance < 0 or cfg.patience < 1 or cfg.min_iterations < 0:
        raise ValueError("Invalid convergence settings.")

    weights = _frame_weights(len(frames_float), cfg.frame_weights)
    estimate = fuse_reconstruct(frames_float, shifts_hr, cfg.scale, weights)
    best_estimate = estimate.copy()
    best_loss = float("inf")
    best_iteration = 0
    losses: list[float] = []
    stale_iterations = 0
    stopped_early = False
    final_per_frame_mse: list[float] = []
    coverage_summary = {"minimum": 0.0, "maximum": 0.0, "mean": 0.0}

    iterator = tqdm(
        range(cfg.iterations),
        desc="Robust IBP",
        leave=False,
        disable=not cfg.show_progress,
    )
    updates_run = 0
    for iteration in iterator:
        loss, per_frame_mse, correction, sensitivity = _data_step(
            estimate,
            frames_float,
            shifts_hr,
            weights,
            cfg,
            with_correction=True,
        )
        assert correction is not None and sensitivity is not None
        losses.append(loss)
        final_per_frame_mse = per_frame_mse

        if loss < best_loss:
            best_loss = loss
            best_estimate = estimate.copy()
            best_iteration = iteration

        if len(losses) > 1:
            improvement = (losses[-2] - losses[-1]) / max(abs(losses[-2]), 1e-12)
            stale_iterations = stale_iterations + 1 if improvement < cfg.tolerance else 0
            if (
                iteration >= cfg.min_iterations
                and stale_iterations >= cfg.patience
            ):
                stopped_early = True
                break

        positive = sensitivity > 1e-8
        normalized = np.zeros_like(correction, dtype=np.float32)
        normalized[positive] = correction[positive] / sensitivity[positive]
        estimate = estimate + cfg.step_size * normalized
        estimate = _laplacian_smooth_step(estimate, cfg.laplacian_weight)
        estimate = _tv_denoise_step(estimate, cfg.tv_weight)
        if cfg.clip:
            estimate = np.clip(estimate, 0.0, 1.0)
        updates_run += 1

        sensitivity_gray = (
            sensitivity.mean(axis=-1) if sensitivity.ndim == 3 else sensitivity
        )
        coverage_summary = {
            "minimum": float(sensitivity_gray.min()),
            "maximum": float(sensitivity_gray.max()),
            "mean": float(sensitivity_gray.mean()),
        }
        iterator.set_postfix(loss=f"{loss:.7f}")

    final_loss, final_per_frame_mse, _, _ = _data_step(
        estimate,
        frames_float,
        shifts_hr,
        weights,
        cfg,
        with_correction=False,
    )
    if final_loss < best_loss:
        best_loss = final_loss
        best_estimate = estimate.copy()
        best_iteration = updates_run
    if not losses or final_loss != losses[-1]:
        losses.append(final_loss)

    best_loss, final_per_frame_mse, _, _ = _data_step(
        best_estimate,
        frames_float,
        shifts_hr,
        weights,
        cfg,
        with_correction=False,
    )

    diagnostics = {
        "losses": losses,
        "best_loss": best_loss,
        "best_iteration": best_iteration,
        "iterations_requested": cfg.iterations,
        "updates_run": updates_run,
        "stopped_early": stopped_early,
        "per_frame_reprojection_mse": final_per_frame_mse,
        "coverage": coverage_summary,
        "robust_loss": cfg.robust_loss,
        "huber_delta": cfg.huber_delta,
        "tolerance": cfg.tolerance,
        "patience": cfg.patience,
    }
    return np.clip(best_estimate, 0.0, 1.0).astype(np.float32), diagnostics


def reconstruct(
    frames: Sequence[np.ndarray],
    cfg: ReconstructionConfig,
) -> tuple[np.ndarray, dict]:
    frames_float = _validate_frames(frames)
    shifts = list(cfg.shifts) if cfg.shifts is not None else [
        Shift(0.0, 0.0),
        Shift(0.5, 0.0),
        Shift(0.0, 0.5),
        Shift(0.5, 0.5),
    ]
    if len(shifts) != len(frames_float):
        raise ValueError(
            f"Received {len(frames_float)} frames but {len(shifts)} shifts."
        )
    shifts_hr = convert_shifts_to_hr(shifts, cfg.scale, cfg.shift_unit)
    weights = _frame_weights(len(frames_float), cfg.frame_weights)
    mode = cfg.mode.lower()
    info: dict = {
        "mode": mode,
        "scale": cfg.scale,
        "shift_convention": "positive dx/dy moves image content right/down",
        "shifts_hr": [shift.as_tuple() for shift in shifts_hr],
        "frame_weights": [float(value) for value in weights],
        "forward_model": {
            "blur_type": cfg.blur_type,
            "blur_kernel": cfg.blur_kernel,
            "blur_sigma": cfg.blur_sigma,
            "blur_stage": cfg.blur_stage,
            "downsample": cfg.interpolation,
            "border": cfg.border_mode,
        },
    }

    if mode == "bicubic":
        return bicubic_baseline(frames_float, cfg.scale, weights), info
    if mode == "interlace":
        return interlace_reconstruct(frames_float, shifts_hr, cfg.scale, weights), info
    if mode == "fuse":
        return fuse_reconstruct(frames_float, shifts_hr, cfg.scale, weights), info
    if mode in {"ibp", "inverse", "backprojection"}:
        super_resolved, diagnostics = ibp_reconstruct(frames_float, shifts_hr, cfg)
        info["optimization"] = diagnostics
        return super_resolved, info
    raise ValueError(f"Unknown reconstruction mode: {cfg.mode}")
