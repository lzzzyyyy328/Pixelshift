from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from .degradation import degrade_one
from .utils import Shift, to_float01


def _validate_calibration_data(
    hr: np.ndarray,
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    scale: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not frames:
        raise ValueError("At least one calibration frame is required.")
    if len(frames) != len(shifts_hr):
        raise ValueError("Number of calibration frames and shifts must match.")
    high_resolution = to_float01(hr)
    observations = [to_float01(frame) for frame in frames]
    expected = (high_resolution.shape[0] // scale, high_resolution.shape[1] // scale)
    for index, frame in enumerate(observations):
        if frame.shape[:2] != expected:
            raise ValueError(
                f"Calibration frame {index} has shape {frame.shape[:2]}; expected {expected}."
            )
        if frame.ndim != high_resolution.ndim:
            raise ValueError("HR target and LR frames must use the same channel layout.")
    return high_resolution, observations


def _fit_noise_model(
    predictions: Sequence[np.ndarray],
    residuals: Sequence[np.ndarray],
    bins: int = 32,
) -> dict:
    signal = np.concatenate([item.reshape(-1) for item in predictions]).astype(np.float64)
    squared = np.concatenate([(item * item).reshape(-1) for item in residuals]).astype(np.float64)
    valid = np.isfinite(signal) & np.isfinite(squared) & (signal > 0.01) & (signal < 0.99)
    signal = signal[valid]
    squared = squared[valid]
    if signal.size < 128:
        return {
            "poisson_a": 0.0,
            "gaussian_b": 0.0,
            "r_squared": 0.0,
            "sample_count": int(signal.size),
            "warning": "Too few unsaturated samples for a reliable noise fit.",
        }

    edges = np.linspace(0.0, 1.0, max(int(bins), 4) + 1)
    centers: list[float] = []
    variances: list[float] = []
    counts: list[int] = []
    assignments = np.digitize(signal, edges) - 1
    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(selected.sum())
        if count < 32:
            continue
        centers.append(float(signal[selected].mean()))
        variances.append(float(squared[selected].mean()))
        counts.append(count)

    if len(centers) < 2:
        return {
            "poisson_a": 0.0,
            "gaussian_b": float(np.mean(squared)),
            "r_squared": 0.0,
            "sample_count": int(signal.size),
            "warning": "Signal range is too narrow to separate Poisson and read noise.",
        }

    x = np.asarray(centers, dtype=np.float64)
    y = np.asarray(variances, dtype=np.float64)
    weight = np.sqrt(np.asarray(counts, dtype=np.float64))
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, _, _, _ = np.linalg.lstsq(
        design * weight[:, None],
        y * weight,
        rcond=None,
    )
    poisson_a = max(float(coefficients[0]), 0.0)
    gaussian_b = max(float(coefficients[1]), 0.0)
    fitted = poisson_a * x + gaussian_b
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 1e-15 else 0.0
    return {
        "poisson_a": poisson_a,
        "gaussian_b": gaussian_b,
        "r_squared": float(r_squared),
        "sample_count": int(signal.size),
        "bin_count": len(centers),
    }


def calibrate_forward_model(
    hr: np.ndarray,
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    *,
    scale: int,
    blur_types: Sequence[str] = ("defocus", "gaussian"),
    blur_kernels: Sequence[int] = (1, 3, 5, 7),
    blur_sigmas: Sequence[float] = (0.6, 1.0, 1.4),
    blur_stage: str = "before_downsample",
    border_mode: str = "replicate",
    interpolation: str = "area",
) -> tuple[dict, list[np.ndarray]]:
    """Grid-search a calibrated optical model and estimate sensor noise."""
    high_resolution, observations = _validate_calibration_data(
        hr,
        frames,
        shifts_hr,
        scale,
    )
    kinds = [item.strip().lower() for item in blur_types if item.strip()]
    kernels = sorted({int(item) for item in blur_kernels})
    sigmas = sorted({float(item) for item in blur_sigmas})
    if not kinds or not kernels:
        raise ValueError("At least one blur type and kernel are required.")
    if any(item not in {"none", "defocus", "gaussian"} for item in kinds):
        raise ValueError("Unsupported blur type in calibration grid.")
    if any(item < 1 for item in kernels):
        raise ValueError("Calibration kernels must be positive.")

    candidates: list[dict] = []
    best_predictions: list[np.ndarray] = []
    best_residuals: list[np.ndarray] = []
    best_mse = float("inf")

    for blur_type in kinds:
        if blur_type == "gaussian":
            combinations = product(kernels, sigmas or [0.0])
        elif blur_type == "defocus":
            combinations = ((kernel, 0.0) for kernel in kernels)
        else:
            combinations = ((1, 0.0),)

        for kernel, sigma in combinations:
            predictions = [
                degrade_one(
                    high_resolution,
                    shift,
                    scale=scale,
                    blur_type=blur_type,
                    blur_kernel=kernel,
                    blur_sigma=sigma,
                    blur_stage=blur_stage,
                    border_mode=border_mode,
                    interpolation=interpolation,
                )
                for shift in shifts_hr
            ]
            residuals = [
                observed - predicted
                for observed, predicted in zip(observations, predictions)
            ]
            per_frame_mse = [float(np.mean(item * item)) for item in residuals]
            mse = float(np.mean(per_frame_mse))
            candidate = {
                "blur_type": blur_type,
                "blur_kernel": int(kernel),
                "blur_sigma": float(sigma),
                "mean_mse": mse,
                "per_frame_mse": per_frame_mse,
            }
            candidates.append(candidate)
            if mse < best_mse:
                best_mse = mse
                best_predictions = predictions
                best_residuals = residuals

    candidates.sort(key=lambda item: item["mean_mse"])
    best = dict(candidates[0])
    best["noise_model"] = _fit_noise_model(best_predictions, best_residuals)
    report = {
        "best_model": best,
        "candidate_count": len(candidates),
        "top_candidates": candidates[: min(10, len(candidates))],
        "shift_convention": "positive dx/dy moves image content right/down",
        "blur_stage": blur_stage,
        "downsample": interpolation,
        "border": border_mode,
    }
    return report, best_predictions
