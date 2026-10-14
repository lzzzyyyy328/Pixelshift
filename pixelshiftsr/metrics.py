from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .degradation import degrade_one
from .utils import Shift, image_to_gray_float, to_float01


def _matching_images(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = to_float01(first)
    b = to_float01(second)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes do not match: {a.shape} vs {b.shape}.")
    return a, b


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    g, p = _matching_images(gt, pred)
    return float(peak_signal_noise_ratio(g, p, data_range=1.0))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    g, p = _matching_images(gt, pred)
    min_side = min(g.shape[:2])
    if min_side < 3:
        raise ValueError("SSIM requires image dimensions of at least 3 pixels.")
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if g.ndim == 2:
        return float(structural_similarity(g, p, data_range=1.0, win_size=win_size))
    return float(
        structural_similarity(
            g,
            p,
            data_range=1.0,
            channel_axis=-1,
            win_size=win_size,
        )
    )


def laplacian_sharpness(img: np.ndarray) -> float:
    gray = image_to_gray_float(img)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(laplacian.var())


def tenengrad(img: np.ndarray) -> float:
    gray = image_to_gray_float(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def entropy(img: np.ndarray, bins: int = 256) -> float:
    gray = image_to_gray_float(img)
    histogram, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def no_reference_metrics(img: np.ndarray) -> dict[str, float]:
    arr = to_float01(img)
    p01, p99 = np.percentile(arr, [1.0, 99.0])
    return {
        "LaplacianSharpness": laplacian_sharpness(arr),
        "Tenengrad": tenengrad(arr),
        "Entropy": entropy(arr),
        "DynamicRangeP01P99": float(p99 - p01),
        "SaturationFraction": float(np.mean((arr <= 0.002) | (arr >= 0.998))),
        "MeanIntensity": float(arr.mean()),
        "StdIntensity": float(arr.std()),
    }


def evaluate(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    g, p = _matching_images(gt, pred)
    difference = g - p
    result = {
        "PSNR": psnr(g, p),
        "SSIM": ssim(g, p),
        "RMSE": float(np.sqrt(np.mean(difference * difference))),
        "MAE": float(np.mean(np.abs(difference))),
    }
    result.update(no_reference_metrics(p))
    return result


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = image_to_gray_float(first).astype(np.float64).ravel()
    b = image_to_gray_float(second).astype(np.float64).ravel()
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def reprojection_metrics(
    sr: np.ndarray,
    frames: Sequence[np.ndarray],
    shifts_hr: Sequence[Shift],
    *,
    scale: int,
    blur_type: str,
    blur_kernel: int,
    blur_sigma: float,
    blur_stage: str,
    border_mode: str,
    interpolation: str,
    frame_weights: Sequence[float] | None = None,
) -> dict:
    if len(frames) != len(shifts_hr):
        raise ValueError("Number of frames and shifts must match.")
    if not frames:
        raise ValueError("At least one frame is required.")
    weights = (
        np.ones(len(frames), dtype=np.float64)
        if frame_weights is None
        else np.asarray(frame_weights, dtype=np.float64)
    )
    if weights.shape != (len(frames),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid frame weights.")

    per_frame: list[dict] = []
    for index, (frame, shift) in enumerate(zip(frames, shifts_hr)):
        observed = to_float01(frame)
        predicted = degrade_one(
            sr,
            shift,
            scale=scale,
            blur_type=blur_type,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            blur_stage=blur_stage,
            border_mode=border_mode,
            interpolation=interpolation,
        )
        difference = observed - predicted
        mse = float(np.mean(difference * difference))
        per_frame.append(
            {
                "frame_index": index,
                "weight": float(weights[index]),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "MAE": float(np.mean(np.abs(difference))),
                "PSNR": float("inf") if mse == 0 else float(-10.0 * np.log10(mse)),
                "NormalizedCorrelation": _normalized_correlation(observed, predicted),
            }
        )

    mse_values = np.asarray([item["MSE"] for item in per_frame], dtype=np.float64)
    mae_values = np.asarray([item["MAE"] for item in per_frame], dtype=np.float64)
    aggregate_mse = float(np.average(mse_values, weights=weights))
    return {
        "aggregate": {
            "WeightedMSE": aggregate_mse,
            "WeightedRMSE": float(np.sqrt(aggregate_mse)),
            "WeightedMAE": float(np.average(mae_values, weights=weights)),
            "WorstFrameMSE": float(mse_values.max()),
        },
        "per_frame": per_frame,
    }
