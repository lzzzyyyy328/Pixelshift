from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .utils import to_float01, image_to_gray_float


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(to_float01(gt), to_float01(pred), data_range=1.0))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    g = to_float01(gt)
    p = to_float01(pred)
    if g.ndim == 2:
        return float(structural_similarity(g, p, data_range=1.0))
    return float(structural_similarity(g, p, data_range=1.0, channel_axis=-1))


def laplacian_sharpness(img: np.ndarray) -> float:
    gray = image_to_gray_float(img)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def evaluate(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "PSNR": psnr(gt, pred),
        "SSIM": ssim(gt, pred),
        "LaplacianSharpness": laplacian_sharpness(pred),
    }
