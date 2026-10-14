from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .utils import Shift, convert_shifts_to_hr, to_float01


@dataclass
class DegradationConfig:
    """Configuration for the calibrated image-formation model."""

    scale: int = 2
    shifts: Sequence[Shift] | None = None
    shift_unit: str = "lr"
    blur_type: str = "defocus"
    blur_kernel: int = 5
    blur_sigma: float = 0.0
    blur_stage: str = "before_downsample"
    noise_poisson_a: float = 0.001
    noise_gaussian_b: float = 0.0001
    seed: int | None = 0
    border_mode: str = "replicate"
    interpolation: str = "area"


def _border_mode(name: str) -> int:
    modes = {
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "constant": cv2.BORDER_CONSTANT,
    }
    try:
        return modes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported border mode: {name!r}") from exc


def _interp(name: str) -> int:
    modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    try:
        return modes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported interpolation mode: {name!r}") from exc


def _validate_model(
    scale: int,
    blur_type: str,
    blur_kernel: int,
    blur_sigma: float,
    blur_stage: str,
) -> None:
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer.")
    if blur_type.lower() not in {"none", "defocus", "gaussian"}:
        raise ValueError("blur_type must be one of: none, defocus, gaussian.")
    if blur_stage.lower() not in {"none", "before_downsample", "after_downsample"}:
        raise ValueError("blur_stage must be one of: none, before_downsample, after_downsample.")
    if blur_kernel < 1:
        raise ValueError("blur_kernel must be at least 1.")
    if blur_sigma < 0:
        raise ValueError("blur_sigma must be non-negative.")


def gaussian_blur(img: np.ndarray, kernel: int = 3, sigma: float = 0.0) -> np.ndarray:
    if kernel <= 1:
        return img.astype(np.float32, copy=False)
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(
        img.astype(np.float32),
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def defocus_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a normalized circular point-spread function."""
    if kernel_size <= 1:
        return img.astype(np.float32, copy=False)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    radius = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= float(kernel.sum())
    return cv2.filter2D(
        img.astype(np.float32),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )


def apply_blur(
    img: np.ndarray,
    blur_type: str,
    blur_kernel: int,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    kind = blur_type.lower()
    if kind == "none" or blur_kernel <= 1:
        return img.astype(np.float32, copy=False)
    if kind == "defocus":
        return defocus_blur(img, blur_kernel)
    if kind == "gaussian":
        return gaussian_blur(img, blur_kernel, blur_sigma)
    raise ValueError(f"Unsupported blur type: {blur_type!r}")


def add_poisson_gaussian_noise(
    img: np.ndarray,
    a: float,
    b: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a heteroscedastic sensor model with variance a * signal + b."""
    if a < 0 or b < 0:
        raise ValueError("Noise coefficients must be non-negative.")
    if a == 0 and b == 0:
        return img.astype(np.float32, copy=False)
    variance = np.maximum(img, 0.0) * a + b
    noise = rng.normal(0.0, np.sqrt(variance), size=img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0.0, 1.0)


def warp_hr(
    img: np.ndarray,
    shift_hr: Shift,
    border_mode: str = "replicate",
    interpolation: str = "linear",
) -> np.ndarray:
    """Move image content by a forward shift in HR-grid pixels."""
    h, w = img.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, shift_hr.dx], [0.0, 1.0, shift_hr.dy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        img.astype(np.float32),
        matrix,
        (w, h),
        flags=_interp(interpolation),
        borderMode=_border_mode(border_mode),
    )


def downsample(img: np.ndarray, scale: int, interpolation: str = "area") -> np.ndarray:
    if not isinstance(scale, int) or scale < 1:
        raise ValueError("scale must be a positive integer.")
    h, w = img.shape[:2]
    if h % scale != 0 or w % scale != 0:
        raise ValueError(f"HR size {(h, w)} must be divisible by scale={scale}.")
    return cv2.resize(
        img.astype(np.float32),
        (w // scale, h // scale),
        interpolation=_interp(interpolation),
    )


def degrade_one(
    hr: np.ndarray,
    shift_hr: Shift,
    scale: int = 2,
    blur_type: str = "defocus",
    blur_kernel: int = 5,
    blur_sigma: float = 0.0,
    blur_stage: str = "before_downsample",
    border_mode: str = "replicate",
    interpolation: str = "area",
) -> np.ndarray:
    """Project one HR estimate into an LR observation.

    Shift convention: positive dx/dy moves image content right/down. The same
    convention is used by registration and reconstruction throughout the package.
    """
    _validate_model(scale, blur_type, blur_kernel, blur_sigma, blur_stage)
    stage = blur_stage.lower()
    x = to_float01(hr)

    if stage == "before_downsample":
        x = apply_blur(x, blur_type, blur_kernel, blur_sigma)

    x = warp_hr(x, shift_hr, border_mode=border_mode, interpolation="linear")
    y = downsample(x, scale=scale, interpolation=interpolation)

    if stage == "after_downsample":
        y = apply_blur(y, blur_type, blur_kernel, blur_sigma)

    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _upsample_adjoint(
    residual_lr: np.ndarray,
    scale: int,
    hr_shape: tuple[int, ...],
    interpolation: str,
) -> np.ndarray:
    target_h, target_w = hr_shape[:2]
    if residual_lr.shape[0] * scale != target_h or residual_lr.shape[1] * scale != target_w:
        raise ValueError("LR residual shape and requested HR shape are inconsistent with scale.")

    if interpolation.lower() == "area":
        up = np.repeat(np.repeat(residual_lr, scale, axis=0), scale, axis=1)
    else:
        up = cv2.resize(
            residual_lr.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return up.astype(np.float32) / float(scale * scale)


def backproject_one(
    residual_lr: np.ndarray,
    shift_hr: Shift,
    hr_shape: tuple[int, ...],
    scale: int = 2,
    blur_type: str = "defocus",
    blur_kernel: int = 5,
    blur_sigma: float = 0.0,
    blur_stage: str = "before_downsample",
    interpolation: str = "area",
) -> np.ndarray:
    """Approximate the adjoint of :func:`degrade_one` for one residual."""
    _validate_model(scale, blur_type, blur_kernel, blur_sigma, blur_stage)
    stage = blur_stage.lower()
    residual = np.asarray(residual_lr, dtype=np.float32)

    if stage == "after_downsample":
        residual = apply_blur(residual, blur_type, blur_kernel, blur_sigma)

    correction = _upsample_adjoint(residual, scale, hr_shape, interpolation)
    correction = warp_hr(
        correction,
        Shift(-shift_hr.dx, -shift_hr.dy),
        border_mode="constant",
        interpolation="linear",
    )

    if stage == "before_downsample":
        correction = apply_blur(correction, blur_type, blur_kernel, blur_sigma)
    return correction.astype(np.float32)


def degrade_sequence(
    hr: np.ndarray,
    cfg: DegradationConfig,
) -> tuple[list[np.ndarray], list[Shift]]:
    shifts = list(cfg.shifts) if cfg.shifts is not None else [
        Shift(0.0, 0.0),
        Shift(0.5, 0.0),
        Shift(0.0, 0.5),
        Shift(0.5, 0.5),
    ]
    if not shifts:
        raise ValueError("At least one shift is required.")
    if cfg.noise_poisson_a < 0 or cfg.noise_gaussian_b < 0:
        raise ValueError("Noise coefficients must be non-negative.")

    shifts_hr = convert_shifts_to_hr(shifts, cfg.scale, cfg.shift_unit)
    rng = np.random.default_rng(cfg.seed)
    frames: list[np.ndarray] = []

    for shift_hr in shifts_hr:
        lr = degrade_one(
            hr,
            shift_hr,
            scale=cfg.scale,
            blur_type=cfg.blur_type,
            blur_kernel=cfg.blur_kernel,
            blur_sigma=cfg.blur_sigma,
            blur_stage=cfg.blur_stage,
            border_mode=cfg.border_mode,
            interpolation=cfg.interpolation,
        )
        lr = add_poisson_gaussian_noise(
            lr,
            cfg.noise_poisson_a,
            cfg.noise_gaussian_b,
            rng,
        )
        frames.append(lr.astype(np.float32))

    return frames, shifts_hr
