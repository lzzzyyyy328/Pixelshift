from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .utils import Shift, image_to_gray_float


@dataclass(frozen=True)
class RegistrationConfig:
    upsample_factor: int = 100
    min_confidence: float = 0.20
    max_shift_lr: float = 2.0
    highpass_sigma: float = 2.0
    ecc_iterations: int = 100


@dataclass(frozen=True)
class RegistrationResult:
    frame_index: int
    forward_shift_lr: Shift
    alignment_shift_lr: Shift
    confidence: float
    normalized_correlation: float
    residual_mse: float
    overlap_ratio: float
    accepted: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "forward_shift_lr": self.forward_shift_lr.as_tuple(),
            "alignment_shift_lr": self.alignment_shift_lr.as_tuple(),
            "confidence": self.confidence,
            "normalized_correlation": self.normalized_correlation,
            "residual_mse": self.residual_mse,
            "overlap_ratio": self.overlap_ratio,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def _validate_frames(frames: Sequence[np.ndarray]) -> None:
    if not frames:
        raise ValueError("No frames provided.")
    shape = frames[0].shape
    if len(shape) not in {2, 3}:
        raise ValueError(f"Unsupported frame shape: {shape}")
    for index, frame in enumerate(frames[1:], start=1):
        if frame.shape != shape:
            raise ValueError(
                f"Frame {index} has shape {frame.shape}; expected {shape}."
            )


def _registration_image(image: np.ndarray, highpass_sigma: float) -> np.ndarray:
    gray = image_to_gray_float(image).astype(np.float32)
    if highpass_sigma > 0:
        low = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=highpass_sigma,
            sigmaY=highpass_sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        gray = gray - low
    gray -= float(gray.mean())
    std = float(gray.std())
    if std < 1e-7:
        raise ValueError("Frame has insufficient texture for phase registration.")
    gray /= std
    h, w = gray.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return (gray * window).astype(np.float32)


def estimate_shift_phase(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 100,
    highpass_sigma: float = 2.0,
    ecc_iterations: int = 100,
) -> Shift:
    """Return the shift that aligns ``moving`` to ``reference`` in LR pixels."""
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving frames must have the same shape.")
    ref = _registration_image(reference, highpass_sigma)
    mov = _registration_image(moving, highpass_sigma)
    shift_yx, _, _ = phase_cross_correlation(
        ref,
        mov,
        upsample_factor=max(int(upsample_factor), 1),
        normalization="phase",
    )
    dy, dx = float(shift_yx[0]), float(shift_yx[1])
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("Registration produced a non-finite shift.")

    # ECC estimates the forward transform (reference -> moving). It removes the
    # interpolation bias that phase correlation can retain on sub-pixel shifts.
    if ecc_iterations > 0:
        reference_gray = image_to_gray_float(reference).astype(np.float32)
        moving_gray = image_to_gray_float(moving).astype(np.float32)
        forward_matrix = np.array(
            [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
            dtype=np.float32,
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(ecc_iterations),
            1e-6,
        )
        try:
            _, forward_matrix = cv2.findTransformECC(
                reference_gray,
                moving_gray,
                forward_matrix,
                cv2.MOTION_TRANSLATION,
                criteria,
                None,
                5,
            )
            dx = -float(forward_matrix[0, 2])
            dy = -float(forward_matrix[1, 2])
        except cv2.error:
            pass
    return Shift(dx=dx, dy=dy)


def warp_lr_to_reference(
    frame: np.ndarray,
    alignment_shift: Shift,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = np.array(
        [[1.0, 0.0, alignment_shift.dx], [0.0, 1.0, alignment_shift.dy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        frame.astype(np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )


def _alignment_quality(
    reference: np.ndarray,
    moving: np.ndarray,
    alignment_shift: Shift,
) -> tuple[float, float, float]:
    ref = image_to_gray_float(reference)
    mov = image_to_gray_float(moving)
    aligned = warp_lr_to_reference(mov, alignment_shift, cv2.BORDER_CONSTANT)

    mask = warp_lr_to_reference(
        np.ones_like(mov, dtype=np.float32),
        alignment_shift,
        cv2.BORDER_CONSTANT,
    )
    valid = mask > 0.999
    overlap = float(np.mean(valid))
    if int(valid.sum()) < 16:
        return 0.0, float("inf"), overlap

    a = ref[valid].astype(np.float64)
    b = aligned[valid].astype(np.float64)
    residual_mse = float(np.mean((a - b) ** 2))
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    ncc = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
    ncc = float(np.clip(ncc, -1.0, 1.0))
    return ncc, residual_mse, overlap


def register_sequence(
    frames: Sequence[np.ndarray],
    cfg: RegistrationConfig | None = None,
) -> list[RegistrationResult]:
    """Estimate physical forward shifts relative to frame zero.

    Phase correlation returns the transform needed to align each moving frame.
    The image-formation model needs the opposite transform, so both values are
    retained explicitly in every result.
    """
    _validate_frames(frames)
    cfg = cfg or RegistrationConfig()
    if cfg.upsample_factor < 1:
        raise ValueError("upsample_factor must be at least 1.")
    if not 0.0 <= cfg.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if cfg.max_shift_lr <= 0:
        raise ValueError("max_shift_lr must be positive.")
    if cfg.ecc_iterations < 0:
        raise ValueError("ecc_iterations must be non-negative.")

    results = [
        RegistrationResult(
            frame_index=0,
            forward_shift_lr=Shift(0.0, 0.0),
            alignment_shift_lr=Shift(0.0, 0.0),
            confidence=1.0,
            normalized_correlation=1.0,
            residual_mse=0.0,
            overlap_ratio=1.0,
            accepted=True,
            reason="reference",
        )
    ]
    reference = frames[0]

    for index, frame in enumerate(frames[1:], start=1):
        try:
            alignment = estimate_shift_phase(
                reference,
                frame,
                upsample_factor=cfg.upsample_factor,
                highpass_sigma=cfg.highpass_sigma,
                ecc_iterations=cfg.ecc_iterations,
            )
            forward = Shift(-alignment.dx, -alignment.dy)
            ncc, residual_mse, overlap = _alignment_quality(
                reference,
                frame,
                alignment,
            )
            confidence = float(np.clip(max(ncc, 0.0) * overlap, 0.0, 1.0))
            magnitude_ok = (
                abs(forward.dx) <= cfg.max_shift_lr
                and abs(forward.dy) <= cfg.max_shift_lr
            )
            confidence_ok = confidence >= cfg.min_confidence
            accepted = magnitude_ok and confidence_ok
            if not magnitude_ok:
                reason = "shift_out_of_range"
            elif not confidence_ok:
                reason = "low_confidence"
            else:
                reason = "accepted"
        except (ValueError, FloatingPointError) as exc:
            alignment = Shift(0.0, 0.0)
            forward = Shift(0.0, 0.0)
            ncc = 0.0
            residual_mse = float("inf")
            overlap = 0.0
            confidence = 0.0
            accepted = False
            reason = f"registration_failed: {exc}"

        results.append(
            RegistrationResult(
                frame_index=index,
                forward_shift_lr=forward,
                alignment_shift_lr=alignment,
                confidence=confidence,
                normalized_correlation=ncc,
                residual_mse=residual_mse,
                overlap_ratio=overlap,
                accepted=accepted,
                reason=reason,
            )
        )
    return results


def estimate_shifts_sequence(
    frames: Sequence[np.ndarray],
    upsample_factor: int = 100,
) -> list[Shift]:
    """Compatibility helper returning moving-to-reference alignment shifts."""
    if not frames:
        raise ValueError("No frames provided.")
    results = register_sequence(
        frames,
        RegistrationConfig(
            upsample_factor=upsample_factor,
            min_confidence=0.0,
            max_shift_lr=float(max(frames[0].shape[:2])),
        ),
    )
    return [result.alignment_shift_lr for result in results]


def estimate_forward_shifts_sequence(
    frames: Sequence[np.ndarray],
    cfg: RegistrationConfig | None = None,
) -> list[Shift]:
    return [result.forward_shift_lr for result in register_sequence(frames, cfg)]
