from __future__ import annotations

from typing import Sequence
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .utils import Shift, image_to_gray_float


def estimate_shift_phase(reference: np.ndarray, moving: np.ndarray, upsample_factor: int = 100) -> Shift:
    """Estimate moving-to-reference translation using phase correlation.

    Returns dx, dy in pixels of the input image grid. Positive values mean the
    moving image should be shifted by (dx, dy) to align with the reference.
    """
    ref = image_to_gray_float(reference)
    mov = image_to_gray_float(moving)
    shift_yx, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample_factor)
    dy, dx = float(shift_yx[0]), float(shift_yx[1])
    return Shift(dx=dx, dy=dy)


def estimate_shifts_sequence(frames: Sequence[np.ndarray], upsample_factor: int = 100) -> list[Shift]:
    if len(frames) == 0:
        raise ValueError("No frames provided.")
    ref = frames[0]
    shifts = [Shift(0.0, 0.0)]
    for frame in frames[1:]:
        shifts.append(estimate_shift_phase(ref, frame, upsample_factor=upsample_factor))
    return shifts


def warp_lr_to_reference(frame: np.ndarray, shift: Shift, border_mode: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.array([[1.0, 0.0, shift.dx], [0.0, 1.0, shift.dy]], dtype=np.float32)
    return cv2.warpAffine(frame.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
