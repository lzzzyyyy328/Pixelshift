from __future__ import annotations

import cv2
import numpy as np


def make_phone_white_screen_led_hr(
    height: int = 2048,
    width: int = 2048,
    *,
    subpixel_width: int = 1,
    gamma_display: float = 2.2,
    backlight_vignette: float = 0.10,
    warm_rgb: tuple[float, float, float] = (1.0, 0.985, 0.97),
    rotation_deg: float = 0.12,
    moire_beat_amp: float = 0.02,
    read_noise_std: float = 0.002,
    seed: int = 0,
) -> np.ndarray:
    """High-resolution 'white screen' close-up: repeating RGB subpixel stripes + mild optical artifacts.

    Intended to mimic a macro photo of a phone in white mode (lit R/G/B columns are visible when zoomed in).
    Values are linear-light in [0, 1]; downstream degradation models optics + sensor sampling.

    Parameters
    ----------
    subpixel_width
        HR pixels per R/G/B stripe column (1 = finest stripe grid for SR stress-test).
    gamma_display
        Assume panel emits with this gamma; we linearize for physics-forward simulation.
    backlight_vignette
        Edge darkening strength (lens + uniform backlight nonuniformity).
    rotation_deg
        Slight screen-to-sensor misalignment (degrees).
    moire_beat_amp
        Low-amplitude beat pattern to emulate sampling moiré sensitivity without a second periodic layer.
    """
    rng = np.random.default_rng(seed)
    sp = max(1, int(subpixel_width))
    yy, xx = np.indices((height, width), dtype=np.float32)
    col = xx.astype(np.int32) // sp
    phase = (col % 3).astype(np.float32)
    r = (phase == 0).astype(np.float32) * warm_rgb[0]
    g = (phase == 1).astype(np.float32) * warm_rgb[1]
    b = (phase == 2).astype(np.float32) * warm_rgb[2]
    rgb = np.stack([r, g, b], axis=-1)

    # Panel emission often gamma-encoded; convert to linear for forward optical model.
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, gamma_display, dtype=np.float32)

    # Soft vignette (center brighter).
    cy, cx = (height - 1) * 0.5, (width - 1) * 0.5
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(rr.max()) + 1e-6
    vign = 1.0 - backlight_vignette * (rr / rmax) ** 2
    rgb *= vign[..., None]

    # Subtle beat (diagonal) — optional moiré sensitivity for experiments.
    if moire_beat_amp > 0:
        beat = moire_beat_amp * np.sin(2 * np.pi * (xx / (6.0 * sp) + yy / (7.0 * sp)))
        rgb *= np.clip(1.0 + beat[..., None], 0.85, 1.15)

    # Tiny rotation (screen vs camera).
    if abs(rotation_deg) > 1e-6:
        h, w = height, width
        center = (w * 0.5, h * 0.5)
        M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        rgb = cv2.warpAffine(
            rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    if read_noise_std > 0:
        rgb += rng.normal(0.0, read_noise_std, size=rgb.shape).astype(np.float32)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_synthetic_hr(height: int = 512, width: int = 512, channels: int = 3, seed: int = 0) -> np.ndarray:
    """Create a deterministic HR test image with edges, curves, textures and text."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = 0.35 + 0.25 * x + 0.20 * y
    checker = (((np.arange(height)[:, None] // 12 + np.arange(width)[None, :] // 12) % 2) * 0.18).astype(np.float32)
    waves = 0.08 * np.sin(2 * np.pi * (8 * x + 3 * y))
    img = np.clip(base + checker + waves, 0, 1)

    rgb = np.stack([
        img,
        np.clip(0.85 * img + 0.10 * np.sin(2 * np.pi * 4 * y), 0, 1),
        np.clip(0.75 * img + 0.15 * np.cos(2 * np.pi * 5 * x), 0, 1),
    ], axis=-1)

    canvas = np.round(rgb * 255).astype(np.uint8)
    cv2.rectangle(canvas, (width // 12, height // 10), (width // 2, height // 3), (255, 255, 255), 2)
    cv2.circle(canvas, (int(width * 0.72), int(height * 0.28)), min(height, width) // 9, (20, 20, 20), 3)
    cv2.line(canvas, (width // 8, int(height * 0.78)), (int(width * 0.90), int(height * 0.62)), (255, 255, 255), 2)
    cv2.putText(canvas, "PixelShiftSR", (width // 14, int(height * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, max(width, height) / 650, (10, 10, 10), 2, cv2.LINE_AA)

    # Add fine random dots for aliasing/super-resolution evaluation.
    for _ in range(max(50, height * width // 6000)):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.circle(canvas, (cx, cy), int(rng.integers(1, 3)), color, -1)

    out = canvas.astype(np.float32) / 255.0
    if channels == 1:
        return (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]).astype(np.float32)
    return out


def make_rggb_color_mask(height: int, width: int) -> np.ndarray:
    """Generate the same 3-channel RGGB color mask used in the uploaded prototype."""
    mask = np.zeros((height, width, 3), dtype=np.float32)
    yy, xx = np.indices((height, width))
    mask[(yy % 2 == 0) & (xx % 2 == 0), 0] = 1.0
    mask[(yy % 2 == 0) & (xx % 2 == 1), 1] = 1.0
    mask[(yy % 2 == 1) & (xx % 2 == 0), 1] = 1.0
    mask[(yy % 2 == 1) & (xx % 2 == 1), 2] = 1.0
    return mask
