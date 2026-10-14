from pixelshiftsr.degradation import DegradationConfig, degrade_sequence
from pixelshiftsr.metrics import psnr, reprojection_metrics
from pixelshiftsr.reconstruction import (
    ReconstructionConfig,
    bicubic_baseline,
    reconstruct,
)
from pixelshiftsr.synthetic import make_synthetic_hr
from pixelshiftsr.utils import Shift


SHIFTS = [
    Shift(0.0, 0.0),
    Shift(0.5, 0.0),
    Shift(0.0, 0.5),
    Shift(0.5, 0.5),
]


def test_robust_ibp_improves_over_bicubic():
    hr = make_synthetic_hr(64, 64, channels=3, seed=2)
    frames, shifts_hr = degrade_sequence(
        hr,
        DegradationConfig(
            scale=2,
            shifts=SHIFTS,
            blur_type="none",
            blur_stage="none",
            noise_poisson_a=0.0,
            noise_gaussian_b=0.0,
        ),
    )
    baseline = bicubic_baseline(frames, scale=2)
    sr, info = reconstruct(
        frames,
        ReconstructionConfig(
            scale=2,
            shifts=SHIFTS,
            mode="ibp",
            iterations=25,
            step_size=0.35,
            blur_type="none",
            blur_stage="none",
            show_progress=False,
        ),
    )

    assert psnr(hr, sr) > psnr(hr, baseline) + 3.0
    losses = info["optimization"]["losses"]
    assert losses[-1] < losses[0]

    reprojection = reprojection_metrics(
        sr,
        frames,
        shifts_hr,
        scale=2,
        blur_type="none",
        blur_kernel=1,
        blur_sigma=0.0,
        blur_stage="none",
        border_mode="replicate",
        interpolation="area",
    )
    assert reprojection["aggregate"]["WeightedMSE"] < 0.0002
