from pixelshiftsr.synthetic import make_synthetic_hr
from pixelshiftsr.utils import Shift
from pixelshiftsr.degradation import DegradationConfig, degrade_sequence
from pixelshiftsr.reconstruction import ReconstructionConfig, reconstruct


def test_pipeline_smoke():
    hr = make_synthetic_hr(64, 64, channels=3, seed=0)
    shifts = [Shift(0, 0), Shift(0.5, 0), Shift(0, 0.5), Shift(0.5, 0.5)]
    frames, _ = degrade_sequence(
        hr,
        DegradationConfig(
            scale=2,
            shifts=shifts,
            shift_unit="lr",
            blur_kernel=1,
            blur_stage="none",
            noise_poisson_a=0.0,
            noise_gaussian_b=0.0,
        ),
    )
    sr, info = reconstruct(frames, ReconstructionConfig(scale=2, shifts=shifts, shift_unit="lr", mode="fuse"))
    assert sr.shape == hr.shape
    assert sr.min() >= 0
    assert sr.max() <= 1
    assert info["mode"] == "fuse"
