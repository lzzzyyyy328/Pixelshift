from pixelshiftsr.calibration import calibrate_forward_model
from pixelshiftsr.degradation import DegradationConfig, degrade_sequence
from pixelshiftsr.synthetic import make_synthetic_hr
from pixelshiftsr.utils import Shift


def test_calibration_recovers_defocus_kernel():
    hr = make_synthetic_hr(64, 64, channels=3, seed=3)
    shifts = [
        Shift(0.0, 0.0),
        Shift(0.5, 0.0),
        Shift(0.0, 0.5),
        Shift(0.5, 0.5),
    ]
    frames, shifts_hr = degrade_sequence(
        hr,
        DegradationConfig(
            scale=2,
            shifts=shifts,
            blur_type="defocus",
            blur_kernel=3,
            blur_stage="before_downsample",
            noise_poisson_a=0.0002,
            noise_gaussian_b=0.00002,
            seed=7,
        ),
    )

    report, predictions = calibrate_forward_model(
        hr,
        frames,
        shifts_hr,
        scale=2,
        blur_types=["defocus", "gaussian"],
        blur_kernels=[1, 3, 5],
        blur_sigmas=[0.6, 1.0],
    )

    assert report["best_model"]["blur_type"] == "defocus"
    assert report["best_model"]["blur_kernel"] == 3
    assert len(predictions) == len(frames)
    assert report["best_model"]["noise_model"]["poisson_a"] >= 0
    assert report["best_model"]["noise_model"]["gaussian_b"] >= 0
