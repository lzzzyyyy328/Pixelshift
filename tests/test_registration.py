import numpy as np

from pixelshiftsr.degradation import warp_hr
from pixelshiftsr.registration import RegistrationConfig, register_sequence
from pixelshiftsr.utils import Shift


def test_registration_reports_forward_shift_with_correct_sign():
    rng = np.random.default_rng(4)
    reference = rng.random((96, 96), dtype=np.float32)
    expected = Shift(0.45, -0.35)
    moving = warp_hr(reference, expected, border_mode="reflect")

    results = register_sequence(
        [reference, moving],
        RegistrationConfig(
            upsample_factor=100,
            min_confidence=0.1,
            max_shift_lr=2.0,
            highpass_sigma=1.5,
        ),
    )

    measured = results[1]
    assert measured.accepted
    assert measured.confidence > 0.65
    assert abs(measured.forward_shift_lr.dx - expected.dx) < 0.03
    assert abs(measured.forward_shift_lr.dy - expected.dy) < 0.03
    assert abs(measured.alignment_shift_lr.dx + expected.dx) < 0.03


def test_registration_rejects_textureless_frame():
    frames = [
        np.zeros((64, 64), dtype=np.float32),
        np.ones((64, 64), dtype=np.float32),
    ]
    results = register_sequence(frames)

    assert results[0].accepted
    assert not results[1].accepted
    assert results[1].confidence == 0.0
    assert results[1].reason.startswith("registration_failed")
