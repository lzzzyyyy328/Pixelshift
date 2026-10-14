import numpy as np
import pytest

from pixelshiftsr.degradation import (
    DegradationConfig,
    add_poisson_gaussian_noise,
    degrade_one,
    degrade_sequence,
)
from pixelshiftsr.utils import Shift


def test_degrade_one_applies_forward_shift():
    image = np.zeros((16, 16), dtype=np.float32)
    image[5, 5] = 1.0

    shifted = degrade_one(
        image,
        Shift(2.0, 1.0),
        scale=1,
        blur_type="none",
        blur_stage="none",
        border_mode="constant",
        interpolation="nearest",
    )

    assert np.unravel_index(np.argmax(shifted), shifted.shape) == (6, 7)


def test_degrade_sequence_changes_with_shift():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:16, 9:17] = 1.0
    frames, shifts_hr = degrade_sequence(
        image,
        DegradationConfig(
            scale=2,
            shifts=[Shift(0.0, 0.0), Shift(0.5, 0.0)],
            blur_type="none",
            blur_stage="none",
            noise_poisson_a=0.0,
            noise_gaussian_b=0.0,
            border_mode="constant",
        ),
    )

    assert shifts_hr[1] == Shift(1.0, 0.0)
    assert not np.allclose(frames[0], frames[1])


def test_noise_coefficients_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        add_poisson_gaussian_noise(
            np.zeros((8, 8), dtype=np.float32),
            -0.1,
            0.0,
            np.random.default_rng(0),
        )
