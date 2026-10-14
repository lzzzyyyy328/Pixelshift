import json

from pixelshiftsr.cli import build_parser
from pixelshiftsr.degradation import DegradationConfig, degrade_sequence
from pixelshiftsr.io import save_image, save_image_sequence
from pixelshiftsr.synthetic import make_synthetic_hr
from pixelshiftsr.utils import Shift


def test_benchmark_command_writes_ranked_report(tmp_path):
    hr = make_synthetic_hr(32, 32, channels=3, seed=9)
    shifts = [
        Shift(0.0, 0.0),
        Shift(0.5, 0.0),
        Shift(0.0, 0.5),
        Shift(0.5, 0.5),
    ]
    frames, _ = degrade_sequence(
        hr,
        DegradationConfig(
            scale=2,
            shifts=shifts,
            blur_type="none",
            blur_stage="none",
            noise_poisson_a=0.0,
            noise_gaussian_b=0.0,
        ),
    )
    hr_path = tmp_path / "hr.png"
    lr_directory = tmp_path / "lr"
    output = tmp_path / "benchmark"
    save_image(hr_path, hr, dtype="uint16")
    save_image_sequence(lr_directory, frames, dtype="uint16")

    argv = [
        "benchmark",
        "--input",
        str(lr_directory),
        "--input-pattern",
        "lr_*.png",
        "--output",
        str(output),
        "--gt",
        str(hr_path),
        "--variants",
        "bicubic,fuse,ibp_huber",
        "--blur-type",
        "none",
        "--blur-stage",
        "none",
        "--iterations",
        "2",
        "--no-progress",
    ]
    args = build_parser().parse_args(argv)
    args._config_meta = {}
    args.func(args)

    report = json.loads((output / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["ranking_metric"] == "PSNR"
    assert set(report["variants"]) == {"bicubic", "fuse", "ibp_huber"}
    assert report["best_variant"] in report["variants"]
