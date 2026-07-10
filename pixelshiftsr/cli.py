from __future__ import annotations

import argparse
import json
from pathlib import Path

from .degradation import DegradationConfig, degrade_sequence
from .io import read_image, read_image_sequence, save_image, save_image_sequence
from .metrics import evaluate
from .reconstruction import ReconstructionConfig, reconstruct, bicubic_baseline
from .synthetic import make_phone_white_screen_led_hr, make_synthetic_hr, make_rggb_color_mask
from .utils import ensure_dir, parse_shifts
from .visualization import make_comparison, make_error_map


def _apply_legacy_noise_args(args: argparse.Namespace) -> None:
    """Map the old Gaussian std flags to the current Poisson-Gaussian model."""
    noise_std = getattr(args, "noise_std", None)
    if noise_std is None:
        return

    std = float(noise_std)
    if getattr(args, "noise_in_255", False):
        std /= 255.0

    args.noise_poisson_a = 0.0
    args.noise_gaussian_b = std * std


def _forward_pipeline_description(blur_stage: str) -> tuple[list[str], str]:
    """Human-readable forward chain matching degradation.degrade_one (+ noise in sequence)."""
    b = (blur_stage or "before_downsample").strip().lower()
    if b == "none":
        return (
            ["warp_hr(shift)", "downsample", "add_poisson_gaussian_noise"],
            "HR → sub-pixel warp → integral downsample → sensor noise (no optical blur).",
        )
    if b == "before_downsample":
        return (
            ["blur(optics)", "warp_hr(shift)", "downsample", "add_poisson_gaussian_noise"],
            "HR → blur → sub-pixel warp → integral downsample → sensor noise.",
        )
    if b == "after_downsample":
        return (
            ["warp_hr(shift)", "downsample", "blur(on_LR)", "add_poisson_gaussian_noise"],
            "HR → warp → downsample → LR-domain blur → sensor noise (nonstandard optics order).",
        )
    return ([], f"unknown blur_stage={blur_stage!r}")


def _build_degradation_meta(
    args: argparse.Namespace,
    hr,
    shifts,
    shifts_hr,
    paths: list,
) -> dict:
    """Rich metadata for reproducibility; blur_stage matches args actually passed to DegradationConfig."""
    blur_stage = (args.blur_stage or "").strip().lower()
    blur_applied = blur_stage not in ("none", "")
    noise_disabled = float(args.noise_poisson_a) <= 0.0 and float(args.noise_gaussian_b) <= 0.0
    steps, summary = _forward_pipeline_description(blur_stage)

    hr_path = Path(args.hr)
    meta = {
        "schema_version": 2,
        "tool": "pixelshiftsr.degrade",
        "summary": summary,
        "hr_path": str(hr_path.resolve()),
        "hr_shape_hwc": [int(x) for x in hr.shape],
        "output_directory": str(Path(paths[0]).parent.resolve()) if paths else "",
        "scale": args.scale,
        "shift_unit": args.shift_unit,
        "input_shifts_lr": [s.as_tuple() for s in shifts],
        "shifts_hr_pixels": [s.as_tuple() for s in shifts_hr],
        "forward_pipeline_steps": steps,
        "blur": {
            "blur_stage": args.blur_stage,
            "blur_applied": blur_applied,
            "blur_type": args.blur_type,
            "blur_kernel": args.blur_kernel,
            "blur_sigma": args.blur_sigma,
            "note": "When blur_stage is 'none', blur_type/kernel/sigma are ignored by degrade_one.",
        },
        "sampling": {
            "downsample_interpolation": args.downsample,
            "border_mode": args.border,
        },
        "noise": {
            "noise_poisson_a": args.noise_poisson_a,
            "noise_gaussian_b": args.noise_gaussian_b,
            "noise_disabled": noise_disabled,
        },
        "seed": args.seed,
        "frames": [str(Path(p).resolve()) for p in paths],
        "reconstruct_hints": {
            "for_ibp_match_degrade": (
                "Use the same --scale, --shifts, --shift-unit, --blur-stage, --blur-type, "
                "--blur-kernel, --blur-sigma, --downsample, --border as this degradation run."
            ),
            "example_blur_stage": args.blur_stage,
        },
        # Flat aliases (schema v1 compatibility)
        "hr": str(args.hr),
        "input_shifts": [s.as_tuple() for s in shifts],
        "blur_type": args.blur_type,
        "blur_kernel": args.blur_kernel,
        "blur_sigma": args.blur_sigma,
    }
    return meta


def _add_common_shift_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale", type=int, default=2, help="Super-resolution scale. Default: 2")
    parser.add_argument("--shifts", type=str, default="0,0;0.5,0;0,0.5;0.5,0.5", help="Shifts as dx,dy;dx,dy. Default is 2x2 half-LR-pixel pattern.")
    parser.add_argument("--shift-unit", choices=["lr", "hr"], default="lr", help="Unit of --shifts. 'lr' is physical LR-pixel unit; 'hr' matches the uploaded prototype script.")


def cmd_make_demo(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output)
    if args.preset == "phone-white":
        hr = make_phone_white_screen_led_hr(
            args.height,
            args.width,
            subpixel_width=args.subpixel_width,
            seed=args.seed,
        )
    else:
        hr = make_synthetic_hr(args.height, args.width, channels=args.channels, seed=args.seed)
    save_image(out / "hr_gt.png", hr)
    if args.make_bayer_mask:
        mask = make_rggb_color_mask(args.height, args.width)
        save_image(out / "hr_rggb_mask.png", mask)
    print(f"Created demo HR image: {out / 'hr_gt.png'}")


def cmd_degrade(args: argparse.Namespace) -> None:
    _apply_legacy_noise_args(args)
    out = ensure_dir(args.output)
    hr = read_image(args.hr)
    shifts = parse_shifts(args.shifts, args.scale)
    cfg = DegradationConfig(
        scale=args.scale,
        shifts=shifts,
        shift_unit=args.shift_unit,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        noise_poisson_a=args.noise_poisson_a,
        noise_gaussian_b=args.noise_gaussian_b,
        seed=args.seed,
        interpolation=args.downsample,
        border_mode=args.border,
    )
    frames, shifts_hr = degrade_sequence(hr, cfg)
    paths = save_image_sequence(out, frames, prefix="lr")
    meta = _build_degradation_meta(args, hr, shifts, shifts_hr, paths)
    (out / "degradation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(paths)} LR frames to {out}")
    print(f"Saved metadata: {out / 'degradation_meta.json'}")


def cmd_reconstruct(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output)
    frames = read_image_sequence(args.input)
    shifts = parse_shifts(args.shifts, args.scale)
    cfg = ReconstructionConfig(
        scale=args.scale,
        shifts=shifts,
        shift_unit=args.shift_unit,
        mode=args.mode,
        iterations=args.iterations,
        step_size=args.step_size,
        tv_weight=args.tv_weight,
        laplacian_weight=args.laplacian_weight,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        interpolation=args.downsample,
        border_mode=args.border,
    )
    sr, info = reconstruct(frames, cfg)
    save_image(out / "sr_result.png", sr)
    # Also save bicubic baseline for visual comparison.
    bicubic = bicubic_baseline(frames, scale=args.scale)
    save_image(out / "bicubic_baseline.png", bicubic)

    gt = read_image(args.gt) if args.gt else None
    report = dict(info)
    if gt is not None:
        report["metrics_sr"] = evaluate(gt, sr)
        report["metrics_bicubic"] = evaluate(gt, bicubic)
        make_error_map(gt, sr, out / "error_map.png")
    if args.save_debug:
        make_comparison(gt, frames[0], sr, out / "comparison.png", scale=args.scale)
    (out / "reconstruction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved SR result: {out / 'sr_result.png'}")
    print(f"Saved report: {out / 'reconstruction_report.json'}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    _apply_legacy_noise_args(args)
    out = ensure_dir(args.output)
    hr_path = Path(args.hr) if args.hr else (out / "hr_gt.png")
    if args.hr is None:
        if getattr(args, "preset", "default") == "phone-white":
            hr = make_phone_white_screen_led_hr(
                args.height, args.width, subpixel_width=args.subpixel_width, seed=args.seed
            )
        else:
            hr = make_synthetic_hr(args.height, args.width, channels=args.channels, seed=args.seed)
        save_image(hr_path, hr)
    else:
        hr = read_image(hr_path)
    lr_dir = ensure_dir(out / "lr_frames")
    shifts = parse_shifts(args.shifts, args.scale)
    deg_cfg = DegradationConfig(
        scale=args.scale,
        shifts=shifts,
        shift_unit=args.shift_unit,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        noise_poisson_a=args.noise_poisson_a,
        noise_gaussian_b=args.noise_gaussian_b,
        seed=args.seed,
        interpolation=args.downsample,
        border_mode=args.border,
    )
    frames, shifts_hr = degrade_sequence(hr, deg_cfg)
    save_image_sequence(lr_dir, frames, prefix="lr")

    rec_cfg = ReconstructionConfig(
        scale=args.scale,
        shifts=shifts,
        shift_unit=args.shift_unit,
        mode=args.mode,
        iterations=args.iterations,
        step_size=args.step_size,
        tv_weight=args.tv_weight,
        laplacian_weight=args.laplacian_weight,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        interpolation=args.downsample,
        border_mode=args.border,
    )
    sr, info = reconstruct(frames, rec_cfg)
    save_image(out / "sr_result.png", sr)
    bicubic = bicubic_baseline(frames, scale=args.scale)
    save_image(out / "bicubic_baseline.png", bicubic)
    make_comparison(hr, frames[0], sr, out / "comparison.png", scale=args.scale)
    make_error_map(hr, sr, out / "error_map.png")

    report = {
        "hr_path": str(hr_path),
        "lr_dir": str(lr_dir),
        "scale": args.scale,
        "shift_unit": args.shift_unit,
        "input_shifts": [s.as_tuple() for s in shifts],
        "shifts_hr_pixels": [s.as_tuple() for s in shifts_hr],
        "reconstruction": info,
        "metrics_sr": evaluate(hr, sr),
        "metrics_bicubic": evaluate(hr, bicubic),
    }
    (out / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pipeline finished. Results are in: {out}")
    print(json.dumps(report["metrics_sr"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PixelShiftSR: degrade HR into shifted LR frames and reconstruct 2x HR image.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-demo", help="Create a synthetic HR image for testing.")
    p.add_argument("--output", type=str, default="runs/demo")
    p.add_argument(
        "--preset",
        choices=["default", "phone-white"],
        default="phone-white",
        help="'phone-white': RGB subpixel stripe white-screen (2048-style macro); 'default': generic test pattern.",
    )
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--subpixel-width", type=int, default=1, help="HR pixels per R/G/B column when preset=phone-white.")
    p.add_argument("--channels", type=int, choices=[1, 3], default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--make-bayer-mask", action="store_true", help="Also create the RGGB color mask used by the original prototype.")
    p.set_defaults(func=cmd_make_demo)

    p = sub.add_parser("degrade", help="Degrade one HR image into four shifted LR images.")
    p.add_argument("--hr", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    _add_common_shift_args(p)
    p.add_argument("--blur-type", choices=["defocus", "gaussian"], default="defocus", help="Blur type for degradation. 'defocus' simulates lens blur (recommended for LED).")
    p.add_argument("--blur-kernel", type=int, default=5, help="Blur kernel size. For defocus, this is the disk diameter.")
    p.add_argument("--blur-sigma", type=float, default=0.0, help="Gaussian blur sigma (only used when blur-type=gaussian).")
    p.add_argument("--blur-stage", choices=["before_downsample", "after_downsample", "none"], default="before_downsample", help="When to apply blur (before_downsample is physically correct).")
    p.add_argument("--noise-poisson-a", type=float, default=0.001, help="Poisson noise coefficient (signal-dependent photon noise).")
    p.add_argument("--noise-gaussian-b", type=float, default=0.0001, help="Gaussian noise variance (sensor read noise, signal-independent).")
    p.add_argument("--noise-std", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--noise-in-255", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--downsample", choices=["area", "linear", "cubic", "nearest", "lanczos"], default="area")
    p.add_argument("--border", choices=["replicate", "reflect", "constant"], default="replicate")
    p.set_defaults(func=cmd_degrade)

    p = sub.add_parser("reconstruct", help="Reconstruct HR image from shifted LR frames.")
    p.add_argument("--input", type=str, required=True, help="Folder containing lr_00.png ... lr_03.png")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--gt", type=str, default=None, help="Optional HR ground truth for metrics.")
    _add_common_shift_args(p)
    p.add_argument("--mode", choices=["bicubic", "interlace", "fuse", "ibp", "inverse"], default="ibp")
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--step-size", type=float, default=0.20)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--laplacian-weight", type=float, default=0.0)
    p.add_argument("--blur-type", choices=["defocus", "gaussian"], default="defocus", help="Blur type for forward model in IBP.")
    p.add_argument("--blur-kernel", type=int, default=5, help="Blur kernel size.")
    p.add_argument("--blur-sigma", type=float, default=0.0, help="Gaussian blur sigma (only used when blur-type=gaussian).")
    p.add_argument("--blur-stage", choices=["before_downsample", "after_downsample", "none"], default="before_downsample")
    p.add_argument("--downsample", choices=["area", "linear", "cubic", "nearest", "lanczos"], default="area")
    p.add_argument("--border", choices=["replicate", "reflect", "constant"], default="replicate")
    p.add_argument("--save-debug", action="store_true")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("pipeline", help="One-click HR -> LR degradation -> HR reconstruction -> metrics.")
    p.add_argument("--hr", type=str, default=None, help="Optional HR image. If omitted, a synthetic HR image is generated.")
    p.add_argument("--output", type=str, default="runs/pipeline")
    p.add_argument(
        "--preset",
        choices=["default", "phone-white"],
        default="phone-white",
        help="When --hr is omitted: synthetic HR style (phone-white = RGB subpixel white-screen).",
    )
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--subpixel-width", type=int, default=1, help="With preset=phone-white: HR pixels per R/G/B column.")
    p.add_argument("--channels", type=int, choices=[1, 3], default=3)
    _add_common_shift_args(p)
    p.add_argument("--mode", choices=["bicubic", "interlace", "fuse", "ibp", "inverse"], default="ibp")
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--step-size", type=float, default=0.20)
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument("--laplacian-weight", type=float, default=0.0)
    p.add_argument("--blur-type", choices=["defocus", "gaussian"], default="defocus", help="Blur type for degradation and IBP.")
    p.add_argument("--blur-kernel", type=int, default=5, help="Blur kernel size.")
    p.add_argument("--blur-sigma", type=float, default=0.0, help="Gaussian blur sigma (only used when blur-type=gaussian).")
    p.add_argument("--blur-stage", choices=["before_downsample", "after_downsample", "none"], default="before_downsample")
    p.add_argument("--noise-poisson-a", type=float, default=0.001, help="Poisson noise coefficient (signal-dependent).")
    p.add_argument("--noise-gaussian-b", type=float, default=0.0001, help="Gaussian noise variance (signal-independent).")
    p.add_argument("--noise-std", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--noise-in-255", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--downsample", choices=["area", "linear", "cubic", "nearest", "lanczos"], default="area")
    p.add_argument("--border", choices=["replicate", "reflect", "constant"], default="replicate")
    p.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
