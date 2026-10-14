from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import sys
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .calibration import calibrate_forward_model
from .config import apply_yaml_config, validate_namespace
from .degradation import DegradationConfig, degrade_one, degrade_sequence
from .io import (
    read_image,
    read_image_sequence_with_paths,
    save_image,
    save_image_sequence,
)
from .metrics import evaluate, no_reference_metrics, reprojection_metrics
from .reconstruction import ReconstructionConfig, bicubic_baseline, reconstruct
from .registration import (
    RegistrationConfig,
    RegistrationResult,
    register_sequence,
    warp_lr_to_reference,
)
from .reporting import input_manifest, namespace_config, provenance, write_json
from .synthetic import (
    make_phone_white_screen_led_hr,
    make_rggb_color_mask,
    make_synthetic_hr,
)
from .utils import Shift, convert_shifts_to_hr, ensure_dir, parse_shifts
from .visualization import (
    make_benchmark_comparison,
    make_comparison,
    make_convergence_plot,
    make_error_map,
)


@dataclass
class ResolvedSequence:
    frames: list[np.ndarray]
    paths: list[Path]
    shifts: list[Shift]
    shift_unit: str
    frame_weights: list[float]
    selected_indices: list[int]
    alignment_shifts_lr: list[Shift]
    registration_report: dict


def _apply_legacy_noise_args(args: argparse.Namespace) -> None:
    noise_std = getattr(args, "noise_std", None)
    if noise_std is None:
        return
    std = float(noise_std)
    if getattr(args, "noise_in_255", False):
        std /= 255.0
    args.noise_poisson_a = 0.0
    args.noise_gaussian_b = std * std


def _forward_steps(blur_stage: str) -> list[str]:
    stage = blur_stage.lower()
    if stage == "before_downsample":
        return ["optical_blur", "subpixel_shift", "pixel_integration", "sensor_noise"]
    if stage == "after_downsample":
        return ["subpixel_shift", "pixel_integration", "lr_blur", "sensor_noise"]
    return ["subpixel_shift", "pixel_integration", "sensor_noise"]


def _build_degradation_report(
    args: argparse.Namespace,
    hr: np.ndarray,
    hr_path: Path,
    shifts: Sequence[Shift],
    shifts_hr: Sequence[Shift],
    frame_paths: Sequence[Path],
) -> dict:
    return {
        "schema_version": 3,
        "provenance": provenance(args.command),
        "configuration_file": getattr(args, "_config_meta", {}),
        "resolved_configuration": namespace_config(args),
        "input": {
            "hr_path": str(hr_path.resolve()),
            "hr_shape": list(hr.shape),
            "manifest": input_manifest([hr_path]),
        },
        "image_formation_model": {
            "shift_convention": "positive dx/dy moves image content right/down",
            "scale": args.scale,
            "shift_unit": args.shift_unit,
            "input_shifts": [shift.as_tuple() for shift in shifts],
            "shifts_hr_pixels": [shift.as_tuple() for shift in shifts_hr],
            "pipeline_steps": _forward_steps(args.blur_stage),
            "blur_type": args.blur_type,
            "blur_kernel": args.blur_kernel,
            "blur_sigma": args.blur_sigma,
            "blur_stage": args.blur_stage,
            "downsample": args.downsample,
            "border": args.border,
            "noise_poisson_a": args.noise_poisson_a,
            "noise_gaussian_b": args.noise_gaussian_b,
            "seed": args.seed,
        },
        "outputs": {
            "frame_count": len(frame_paths),
            "dtype": args.output_dtype,
            "manifest": input_manifest(frame_paths),
        },
    }


def _known_registration(
    frames: Sequence[np.ndarray],
    paths: Sequence[Path],
    shifts: Sequence[Shift],
    shift_unit: str,
    scale: int,
) -> ResolvedSequence:
    if len(frames) != len(shifts):
        raise ValueError(
            f"Found {len(frames)} frames but received {len(shifts)} known shifts."
        )
    if shift_unit == "lr":
        shifts_lr = list(shifts)
    else:
        shifts_lr = [Shift(item.dx / scale, item.dy / scale) for item in shifts]
    frame_rows = []
    for index, (path, shift_lr) in enumerate(zip(paths, shifts_lr)):
        frame_rows.append(
            {
                "frame_index": index,
                "file": str(path.resolve()),
                "forward_shift_lr": shift_lr.as_tuple(),
                "alignment_shift_lr": (-shift_lr.dx, -shift_lr.dy),
                "confidence": 1.0,
                "accepted": True,
                "reason": "known_shift",
            }
        )
    return ResolvedSequence(
        frames=list(frames),
        paths=list(paths),
        shifts=list(shifts),
        shift_unit=shift_unit,
        frame_weights=[1.0] * len(frames),
        selected_indices=list(range(len(frames))),
        alignment_shifts_lr=[Shift(-item.dx, -item.dy) for item in shifts_lr],
        registration_report={
            "mode": "known",
            "accepted_count": len(frames),
            "rejected_count": 0,
            "selected_indices": list(range(len(frames))),
            "frames": frame_rows,
        },
    )


def _estimated_registration(
    frames: Sequence[np.ndarray],
    paths: Sequence[Path],
    args: argparse.Namespace,
) -> ResolvedSequence:
    if args.min_accepted_frames < 1:
        raise ValueError("min_accepted_frames must be at least 1.")
    config = RegistrationConfig(
        upsample_factor=args.registration_upsample,
        min_confidence=args.min_registration_confidence,
        max_shift_lr=args.max_shift_lr,
        highpass_sigma=args.registration_highpass,
        ecc_iterations=args.registration_ecc_iterations,
    )
    results = register_sequence(frames, config)
    accepted_indices = [item.frame_index for item in results if item.accepted]
    if len(accepted_indices) < args.min_accepted_frames:
        raise ValueError(
            "Automatic registration accepted only "
            f"{len(accepted_indices)} frames; at least {args.min_accepted_frames} are required."
        )
    selected_indices = (
        list(range(len(frames))) if args.keep_rejected else accepted_indices
    )
    selected_results = [results[index] for index in selected_indices]
    weights = [item.confidence if item.accepted else 0.0 for item in selected_results]
    rows = []
    for result, path in zip(results, paths):
        row = result.as_dict()
        row["file"] = str(path.resolve())
        rows.append(row)
    return ResolvedSequence(
        frames=[frames[index] for index in selected_indices],
        paths=[paths[index] for index in selected_indices],
        shifts=[item.forward_shift_lr for item in selected_results],
        shift_unit="lr",
        frame_weights=weights,
        selected_indices=selected_indices,
        alignment_shifts_lr=[item.alignment_shift_lr for item in results],
        registration_report={
            "mode": "estimate",
            "configuration": {
                "upsample_factor": config.upsample_factor,
                "min_confidence": config.min_confidence,
                "max_shift_lr": config.max_shift_lr,
                "highpass_sigma": config.highpass_sigma,
                "ecc_iterations": config.ecc_iterations,
            },
            "accepted_count": len(accepted_indices),
            "rejected_count": len(results) - len(accepted_indices),
            "selected_indices": selected_indices,
            "frames": rows,
        },
    )


def _resolve_sequence(
    frames: Sequence[np.ndarray],
    paths: Sequence[Path],
    args: argparse.Namespace,
) -> ResolvedSequence:
    if len(frames) != len(paths):
        raise ValueError("Frame and path counts differ.")
    if args.registration == "estimate":
        return _estimated_registration(frames, paths, args)
    shifts = parse_shifts(args.shifts, args.scale)
    return _known_registration(frames, paths, shifts, args.shift_unit, args.scale)


def _reconstruction_config(
    args: argparse.Namespace,
    sequence: ResolvedSequence,
) -> ReconstructionConfig:
    return ReconstructionConfig(
        scale=args.scale,
        shifts=sequence.shifts,
        shift_unit=sequence.shift_unit,
        frame_weights=sequence.frame_weights,
        mode=args.mode,
        iterations=args.iterations,
        step_size=args.step_size,
        robust_loss=args.robust_loss,
        huber_delta=args.huber_delta,
        tolerance=args.tolerance,
        patience=args.patience,
        min_iterations=args.min_iterations,
        tv_weight=args.tv_weight,
        laplacian_weight=args.laplacian_weight,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        interpolation=args.downsample,
        border_mode=args.border,
        show_progress=not args.no_progress,
    )


def _reprojection(
    image: np.ndarray,
    sequence: ResolvedSequence,
    args: argparse.Namespace,
) -> dict:
    shifts_hr = convert_shifts_to_hr(
        sequence.shifts,
        args.scale,
        sequence.shift_unit,
    )
    return reprojection_metrics(
        image,
        sequence.frames,
        shifts_hr,
        scale=args.scale,
        blur_type=args.blur_type,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        blur_stage=args.blur_stage,
        border_mode=args.border,
        interpolation=args.downsample,
        frame_weights=sequence.frame_weights,
    )


def _save_debug_outputs(
    output: Path,
    all_frames: Sequence[np.ndarray],
    sequence: ResolvedSequence,
    sr: np.ndarray,
    args: argparse.Namespace,
) -> None:
    aligned_dir = ensure_dir(output / "registered_frames")
    for index, (frame, alignment) in enumerate(
        zip(all_frames, sequence.alignment_shifts_lr)
    ):
        aligned = warp_lr_to_reference(frame, alignment)
        save_image(aligned_dir / f"aligned_{index:02d}.png", aligned, dtype="uint16")

    residual_dir = ensure_dir(output / "reprojection_residuals")
    shifts_hr = convert_shifts_to_hr(sequence.shifts, args.scale, sequence.shift_unit)
    for index, (frame, shift_hr) in enumerate(zip(sequence.frames, shifts_hr)):
        prediction = degrade_one(
            sr,
            shift_hr,
            scale=args.scale,
            blur_type=args.blur_type,
            blur_kernel=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            blur_stage=args.blur_stage,
            border_mode=args.border,
            interpolation=args.downsample,
        )
        save_image(
            residual_dir / f"predicted_{index:02d}.png",
            prediction,
            dtype="uint16",
        )
        residual = np.clip(np.abs(frame - prediction) * 8.0, 0.0, 1.0)
        save_image(
            residual_dir / f"residual_x8_{index:02d}.png",
            residual,
            dtype="uint8",
        )


def _execute_reconstruction(
    args: argparse.Namespace,
    output: Path,
    all_frames: Sequence[np.ndarray],
    all_paths: Sequence[Path],
    sequence: ResolvedSequence,
    gt: np.ndarray | None,
    gt_path: Path | None,
    report_name: str,
    extra_report: dict | None = None,
) -> dict:
    config = _reconstruction_config(args, sequence)
    sr, reconstruction_info = reconstruct(sequence.frames, config)
    baseline = bicubic_baseline(
        sequence.frames,
        scale=args.scale,
        frame_weights=sequence.frame_weights,
    )

    save_image(output / "sr_result.png", sr, dtype="uint16")
    save_image(output / "sr_result.npy", sr)
    save_image(output / "bicubic_baseline.png", baseline, dtype="uint16")
    make_comparison(
        gt,
        sequence.frames[0],
        sr,
        output / "comparison.png",
        scale=args.scale,
        baseline=baseline,
    )

    sr_reprojection = _reprojection(sr, sequence, args)
    baseline_reprojection = _reprojection(baseline, sequence, args)
    sr_mse = sr_reprojection["aggregate"]["WeightedMSE"]
    baseline_mse = baseline_reprojection["aggregate"]["WeightedMSE"]
    quality = {
        "no_reference_sr": no_reference_metrics(sr),
        "no_reference_bicubic": no_reference_metrics(baseline),
        "reprojection_mse_reduction_percent": (
            100.0 * (baseline_mse - sr_mse) / baseline_mse
            if baseline_mse > 0
            else 0.0
        ),
    }
    if gt is not None:
        metrics_sr = evaluate(gt, sr)
        metrics_baseline = evaluate(gt, baseline)
        quality["with_ground_truth_sr"] = metrics_sr
        quality["with_ground_truth_bicubic"] = metrics_baseline
        quality["psnr_gain_db"] = metrics_sr["PSNR"] - metrics_baseline["PSNR"]
        quality["ssim_gain"] = metrics_sr["SSIM"] - metrics_baseline["SSIM"]
        make_error_map(gt, sr, output / "error_map.png")

    optimization = reconstruction_info.get("optimization", {})
    if optimization.get("losses"):
        make_convergence_plot(
            optimization["losses"],
            output / "convergence.png",
        )
    if args.save_debug:
        _save_debug_outputs(output, all_frames, sequence, sr, args)

    report = {
        "schema_version": 3,
        "provenance": provenance(args.command),
        "configuration_file": getattr(args, "_config_meta", {}),
        "resolved_configuration": namespace_config(args),
        "inputs": {
            "all_frames": input_manifest(all_paths),
            "selected_indices": sequence.selected_indices,
            "ground_truth": input_manifest([gt_path]) if gt_path is not None else None,
        },
        "registration": sequence.registration_report,
        "reconstruction": reconstruction_info,
        "quality": quality,
        "reprojection_sr": sr_reprojection,
        "reprojection_bicubic": baseline_reprojection,
        "outputs": {
            "sr_png": str((output / "sr_result.png").resolve()),
            "sr_float_npy": str((output / "sr_result.npy").resolve()),
            "bicubic_png": str((output / "bicubic_baseline.png").resolve()),
            "comparison": str((output / "comparison.png").resolve()),
        },
    }
    if extra_report:
        report.update(extra_report)
    write_json(output / report_name, report)
    return report


def cmd_make_demo(args: argparse.Namespace) -> None:
    output = ensure_dir(args.output)
    if args.preset == "phone-white":
        hr = make_phone_white_screen_led_hr(
            args.height,
            args.width,
            subpixel_width=args.subpixel_width,
            seed=args.seed,
        )
    else:
        hr = make_synthetic_hr(
            args.height,
            args.width,
            channels=args.channels,
            seed=args.seed,
        )
    save_image(output / "hr_gt.png", hr, dtype=args.output_dtype)
    if args.make_bayer_mask:
        mask = make_rggb_color_mask(args.height, args.width)
        save_image(output / "hr_rggb_mask.png", mask, dtype=args.output_dtype)
    print(f"Created demo target: {output / 'hr_gt.png'}")


def cmd_degrade(args: argparse.Namespace) -> None:
    _apply_legacy_noise_args(args)
    output = ensure_dir(args.output)
    hr_path = Path(args.hr)
    hr = read_image(hr_path)
    shifts = parse_shifts(args.shifts, args.scale)
    config = DegradationConfig(
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
    frames, shifts_hr = degrade_sequence(hr, config)
    paths = save_image_sequence(output, frames, prefix="lr", dtype=args.output_dtype)
    report = _build_degradation_report(
        args,
        hr,
        hr_path,
        shifts,
        shifts_hr,
        paths,
    )
    write_json(output / "degradation_meta.json", report)
    print(f"Saved {len(paths)} LR frames and metadata to: {output}")


def cmd_reconstruct(args: argparse.Namespace) -> None:
    output = ensure_dir(args.output)
    all_frames, all_paths = read_image_sequence_with_paths(
        args.input,
        pattern=args.input_pattern,
    )
    sequence = _resolve_sequence(all_frames, all_paths, args)
    gt_path = Path(args.gt) if args.gt else None
    gt = read_image(gt_path) if gt_path is not None else None
    report = _execute_reconstruction(
        args,
        output,
        all_frames,
        all_paths,
        sequence,
        gt,
        gt_path,
        "reconstruction_report.json",
    )
    print(f"Saved reconstruction to: {output / 'sr_result.png'}")
    print(
        "Reprojection MSE reduction vs bicubic: "
        f"{report['quality']['reprojection_mse_reduction_percent']:.2f}%"
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the comparison and ablation set used by technical disclosures."""
    output = ensure_dir(args.output)
    all_frames, all_paths = read_image_sequence_with_paths(
        args.input,
        pattern=args.input_pattern,
    )
    sequence = _resolve_sequence(all_frames, all_paths, args)
    gt_path = Path(args.gt) if args.gt else None
    gt = read_image(gt_path) if gt_path is not None else None
    base_config = _reconstruction_config(args, sequence)

    requested = _csv_strings(args.variants)
    variants = list(dict.fromkeys(requested))
    supported = {"bicubic", "fuse", "ibp_l2", "ibp_huber"}
    unknown = sorted(set(variants) - supported)
    if unknown:
        raise ValueError(f"Unknown benchmark variants: {', '.join(unknown)}")
    if not variants:
        raise ValueError("At least one benchmark variant is required.")

    variant_directory = ensure_dir(output / "variants")
    images: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    for name in variants:
        if name == "bicubic":
            config = replace(base_config, mode="bicubic")
        elif name == "fuse":
            config = replace(base_config, mode="fuse")
        elif name == "ibp_l2":
            config = replace(base_config, mode="ibp", robust_loss="l2")
        else:
            config = replace(base_config, mode="ibp", robust_loss="huber")

        started = perf_counter()
        image, reconstruction_info = reconstruct(sequence.frames, config)
        elapsed = perf_counter() - started
        images[name] = image
        save_image(variant_directory / f"{name}.png", image, dtype="uint16")
        save_image(variant_directory / f"{name}.npy", image)

        result = {
            "elapsed_seconds": elapsed,
            "reconstruction": reconstruction_info,
            "no_reference": no_reference_metrics(image),
            "reprojection": _reprojection(image, sequence, args),
        }
        if gt is not None:
            result["with_ground_truth"] = evaluate(gt, image)
        results[name] = result

        optimization = reconstruction_info.get("optimization", {})
        if optimization.get("losses"):
            make_convergence_plot(
                optimization["losses"],
                variant_directory / f"{name}_convergence.png",
            )

    if gt is not None:
        ranking = sorted(
            variants,
            key=lambda name: results[name]["with_ground_truth"]["PSNR"],
            reverse=True,
        )
        ranking_metric = "PSNR"
    else:
        ranking = sorted(
            variants,
            key=lambda name: results[name]["reprojection"]["aggregate"]["WeightedMSE"],
        )
        ranking_metric = "WeightedReprojectionMSE"

    if "bicubic" in results:
        baseline_mse = results["bicubic"]["reprojection"]["aggregate"]["WeightedMSE"]
        for name in variants:
            mse = results[name]["reprojection"]["aggregate"]["WeightedMSE"]
            results[name]["reprojection_mse_reduction_vs_bicubic_percent"] = (
                100.0 * (baseline_mse - mse) / baseline_mse
                if baseline_mse > 0
                else 0.0
            )

    comparison_path = output / "benchmark_comparison.png"
    make_benchmark_comparison(images, comparison_path, gt=gt)
    report = {
        "schema_version": 1,
        "provenance": provenance(args.command),
        "configuration_file": getattr(args, "_config_meta", {}),
        "resolved_configuration": namespace_config(args),
        "inputs": {
            "frames": input_manifest(all_paths),
            "ground_truth": input_manifest([gt_path]) if gt_path is not None else None,
        },
        "registration": sequence.registration_report,
        "ranking_metric": ranking_metric,
        "ranking": ranking,
        "best_variant": ranking[0],
        "variants": results,
        "outputs": {
            "variant_directory": str(variant_directory.resolve()),
            "comparison": str(comparison_path.resolve()),
        },
    }
    write_json(output / "benchmark_report.json", report)
    print(f"Benchmark completed. Best variant by {ranking_metric}: {ranking[0]}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    _apply_legacy_noise_args(args)
    output = ensure_dir(args.output)
    hr_path = Path(args.hr) if args.hr else output / "hr_gt.png"
    if args.hr:
        hr = read_image(hr_path)
    elif args.preset == "phone-white":
        hr = make_phone_white_screen_led_hr(
            args.height,
            args.width,
            subpixel_width=args.subpixel_width,
            seed=args.seed,
        )
        save_image(hr_path, hr, dtype=args.output_dtype)
    else:
        hr = make_synthetic_hr(
            args.height,
            args.width,
            channels=args.channels,
            seed=args.seed,
        )
        save_image(hr_path, hr, dtype=args.output_dtype)

    shifts = parse_shifts(args.shifts, args.scale)
    degradation_config = DegradationConfig(
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
    generated_frames, shifts_hr = degrade_sequence(hr, degradation_config)
    lr_directory = ensure_dir(output / "lr_frames")
    all_paths = save_image_sequence(
        lr_directory,
        generated_frames,
        prefix="lr",
        dtype=args.output_dtype,
    )
    all_frames = [read_image(path) for path in all_paths]
    degradation_report = _build_degradation_report(
        args,
        hr,
        hr_path,
        shifts,
        shifts_hr,
        all_paths,
    )
    write_json(lr_directory / "degradation_meta.json", degradation_report)

    sequence = _resolve_sequence(all_frames, all_paths, args)
    report = _execute_reconstruction(
        args,
        output,
        all_frames,
        all_paths,
        sequence,
        hr,
        hr_path,
        "pipeline_report.json",
        extra_report={"degradation": degradation_report["image_formation_model"]},
    )
    print(f"Pipeline completed: {output}")
    print(
        f"PSNR gain vs bicubic: {report['quality'].get('psnr_gain_db', 0.0):.3f} dB"
    )


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def cmd_calibrate(args: argparse.Namespace) -> None:
    output = ensure_dir(args.output)
    hr_path = Path(args.hr)
    hr = read_image(hr_path)
    all_frames, all_paths = read_image_sequence_with_paths(
        args.input,
        pattern=args.input_pattern,
    )
    sequence = _resolve_sequence(all_frames, all_paths, args)
    shifts_hr = convert_shifts_to_hr(
        sequence.shifts,
        args.scale,
        sequence.shift_unit,
    )
    calibration, predictions = calibrate_forward_model(
        hr,
        sequence.frames,
        shifts_hr,
        scale=args.scale,
        blur_types=_csv_strings(args.blur_types),
        blur_kernels=_csv_ints(args.blur_kernels),
        blur_sigmas=_csv_floats(args.blur_sigmas),
        blur_stage=args.blur_stage,
        border_mode=args.border,
        interpolation=args.downsample,
    )
    prediction_dir = ensure_dir(output / "best_model_predictions")
    prediction_paths = save_image_sequence(
        prediction_dir,
        predictions,
        prefix="predicted_lr",
        dtype="uint16",
    )
    report = {
        "schema_version": 1,
        "provenance": provenance(args.command),
        "configuration_file": getattr(args, "_config_meta", {}),
        "resolved_configuration": namespace_config(args),
        "inputs": {
            "hr": input_manifest([hr_path]),
            "frames": input_manifest(all_paths),
        },
        "registration": sequence.registration_report,
        "calibration": calibration,
        "outputs": {"best_model_predictions": input_manifest(prediction_paths)},
    }
    write_json(output / "calibration_report.json", report)
    best = calibration["best_model"]
    print(
        "Best calibrated model: "
        f"{best['blur_type']} kernel={best['blur_kernel']} "
        f"sigma={best['blur_sigma']:g}, MSE={best['mean_mse']:.8f}"
    )


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML configuration. Explicit CLI flags take precedence.",
    )


def _add_common_shift_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--shifts",
        type=str,
        default="0,0;0.5,0;0,0.5;0.5,0.5",
        help="Forward shifts as dx,dy;dx,dy.",
    )
    parser.add_argument("--shift-unit", choices=["lr", "hr"], default="lr")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--blur-type",
        choices=["none", "defocus", "gaussian"],
        default="defocus",
    )
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    parser.add_argument(
        "--blur-stage",
        choices=["before_downsample", "after_downsample", "none"],
        default="before_downsample",
    )
    parser.add_argument(
        "--downsample",
        choices=["area", "linear", "cubic", "nearest", "lanczos"],
        default="area",
    )
    parser.add_argument(
        "--border",
        choices=["replicate", "reflect", "constant"],
        default="replicate",
    )


def _add_noise_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--noise-poisson-a", type=float, default=0.001)
    parser.add_argument("--noise-gaussian-b", type=float, default=0.0001)
    parser.add_argument("--noise-std", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--noise-in-255", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dtype",
        choices=["uint8", "uint16"],
        default="uint16",
    )


def _add_registration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registration",
        choices=["known", "estimate"],
        default="known",
    )
    parser.add_argument("--registration-upsample", type=int, default=100)
    parser.add_argument("--registration-highpass", type=float, default=2.0)
    parser.add_argument("--registration-ecc-iterations", type=int, default=100)
    parser.add_argument("--min-registration-confidence", type=float, default=0.20)
    parser.add_argument("--max-shift-lr", type=float, default=2.0)
    parser.add_argument("--min-accepted-frames", type=int, default=2)
    parser.add_argument("--keep-rejected", action="store_true")


def _add_reconstruction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=["bicubic", "interlace", "fuse", "ibp", "inverse"],
        default="ibp",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--step-size", type=float, default=0.20)
    parser.add_argument("--robust-loss", choices=["huber", "l2"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-iterations", type=int, default=10)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--laplacian-weight", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--save-debug", action="store_true")


def _add_input_sequence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input-pattern", type=str, default="*.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physics-calibrated pixel-shift super-resolution toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("make-demo", help="Create a synthetic HR target.")
    command.add_argument("--output", type=str, default="runs/demo")
    command.add_argument(
        "--preset",
        choices=["default", "phone-white"],
        default="phone-white",
    )
    command.add_argument("--height", type=int, default=512)
    command.add_argument("--width", type=int, default=512)
    command.add_argument("--subpixel-width", type=int, default=1)
    command.add_argument("--channels", type=int, choices=[1, 3], default=3)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--output-dtype", choices=["uint8", "uint16"], default="uint16")
    command.add_argument("--make-bayer-mask", action="store_true")
    command.set_defaults(func=cmd_make_demo)

    command = subparsers.add_parser("degrade", help="Generate calibrated LR frames.")
    command.add_argument("--hr", type=str, required=True)
    command.add_argument("--output", type=str, required=True)
    _add_config_arg(command)
    _add_common_shift_args(command)
    _add_model_args(command)
    _add_noise_args(command)
    command.set_defaults(func=cmd_degrade)

    command = subparsers.add_parser(
        "reconstruct",
        help="Register and reconstruct an LR sequence.",
    )
    _add_input_sequence_args(command)
    command.add_argument("--output", type=str, required=True)
    command.add_argument("--gt", type=str, default=None)
    _add_config_arg(command)
    _add_common_shift_args(command)
    _add_model_args(command)
    _add_registration_args(command)
    _add_reconstruction_args(command)
    command.set_defaults(func=cmd_reconstruct)

    command = subparsers.add_parser(
        "pipeline",
        help="Run synthetic degradation, reconstruction, and evaluation.",
    )
    command.add_argument("--hr", type=str, default=None)
    command.add_argument("--output", type=str, default="runs/pipeline")
    command.add_argument(
        "--preset",
        choices=["default", "phone-white"],
        default="phone-white",
    )
    command.add_argument("--height", type=int, default=512)
    command.add_argument("--width", type=int, default=512)
    command.add_argument("--subpixel-width", type=int, default=1)
    command.add_argument("--channels", type=int, choices=[1, 3], default=3)
    _add_config_arg(command)
    _add_common_shift_args(command)
    _add_model_args(command)
    _add_noise_args(command)
    _add_registration_args(command)
    _add_reconstruction_args(command)
    command.set_defaults(func=cmd_pipeline)

    command = subparsers.add_parser(
        "benchmark",
        help="Run bicubic/fusion/L2/Huber ablation comparisons.",
    )
    _add_input_sequence_args(command)
    command.add_argument("--output", type=str, required=True)
    command.add_argument("--gt", type=str, default=None)
    command.add_argument(
        "--variants",
        type=str,
        default="bicubic,fuse,ibp_l2,ibp_huber",
    )
    _add_config_arg(command)
    _add_common_shift_args(command)
    _add_model_args(command)
    _add_registration_args(command)
    _add_reconstruction_args(command)
    command.set_defaults(func=cmd_benchmark)

    command = subparsers.add_parser(
        "calibrate",
        help="Fit optical blur and sensor noise from a known HR target.",
    )
    command.add_argument("--hr", type=str, required=True)
    _add_input_sequence_args(command)
    command.add_argument("--output", type=str, required=True)
    _add_config_arg(command)
    _add_common_shift_args(command)
    _add_registration_args(command)
    command.add_argument("--blur-types", type=str, default="defocus,gaussian")
    command.add_argument("--blur-kernels", type=str, default="1,3,5,7")
    command.add_argument("--blur-sigmas", type=str, default="0.6,1.0,1.4")
    command.add_argument(
        "--blur-stage",
        choices=["before_downsample", "after_downsample", "none"],
        default="before_downsample",
    )
    command.add_argument(
        "--downsample",
        choices=["area", "linear", "cubic", "nearest", "lanczos"],
        default="area",
    )
    command.add_argument(
        "--border",
        choices=["replicate", "reflect", "constant"],
        default="replicate",
    )
    command.set_defaults(func=cmd_calibrate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args._config_meta = apply_yaml_config(args, sys.argv[1:])
        validate_namespace(parser, args)
        args.func(args)
    except (FileNotFoundError, ValueError, IOError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
