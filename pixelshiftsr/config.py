from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def load_yaml_config(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("The configuration root must be a mapping.")
    return payload


def _explicit_options(argv: Sequence[str]) -> set[str]:
    explicit: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            name = token[2:].split("=", 1)[0].replace("-", "_")
            explicit.add(name)
    return explicit


def _shift_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("shifts must be a string or a list of [dx, dy] pairs.")
    pairs: list[str] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each configured shift must contain dx and dy.")
        pairs.append(f"{float(item[0]):g},{float(item[1]):g}")
    return ";".join(pairs)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a mapping.")
    return value


def apply_yaml_config(
    args: argparse.Namespace,
    argv: Sequence[str],
) -> dict:
    """Apply YAML values while preserving explicitly supplied CLI options."""
    config_path = getattr(args, "config", None)
    if not config_path:
        return {}
    raw = load_yaml_config(config_path)
    allowed_root_keys = {
        "schema_version",
        "scale",
        "shifts",
        "shift_unit",
        "model",
        "degradation",
        "registration",
        "reconstruction",
        "calibration",
    }
    unknown_root_keys = sorted(set(raw) - allowed_root_keys)
    if unknown_root_keys:
        raise ValueError(
            "Unknown top-level configuration keys: "
            + ", ".join(unknown_root_keys)
        )
    command = str(args.command)
    explicit = _explicit_options(argv)
    values: dict[str, Any] = {}

    for key in ("scale", "shifts", "shift_unit"):
        if key in raw:
            values[key] = raw[key]

    sections: list[tuple[str, Mapping[str, Any]]] = []
    if command in {"degrade", "reconstruct", "pipeline", "benchmark"}:
        sections.append(("model", _mapping(raw.get("model"), "model")))
    if command in {"degrade", "pipeline"}:
        sections.append(
            ("degradation", _mapping(raw.get("degradation"), "degradation"))
        )
    if command in {"reconstruct", "pipeline", "calibrate", "benchmark"}:
        registration = dict(_mapping(raw.get("registration"), "registration"))
        registration_aliases = {
            "mode": "registration",
            "upsample_factor": "registration_upsample",
            "highpass_sigma": "registration_highpass",
            "min_confidence": "min_registration_confidence",
            "ecc_iterations": "registration_ecc_iterations",
        }
        for source, destination in registration_aliases.items():
            if source in registration:
                registration[destination] = registration.pop(source)
        sections.append(("registration", registration))
    if command in {"reconstruct", "pipeline", "benchmark"}:
        sections.append(
            (
                "reconstruction",
                _mapping(raw.get("reconstruction"), "reconstruction"),
            )
        )
    if command == "calibrate":
        sections.append(
            ("calibration", _mapping(raw.get("calibration"), "calibration"))
        )

    for section_name, section in sections:
        for key, value in section.items():
            if key in values:
                raise ValueError(
                    f"Duplicate active configuration key {key!r} in section {section_name!r}."
                )
            values[key] = value

    if "shifts" in values:
        values["shifts"] = _shift_text(values["shifts"])

    applied: dict[str, Any] = {}
    for key, value in values.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown configuration key {key!r} for command {command!r}."
            )
        if key not in explicit:
            setattr(args, key, value)
            applied[key] = value
    return {
        "path": str(Path(config_path).resolve()),
        "applied": applied,
        "cli_overrides": sorted(explicit.intersection(values)),
    }


def validate_namespace(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Validate values applied after argparse, including YAML choices and types."""
    actions = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            selected = action.choices.get(str(getattr(args, action.dest, "")))
            if selected is not None:
                actions.extend(selected._actions)

    for action in actions:
        destination = action.dest
        if destination in {argparse.SUPPRESS, "help"} or not hasattr(args, destination):
            continue
        value = getattr(args, destination)
        if value is None:
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f"{destination} must be true or false.")
            continue
        if action.type is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid configured value for {destination}: {value!r}."
                ) from exc
            setattr(args, destination, value)
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(str(item) for item in action.choices)
            raise ValueError(
                f"Invalid configured value for {destination}: {value!r}. "
                f"Choose from: {choices}."
            )
