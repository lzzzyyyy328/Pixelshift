from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import skimage

from . import __version__


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    text = json.dumps(
        json_safe(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def provenance(command: str) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "pixelshiftsr_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
        },
        "argv": sys.argv,
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(paths: Sequence[str | Path]) -> list[dict]:
    manifest: list[dict] = []
    for item in paths:
        path = Path(item).resolve()
        manifest.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def namespace_config(namespace: Any) -> dict:
    return {
        key: value
        for key, value in vars(namespace).items()
        if key != "func" and not key.startswith("_") and not callable(value)
    }
