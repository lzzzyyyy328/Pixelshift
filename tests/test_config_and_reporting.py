import json

from pixelshiftsr.cli import build_parser
from pixelshiftsr.config import apply_yaml_config, validate_namespace
from pixelshiftsr.reporting import write_json


def test_yaml_config_applies_and_cli_wins(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
scale: 2
model:
  blur_type: gaussian
  blur_kernel: 5
degradation:
  noise_poisson_a: 0.002
registration:
  mode: estimate
reconstruction:
  iterations: 90
  robust_loss: l2
""".strip(),
        encoding="utf-8",
    )
    argv = [
        "pipeline",
        "--config",
        str(config),
        "--iterations",
        "3",
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata = apply_yaml_config(args, argv)
    validate_namespace(parser, args)

    assert args.blur_type == "gaussian"
    assert args.blur_kernel == 5
    assert args.registration == "estimate"
    assert args.robust_loss == "l2"
    assert args.iterations == 3
    assert metadata["cli_overrides"] == ["iterations"]


def test_json_reports_are_strict(tmp_path):
    path = write_json(tmp_path / "report.json", {"finite": 1.0, "infinite": float("inf")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"finite": 1.0, "infinite": None}
