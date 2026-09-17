"""Compatibility command for the historical S-series ResNet18 sample.

The canonical entrypoint owns argument parsing and execution.  This shim
supplies the old sample's defaults and maps its documented S100/S600 model
directories to exact manifest references.  An arbitrary custom model path
uses the shared board identity when no target is supplied; an explicit
canonical ``--asset-id`` remains available when selecting a particular
manifest artifact. Chip and tensor protocol are never inferred from a
filename.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "samples/vision/resnet/runtime/python/main.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the canonical ResNet18 entrypoint.")


_ROOT = _repository_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
_OLD_SAMPLE = _SCRIPT_DIR.parents[1]
_PLATFORM_ROOT = _SCRIPT_DIR.parents[4]
_DEFAULT_MODEL = _OLD_SAMPLE / "model" / "s100" / "resnet18_224x224_nv12.hbm"
_DEFAULT_IMAGE = _OLD_SAMPLE / "test_data" / "zebra_cls.jpg"
_DEFAULT_LABELS = _PLATFORM_ROOT / "datasets" / "imagenet" / "imagenet_classes.names"


def _has_option(arguments: list[str], *names: str) -> bool:
    return any(
        value == name or value.startswith(name + "=")
        for value in arguments
        for name in names
    )


def _option_value(arguments: list[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _known_target(model_path: str) -> str | None:
    parts = {part.lower() for part in Path(model_path).expanduser().parts}
    if "s600" in parts:
        return "s600"
    if "s100" in parts:
        return "s100"
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the canonical entrypoint with historical S-series defaults."""

    from samples.vision.resnet.runtime.python.main import main as canonical_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    model_path = _option_value(arguments, "--model-path")
    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
        arguments.extend(["--model-path", model_path])

    explicit_target = _option_value(arguments, "--target")
    target = explicit_target or _known_target(model_path)
    inspection = any(
        value in ("--help", "-h", "--list-models", "--dry-run")
        for value in arguments
    )
    if target in (None, "auto") and not inspection:
        # The old S18 command had no target flag. Resolve that omission from
        # the shared board identity for custom paths; never use a filename as
        # chip evidence. Leave the canonical error visible if identity is
        # unavailable on a host.
        try:
            from samples._shared.platforms import resolve_target

            detected = resolve_target("auto")
        except ValueError:
            detected = None
        if detected in ("s100", "s600"):
            target = detected
    if not _has_option(arguments, "--target") and target is not None:
        arguments.extend(["--target", target])

    if not _has_option(arguments, "--asset-id") and target in ("s100", "s600"):
        arguments.extend([
            "--asset-id",
            f"s:resnet18:{target}/resnet18_224x224_nv12.hbm",
        ])
    if not _has_option(arguments, "--test-img"):
        arguments.extend(["--test-img", str(_DEFAULT_IMAGE)])
    if not _has_option(arguments, "--label-file"):
        arguments.extend(["--label-file", str(_DEFAULT_LABELS)])
    return canonical_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
