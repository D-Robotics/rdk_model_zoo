"""Compatibility command for the historical X5 ResNet sample.

The canonical entrypoint owns argument parsing and execution.  This shim only
supplies the X5 sample's published defaults and qualified manifest reference;
all original flags (including ``--topk``) remain accepted by the canonical
parser.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "samples/vision/resnet/runtime/python/main.py").is_file()
            and (candidate / "samples/vision/resnet/conversion/export_resnet18_onnx.py").is_file()
        ):
            return candidate
    raise RuntimeError("Could not locate the canonical ResNet18 entrypoint.")


_ROOT = _repository_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
_OLD_SAMPLE = _SCRIPT_DIR.parents[1]
_DEFAULT_MODEL = _OLD_SAMPLE / "model" / "resnet18_224x224_nv12.bin"
_DEFAULT_IMAGE = _OLD_SAMPLE / "test_data" / "white_wolf.JPEG"
_DEFAULT_RESULT = _OLD_SAMPLE / "test_data" / "result.jpg"
_DEFAULT_LABELS = _SCRIPT_DIR.parents[4] / "datasets" / "imagenet" / "imagenet_classes.names"


def _has_option(arguments: list[str], *names: str) -> bool:
    return any(
        value == name or value.startswith(name + "=")
        for value in arguments
        for name in names
    )


def main(argv: list[str] | None = None) -> int:
    """Run the canonical entrypoint with historical X5 defaults."""

    from samples.vision.resnet.runtime.python.main import main as canonical_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(arguments, "--target"):
        arguments.extend(["--target", "x5"])
    if not _has_option(arguments, "--asset-id"):
        arguments.extend(["--asset-id", "x5:resnet:resnet18_224x224_nv12.bin"])
    if not _has_option(arguments, "--model-path"):
        arguments.extend(["--model-path", str(_DEFAULT_MODEL)])
    if not _has_option(arguments, "--test-img"):
        arguments.extend(["--test-img", str(_DEFAULT_IMAGE)])
    if not _has_option(arguments, "--label-file"):
        arguments.extend(["--label-file", str(_DEFAULT_LABELS)])
    if not _has_option(arguments, "--img-save-path"):
        arguments.extend(["--img-save-path", str(_DEFAULT_RESULT)])
    return canonical_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
