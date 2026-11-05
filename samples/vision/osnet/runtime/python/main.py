"""OSNet entrypoint: arguments, model construction, predict and output."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.osnet.model.artifacts import load_manifest, require_artifact
from samples.vision.osnet.runtime.python.cli import (
    build_parser, model_path, print_selection, read_image, save_feature,
)


def main(argv=None):
    """Run the native sample, returning 0 on success or 2 on a reported error."""
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(json.dumps(load_manifest()["artifacts"], indent=2))
        return 0
    if args.dry_run:
        print_selection(args)
        return 0
    try:
        path = model_path(args)
        record = require_artifact(path, args.variant)
        expected_target = record.get("target", "x5")
        if args.target not in ("auto", expected_target):
            raise ValueError(f"Artifact target {expected_target} differs from --target {args.target}")
        from samples.vision.osnet.runtime.python.matching import OSNet, OSNetConfig

        model = OSNet(OSNetConfig(path, target=expected_target, normalize=not args.raw))
        image = read_image(args.test_img)
        feature = model.predict(image)
        save_feature(feature, args.output)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
