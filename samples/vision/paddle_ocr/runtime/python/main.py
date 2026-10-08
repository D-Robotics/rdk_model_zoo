# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native entrypoint for the bounded Python PaddleOCR composition pilot.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run/prepare modes, resolve the pair, construct the
two-stage pipeline, call ``predict``, show the result.  Option
declarations, the model-free modes and result rendering live in
``cli.py``; the detection → crop → recognition composition itself lives
in ``pipeline.py``.  The entrypoint is the only module that adjusts
``sys.path`` for direct full-checkout invocation; SDK, OpenCV and
pyclipper imports stay behind the selected operation so help/list/dry-run
remain host-safe.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence


_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    # ``python samples/vision/paddle_ocr/runtime/python/main.py`` is supported
    # from any current directory inside a full source checkout.
    sys.path.insert(0, str(_ROOT))

from samples.vision.paddle_ocr.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    default_image,
    print_result,
    read_bgr_image,
    resolve_pair_from_args,
    run_dry_run,
    run_list_models,
    run_prepare,
    write_json_output,
)
from samples.vision.paddle_ocr.runtime.python.model_binding import (  # noqa: E402
    BindingError,
    SUPPORTED_TARGETS,  # noqa: F401 - re-exported for existing callers
    list_available_pairs,  # noqa: F401 - re-exported for existing callers
    resolve_pair,  # noqa: F401 - re-exported for existing callers
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute list, dry-run, prepare or inference and return a process code."""

    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target, args.output_format)
        if args.dry_run:
            return run_dry_run(args)
        if args.prepare:
            return run_prepare(args)

        pair = resolve_pair_from_args(args, for_execution=True)
        detector_path = pair.detector_model_path
        recognizer_path = pair.recognizer_model_path
        if not detector_path.is_file():
            raise FileNotFoundError(f"detector model file not found: {detector_path}")
        if not recognizer_path.is_file():
            raise FileNotFoundError(f"recognizer model file not found: {recognizer_path}")

        # A real execution must prove the exact detected board before importing
        # hbm_runtime.  The stage runners repeat this check immediately before load.
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(pair.target)

        # Imported inside real execution: OpenCV and hbm_runtime load only
        # after the pair, file, and board checks above have passed.
        from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

        image = read_bgr_image(
            args.test_img if args.test_img else default_image(pair.target)
        )
        result = OCRPipeline.from_models(
            pair,
            vocabulary_path=args.vocabulary_path,
            priority=args.priority,
            bpu_cores=args.bpu_cores,
        ).predict(image)
        payload = result.as_dict()
        payload["image_shape"] = list(image.shape)
        print_result(payload, args.output_format)
        if args.json_output:
            write_json_output(args.json_output, payload)
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
