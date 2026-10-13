# Fixture provenance

`x5-board-manifest.json` is a verbatim manifest captured from an RDK X5 (8 GB)
board run of the unified `yolov5_cpp --dump-dir` output. It is checked in so the
comparison-tool schema regression in `tests/test_native_evaluator.py` runs in
every checkout instead of skipping when no sibling workspace is present. Do not
edit the copy; supersede it only with a new verbatim board manifest.
