# Python runtime

`main.py` is the native entrypoint for the bounded X5 PP-OCRv3 and S100
PP-OCRv6 pilot. It resolves a qualified detector/recognizer pair, checks the
execution target, loads each model lazily through `model_runner.py`, and emits
ordered boxes and texts.

## Command and parameter index

```bash
python samples/vision/paddle_ocr/runtime/python/main.py --help
python samples/vision/paddle_ocr/runtime/python/main.py --list-models --target auto
python samples/vision/paddle_ocr/runtime/python/main.py --dry-run --target x5
bash samples/vision/paddle_ocr/runtime/python/run.sh --dry-run --target s100
```

| Parameter | Meaning |
| --- | --- |
| `--target` | `auto`, `x5`, `s100`, `s100p`, or `s600`; real execution requires an exact detected target |
| `--det-asset-id` / `--rec-asset-id` | qualified references printed by `--list-models` |
| `--det-model-path` / `--rec-model-path` | existing local artifacts; both are required together and are never downloaded implicitly |
| `--vocabulary-path` | optional S100 UTF-8 dictionary path; its audited digest is checked |
| `--test-img` | BGR image path; defaults to the selected bundled fixture |
| `--output-format` | `text` or `json` |
| `--json-output` | optional local JSON copy of an inference result |
| `--priority` / `--bpu-cores` | lazy runtime scheduling, default `0` and `[0]` |
| `--list-models` | list manifest-backed pairs without SDK, OpenCV or pyclipper |
| `--dry-run` | resolve and print a static contract without model loading |
| `--prepare` / `--model-dir` | explicit model preparation; this is the only URL-capable operation |

The JSON result contains `target`, `image_shape`, `detector_asset`,
`recognizer_asset`, `boxes`, and aligned `texts`. Model output semantics remain
the observed score-map/CTC policies; no unverified activation is inserted.
`S100P` and `S600` have no audited PaddleOCR pair and are rejected.

The package modules use absolute full-checkout imports and do not alter
`sys.path`; only the entrypoint supports direct script invocation from an
arbitrary current directory. `hbm_runtime` and `pyclipper` are imported only
when their execution stage needs them. See the [parent pilot README](../../README.md)
for exact contracts, source mapping, dependencies and acceptance status, and
the [existing X5 runtime](../../../../../platforms/x5/samples/vision/paddleocr/runtime/python/README.md)
and [existing S100 runtime](../../../../../platforms/s/samples/vision/paddle_ocr/runtime/python/README.md)
for legacy commands.
