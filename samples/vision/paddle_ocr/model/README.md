# PaddleOCR model preparation

<a id="artifacts"></a>
## Artifacts

This directory holds no model binaries. The sample consumes four published
manifest rows — two per pair — and never treats a detector or recognizer as
interchangeable across pairs:

| Target | Role | Qualified reference | Format |
| --- | --- | --- | --- |
| X5 | detector | `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` | `.bin`, march `bayes-e`, packed NV12 640×640 |
| X5 | recognizer | `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` | `.bin`, march `bayes-e`, RGB featuremap 48×320 |
| S100 | detector | `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `.hbm`, march `nash-e`, split NV12 640×640 |
| S100 | recognizer | `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | `.hbm`, march `nash-e`, RGB 48×320 |

The rows live in the platform release manifests
(`platforms/x5/docs/release/models.yaml` and
`platforms/s/docs/release/models.yaml` during the migration window). There
is no audited S100P or S600 PaddleOCR row, so no artifact is claimed for
those targets.

<a id="preparation"></a>
## Preparation

Preparation is explicit and centralized in the Python entrypoint's
`--prepare` mode — the only network-capable operation in the sample. It
reads the manifest URLs, fetches both files of one pair, and prints the
observed SHA-256 digests (cwd: repository root; success: exit 0, two files
under `--model-dir`):

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

For S100 substitute the two `s:paddle_ocr:s100/...` references and a
`--model-dir` such as `/opt/hobot/model/s100/basic`. Normal inference never
downloads: it requires both local paths to exist already (see
[runtime parameters](../runtime/python/README.md#parameters)). A manual copy
of previously downloaded artifacts is equally valid preparation; verify the
digest against the value printed by `--prepare` on first use.

<a id="accompanying-files"></a>
## Accompanying files

Each recognizer consumes a character dictionary as part of its pair
contract:

| Target | Dictionary | Location | Notes |
| --- | --- | --- | --- |
| X5 | fixed 96-character alphabet | embedded in `runtime/python/model_binding.py` (`X5_ALPHABET`) | not a file; blank class prepended at decode time |
| S100 | PP-OCRv6 UTF-8 dictionary | [`test_data/s100/ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt) | 18,708 lines; blank prepended and one trailing space appended at load → 18,710 classes |

`--vocabulary-path` may substitute the S100 dictionary only when the file's
SHA-256 matches the audited digest below; any other content is rejected.
The S-series C++ runtime additionally uses a TrueType font for result
rendering; it is carried from the audited source delivery at
`platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf` and is
selectable with the C++ `--font_path` flag.

<a id="local-paths"></a>
## Local paths

- X5 pair: wherever `--prepare --model-dir` wrote it, e.g.
  `/tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin`; pass both paths with
  `--det-model-path`/`--rec-model-path` together with both qualified
  references.
- Default lookup: when both path flags are omitted, the pair resolves to
  this directory — `samples/vision/paddle_ocr/model/<filename>` — so
  copying the artifacts here makes the explicit path flags unnecessary
  (custom paths must always be supplied as a detector+recognizer pair).
- S100 pair: RDK S images ship the pair under
  `/opt/hobot/model/s100/basic/`; a full checkout may also `--prepare`
  into that directory when the images are absent.
- The dictionary and fixtures are checked in; no download applies to them.

<a id="formats-checksums"></a>
## Formats and checksums

| File | SHA-256 | Status |
| --- | --- | --- |
| `en_PP-OCRv3_det_640x640_nv12.bin` | null | no publisher digest recorded for the manifest row |
| `en_PP-OCRv3_rec_48x320_rgb.bin` | null | no publisher digest recorded |
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | null | no publisher digest recorded |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | null | no publisher digest recorded |
| `test_data/s100/ppocrv6_dict.txt` | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` | audited, enforced for `--vocabulary-path` replacements |

Unknown digests stay `null` rather than being copied or guessed; record the
observed digest printed by `--prepare` in your own evidence when a board run
needs provenance. Runtime tensor contracts (names, shapes, dtypes) for each
artifact are enforced at load time and documented in
[runtime stage I/O](../runtime/python/README.md#stage-io).
