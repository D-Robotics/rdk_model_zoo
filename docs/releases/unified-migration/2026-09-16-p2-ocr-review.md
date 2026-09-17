# P2 OCR independent review

Date: 2026-09-16. Fixed original-source baseline:
`cd74a2b241075bb21036d8d0855d0403f8e8c963`.

Scope: the bounded Python OCR pilot in `samples/vision/paddle_ocr`, its tests,
native entrypoint, copied fixtures and bilingual documentation. Requirements:
[P2 plan, Task 3](../../superpowers/plans/2026-09-16-p2-protocols.md) and
[OCR source audit](p2-ocr-source-audit.md). This review did not change implementation
or tests, access a board or network, download an artifact, or make a commit.
The old X5/S Python, S C++, conversion and evaluation paths are compatibility
references; this review does not declare them migrated. Segmentation/pose and
later repository-wide migration work are outside the reduced representative scope.

## Review status

Independent review is complete. All five reported code issues and the delivery
corrections below are fixed and independently rechecked. There are no remaining
actionable P1/P2 findings within this bounded scope. The final full OCR host
suite passed 31 tests. The coordinator subsequently supplied three successful
board-comparison logs; those are separately attributed below. Host tests alone
do not establish board acceptance, accuracy or performance.

## Contract and source facts

The binding enumerates exactly two pairs from the shared manifest reader,
using four existing qualified references rather than a second URL registry:

| Target/stage | Qualified reference | Observed model name | Input metadata / physical buffer | Output |
| --- | --- | --- | --- | --- |
| X5 detector | `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` | `en_PP-OCRv3_det_infer-deploy_640x640_nv12` | `x` NV12 `[1,3,640,640]` / U8 `[1,960,640,1]` | `sigmoid_0.tmp_0`, F32 `[1,1,640,640]` |
| X5 recognizer | `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` | `en_PP-OCRv3_rec_infer-deploy_48x320_rgb_NCHW` | `x` F32 `[1,3,48,320]` | `softmax_2.tmp_0`, F32 `[1,40,97,1]` |
| S100 detector | `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `PP-OCRv6_det_infer-deploy_640x640_nv12` | `x_y` U8 `[1,640,640,1]`; `x_uv` U8 `[1,320,320,2]` | `fetch_name_0`, F32 `[1,1,640,640]` |
| S100 recognizer | `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | `PP-OCRv6_rec_infer-deploy_48x320_rgb` | `x` F32 `[1,3,48,320]` | `fetch_name_0`, F32 `[1,40,18710]` |

Six saved metadata records from [X5 8G](evidence/x5-8g-p2-ocr-metadata.log),
[X5 4G](evidence/x5-4g-p2-ocr-metadata.log) and
[S100](evidence/s100-p2-ocr-metadata.log) were parsed on the host and passed
`RuntimeMetadata.from_runtime` followed by `bind_stage`. This was a check of
existing records, not a new board capture.

The X5 alphabet exactly matches the original 96-character literal, including
two separate trailing space classes. The S100 table exactly matches the
original UTF-8 file's token order: 18,708 source lines, a blank at index zero
and one appended space, giving 18,710 classes. The source file's raw-byte SHA-256
is `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6`;
the digest is not a hash of a token table after adding blank/space.

X5 LINEAR packed NV12 and S100 AREA split NV12 remain distinct. Recognition
keeps the original resize/divide/cast/channel-order sequence. Score thresholds
and CTC argmax retain source behavior without an added activation. Actual
runner buffers are checked for exact names, shape, dtype and finite values.
Unsupported S100P/S600 pairs and mixed asset pairs are rejected. Production
execution checks the detected target before SDK construction; injected callables
retain the same physical tensor validation without requiring a board SDK.

## Findings and repair evidence

### OCR-1 — P1: bound inputs raised `AttributeError` (fixed)

Found by this reviewer. `validate_stage_inputs` read `runtime_input_shapes`
and `runtime_input_dtypes` from a `StageBinding`, but the original candidate
only stored those fields in its nested `StageContract`. Binding valid metadata,
preparing an image and passing the resulting tensors to the validator raised
`AttributeError` for all four target/stage combinations.

`StageBinding` now exposes both contract-backed properties. The same path and
the added `test_bound_stage_validates_physical_runtime_tensors` pass. See
[model_binding.py](../../../samples/vision/paddle_ocr/runtime/python/model_binding.py),
`StageBinding` properties at lines 258–267 and `validate_stage_inputs` at line 444.

### OCR-2 — P2: S100 degenerate crop behavior changed (fixed)

Found and first reproduced by the coordinating agent; independently rechecked
here. The candidate added a `ValueError` for zero-width/height rectangles.
The S source instead lets OpenCV produce its result, while X5 returns the
original image. With `np.arange(9*13*3, dtype=np.uint8).reshape(9,13,3)`,
an all-zero box yields S shape `(9,13,3)`, sum 351; the collinear box
`[[0,0],[3,0],[3,0],[0,0]]` yields S shape `(13,9,3)`, sum 351.

Both cases now match the original S function pixel for pixel. X5 still returns
the original image for both boxes. Four target/case comparisons and the
corresponding geometry regression test passed. See
[geometry.py](../../../samples/vision/paddle_ocr/runtime/python/geometry.py),
`crop_and_rotate_image`, lines 120–135.

### OCR-3 — P2: X5 ragged Clipper output was silently skipped (fixed)

Found by the coordinating agent; independently rechecked here. The initial
shared helper used S's filtering order for X5. X5's original source converts
Clipper output with `np.array` before checking polygon count, so a ragged
multi-polygon raises on this NumPy version. S filters it first and skips it.

The helper now takes the target and preserves each order. Ten comparisons
against original-source functions covered empty output, an empty polygon,
one polygon, equally sized multiple polygons and ragged multiple polygons.
All matched, including X5 `ValueError` versus S empty output for the ragged
case. See [geometry.py](../../../samples/vision/paddle_ocr/runtime/python/geometry.py),
`dilate_contours`, lines 66–79, and its target argument in
`OCRPipeline.postprocess_detection`.

### OCR-4 — P2: unsupported quantization metadata disappeared (fixed)

Found by this reviewer. `_normalise_numbers` discarded values it could not
convert to a scalar. Consequently `output_scales={name:[0.5]}` or
`[0.5,0.75]`, and `output_zero_points={name:[0]}`, became empty mappings and
bypassed the F32 contract's explicit rejection of unverified quantization.
All four target/stage contracts accepted these values before repair.

Normalization now retains the presence of unsupported values. Twenty-four
probes through `from_runtime` and `bind_stage` rejected lists, a NumPy vector,
an unknown string and `None` attached to the bound output. The added binding
test also passes. See
[model_binding.py](../../../samples/vision/paddle_ocr/runtime/python/model_binding.py),
`_normalise_numbers` at line 723 and the `bind_stage` rejection at lines 424–427.

### OCR-5 — P2: preparation could map two assets to one file (fixed)

Found by this reviewer in `_prepare`. Distinct explicit paths ending in
`detector/model.bin` and `recognizer/model.bin` both become
`<model-dir>/model.bin` after applying `--model-dir`. The loop did not reject
the collision. Because these manifest rows have no publisher digest, the
shared downloader accepted the first file as the existing second destination.

A local-only reproduction replaced `urllib.request.urlopen` with `BytesIO`
and exercised the real shared downloader in a temporary directory: exit 0,
one transport call, two reported prepared assets with one identical path and
digest, and only detector-response bytes on disk. Directly supplying the same
destination for both stages had the same problem.

The resolver now rejects identical explicit model paths, and `_prepare`
rejects collisions after applying `--model-dir`. Both original reproductions
were rerun independently: different parents with the same basename plus
`--model-dir`, and directly identical paths. Each returned exit 2 with zero
`download_asset` calls, zero transport calls and no artifact created. See
[main.py](../../../samples/vision/paddle_ocr/runtime/python/main.py),
`_prepare`, lines 225–228; the direct-path check is in
`model_binding.resolve_pair`, lines 354–358.

### Delivery corrections

- The coordinating agent identified that automatic Windows line-ending
  conversion could change the hash-bound dictionary and the shell wrapper.
  The local [.gitattributes](../../../samples/vision/paddle_ocr/.gitattributes)
  now sets `text eol=lf` for both. `git check-attr` confirmed `text: set` and
  `eol: lf`; this review did not create a fresh checkout to claim that test.
- Final documentation inspection found 18 relative links with one missing
  parent component in test-data provenance, runtime compatibility and evaluator
  source-audit links. The Chinese top-level README also described the source
  dictionary digest as if blank/space tokens had already been added. All were
  corrected. A final pass resolved all 49 local Markdown links successfully,
  and the Chinese text now identifies the raw source-file digest separately
  from token construction. Both language versions retain the finite target
  scope and do not claim accuracy/performance measurements.

## Verification executed by this reviewer

Host: Python 3.13.5, NumPy 2.2.6 and OpenCV 4.11.0 on Windows. Real Clipper
comparisons used the existing isolated local pyclipper 1.4.0 directory; no
dependency was downloaded or installed during review.

| Check | Executed result |
| --- | --- |
| Final full OCR unittest suite after all code repairs | 31 tests passed in 3.000 seconds, including subprocess invocation from temporary working directories |
| Original-source preprocessing | 32/32 stage comparisons were exact: four image sizes, two seeds, two targets, two stages |
| Original-source ordinary detector geometry | 30/30 scene comparisons passed: three image sizes, five score maps, two targets; these scenes produced 28 ordered box/crop pairs in total, all exact |
| Original-source CTC | 14/14 score fixtures produced identical strings for both vocabularies, including blanks, repeats, duplicate space classes, ties and random scores |
| Degenerate crop / Clipper policies after repair | 4/4 crop and 10/10 polygon-policy comparisons matched the original functions |
| Quantization presence after repair | 24/24 unsupported metadata cases rejected |
| Stage failure and result lifetime probes | Detector exception/wrong dtype prevented recognition; recognizer exception/nonfinite output stopped at the first failed crop with stage context; two-crop ordering, independent result boxes and crop storage passed |
| Core lazy loading | Fresh-process import and construction did not load SDK, cv2, pyclipper or scipy; target mismatch prevented SDK factory construction |
| Inspection modes | Help, list-auto and dry-run auto/X5/S100 passed while SDK/cv2/pyclipper/scipy/NumPy imports and URL transport were blocked; JSON modes parsed successfully |
| Preparation collisions after repair | Both different-parent/same-basename and directly identical destinations failed before any downloader/transport invocation |
| Delivered documentation and attributes | 49 local Markdown links resolved; dictionary and run.sh both report `text: set`, `eol: lf` |

Final suite command executed from the worktree root:

```text
python -B -m unittest discover -s samples/vision/paddle_ocr/tests -p 'test_*.py' -v
```

Original-source comparisons extracted the relevant functions/classes directly
from `git show <fixed baseline>:<path>` with AST, then executed those functions
against candidate outputs. They did not copy an expected implementation from
the candidate. Geometry comparisons included rotated rectangles, edge/out-of-image
boxes, empty maps and the float32 threshold boundary.

The three delivered fixtures matched their original platform files byte for byte:

| File | SHA-256 |
| --- | --- |
| `test_data/x5/paddleocr_test.jpg` | `5b4a7fb523c7c459c8d3cec67480c1872cd7b3674b34505467420561ad8c577e` |
| `test_data/s100/gt_2322.jpg` | `18a214e1c637fb3a53f71673c6f6a689b5f16d755237ab7e9e58ddc32223580b` |
| `test_data/s100/ppocrv6_dict.txt` | `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6` |

## Reviewed runtime identity and supplied board evidence

The coordinator's candidate archive is
`c27c8bfeb3148c23b2c1d718913b1acd40e43c8d86ee86004a11a3a4d2d48547`.
This reviewer independently hashed the local archive and compared the eight
runtime files below with its entries, without extracting or changing them.
All eight matched the final reviewed worktree bytes.

Paths in this table are relative to `samples/vision/paddle_ocr/runtime/python/`.

| File | SHA-256 |
| --- | --- |
| `model_binding.py` | `8a973c2dcc94b23d9c25765b4d4bb0950ebcc5ed27d438d513f4540936f346f2` |
| `model_runner.py` | `37c4ce2e329c71ef5efaa485b80cfff1061df21cf5ec46ee809dc5422abfa494` |
| `tensor_io.py` | `9b097201f6023e2c8dedc7d7ff44782ca69906c3cac3620a525db0aa5137c87e` |
| `geometry.py` | `3210f3abf2401ac8efa9321e7dc5a79abb56b3270f33229cff2228b1ac2e3a28` |
| `decode.py` | `936caa8af5a2aab6e5bfc9c66ecd4c938864231c1f3054fc005b01a4c5e78822` |
| `pipeline.py` | `1b04154485142ac9eb90bb64497d1274bd52ddd8157272a8636547583a71f4f6` |
| `main.py` | `b318b505a4cb85d2d58b7ef2526f5d2387ad2e9f5ddf17e3b73197f44093b8a5` |
| `run.sh` | `8ed3c45e662130a7ab555f77802c1c119197782cebdd31399ef3a0cd21847a57` |

The following runs were executed by the coordinator, not by this reviewer.
Their saved logs were inspected locally. Each records the archive digest above,
the two model digests/references, default and changed-aspect inputs,
`max_raw_output_abs_diff: 0.0`, exact inputs/crops/boxes, exact native JSON,
result lifetime, an overall comparison-pass marker and exit status 0.

| Coordinator evidence | Default / changed-aspect detection counts |
| --- | --- |
| [X5 8G](evidence/x5-8g-p2-ocr-comparison-attempt1.log) | 6 / 5 |
| [X5 4G](evidence/x5-4g-p2-ocr-comparison-attempt1.log) | 6 / 5 |
| [S100](evidence/s100-p2-ocr-comparison-attempt1.log) | 9 / 6 |

## Acceptance boundary

The reviewer's evidence is source inspection and host execution only. Real
candidate detector/recognizer inference and native shell execution were not
run by this reviewer. The three coordinator board comparisons above are
separate evidence for these four artifacts and these input cases; they do not
extend support to other OCR models or targets. Accuracy and performance remain
unmeasured here. No result above certifies publisher provenance for
the four model artifacts, whose existing manifest rows omit publisher SHA-256.
