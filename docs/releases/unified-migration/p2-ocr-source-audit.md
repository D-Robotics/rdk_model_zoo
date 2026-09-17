# P2 PaddleOCR source audit: X5 versus S

Date: 2026-09-16
Scope: static, read-only comparison of the existing X5 and S PaddleOCR samples.
Decision status: **candidate for one logical OCR Sample with target-local adapters; not source-equivalent and not yet a migration/board-acceptance claim.**

This audit follows the source-equivalence rule in the active [X5/S model-zoo specification](../../superpowers/specs/2026-09-16-rdk-model-zoo-x5-s-agent-people-spec.md) and expands the `paddleocr` / `paddle_ocr` row in the [X5/S migration map](x5-s-migration-map.md). It inspects the checked-in source, conversion notes, release manifests and existing test data only. The supplied board evidence is linked where it constrains the comparison; this audit did not download or load a model or run a board.

## Finding

The two samples share the high-level shape `image -> DB detector -> contour boxes/crops -> CTC recognizer`, but they are different published model families and different runtime contracts:

- X5 is **PP-OCRv3**, with an English fixed 96-character alphabet and two `.bin` assets.
- S is **PP-OCRv6**, with a line-oriented large dictionary (the checked-in `ppocrv6_dict.txt` has 18,708 entries) and two `.hbm` assets; the release manifest currently records only the `s100/` pair.
- X5 packs the detector's Y and UV planes into one NV12 tensor in Python. S Python and S C++ provide two detector tensors (Y and UV), writing plane data with device strides in the C++ path.
- The X5 wrapper hard-codes the recognition output `(T=40, C=97)` and its alphabet. The S wrappers read recognition dimensions from model metadata and receive the dictionary from the entry point.
- X5 has a Python runtime only. S has independent Python detector/recognizer wrappers and a native C++ implementation with explicit DNN/UCP memory management.
- Conversion inputs, toolchains, `march` values, output prefixes, calibration paths and output-quantization handling differ. The same YAML or binary cannot be substituted by filename.

Therefore a future unified sample may share the **logical two-stage orchestration and a carefully parameterized result contract**, but it must retain per-target model bindings and preprocessing/postprocessing policies until the acceptance evidence below exists.

## Source and asset inventory

The source anchors below are the symbols used for the comparison. Line numbers refer to the files in this worktree and are included to make the audit reviewable.

| Area | X5 | S | Audit consequence |
| --- | --- | --- | --- |
| Python entry | [`runtime/python/main.py`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/main.py), `main()` 122-155 | [`runtime/python/main.py`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/main.py), `_resolve_model_soc()` 67-80 and `main()` 83-182 | Both construct a two-stage run, but S selects an SoC variant and downloads in `main`; X5 uses local sample paths. Do not copy S's fallback/download policy into a unified entry. |
| Python wrapper | [`paddleocr.py`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py), `PaddleOCR` 130-408 | [`paddle_ocr.py`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/paddle_ocr.py), `PaddleOCRDet` 185-374 and `PaddleOCRRec` 381-542 | X5 exposes one integrated `predict(image) -> (boxes, texts)`. S exposes detector and recognizer as separate public stages; its entry point composes them. |
| Native runtime | No X5 PaddleOCR C++ entry in the source tree. | [`runtime/cpp/src/main.cpp`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/main.cpp), `main()` 115-233; [`paddle_ocr.cpp`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/paddle_ocr.cpp), `PaddleOCRDet::init()` 244-287 and `PaddleOCRRec::init()` 313-360 | S C++ is a delivered capability and must remain available if the unified sample claims it. It cannot be treated as an X5 implementation detail. |
| X5 release assets | [`platforms/x5/docs/release/models.yaml`](../../../platforms/x5/docs/release/models.yaml): `paddleocr` 430-447: `en_PP-OCRv3_det_640x640_nv12.bin`, `en_PP-OCRv3_rec_48x320_rgb.bin` | — | Use qualified references `x5:paddleocr:<filename>`, not a cross-platform `paddleocr` model ID. |
| S release assets | — | [`platforms/s/docs/release/models.yaml`](../../../platforms/s/docs/release/models.yaml): `paddle_ocr` 432-449: `s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`, `s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | The manifest does not contain `s600/` or `s100p/` rows. Source README/scripts mention those variants, but that is not a published manifest asset fact. Use `s:paddle_ocr:s100/<filename>` for the recorded pair and treat other variants as requiring separate evidence. |
| Existing benchmarks | [`platforms/x5/docs/release/benchmarks.yaml`](../../../platforms/x5/docs/release/benchmarks.yaml): `paddleocr-v3-det-x5` 247-258 and `paddleocr-v3-rec-x5` 259-270 | No `paddle`/`ocr` benchmark rows in [`platforms/s/docs/release/benchmarks.yaml`](../../../platforms/s/docs/release/benchmarks.yaml) | X5's 158.12/245.68 FPS values are PP-OCRv3 README throughput references; they are not S measurements and cannot be reused. |
| Evaluation | No X5 `paddleocr/evaluator/` directory. | [`evaluator/README.md`](../../../platforms/s/samples/vision/paddle_ocr/evaluator/README.md) is only a three-line “content pending” note | Neither source provides an executable OCR accuracy evaluator or a machine-readable expected result. |

The two checked-in default images are also different fixtures: X5 `paddleocr_test.jpg` is 3888x2592, while S `gt_2322.jpg` is 760x1080. Existing output pictures are demonstration artifacts, not a cross-platform golden set.

### Supplied artifact metadata evidence

Read-only metadata captures are checked in for both X5 captures and for S100: [X5 8G](evidence/x5-8g-p2-ocr-metadata.log), [X5 4G](evidence/x5-4g-p2-ocr-metadata.log), and [S100](evidence/s100-p2-ocr-metadata.log). The two X5 captures report the same model contracts.

| Captured target and qualified assets | Detector metadata | Recognizer metadata |
| --- | --- | --- |
| X5 8G/4G: `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` and `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` | Model `en_PP-OCRv3_det_infer-deploy_640x640_nv12`; input `x`, shape `[1,3,640,640]`, `NV12`; output `sigmoid_0.tmp_0`, shape `[1,1,640,640]`, `F32` | Model `en_PP-OCRv3_rec_infer-deploy_48x320_rgb_NCHW`; input `x`, shape `[1,3,48,320]`, `F32`; output `softmax_2.tmp_0`, shape `[1,40,97,1]`, `F32` |
| S100: `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` and `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | Model `PP-OCRv6_det_infer-deploy_640x640_nv12`; inputs `x_y` `[1,640,640,1]` and `x_uv` `[1,320,320,2]`, both `U8`; output `fetch_name_0`, shape `[1,1,640,640]`, `F32` | Model `PP-OCRv6_rec_infer-deploy_48x320_rgb`; input `x`, shape `[1,3,48,320]`, `F32`; output `fetch_name_0`, shape `[1,40,18710]`, `F32` |

The X5 model metadata exposes an NV12 input as `[1,3,H,W]`, while the checked-in Python wrapper constructs the packed runtime buffer `(1,H*3/2,W,1)`; this is a runtime representation detail that must stay in the X5 adapter. S100 exposes the detector's Y and UV planes as distinct inputs, matching its Python and C++ writers. The output names `sigmoid_0.tmp_0`, `softmax_2.tmp_0` and `fetch_name_0` are graph/runtime names only; they do not prove whether a returned score vector is logits or probabilities. Keep the legacy post-processing policy explicit and require graph or numerical evidence before applying an activation.

All metadata records have `publisher_sha256: null`; the observed local digests are X5 detector `7b5b7f881f01be03d8bb398bce1afa4c527bb9c2638667627b80a918a81627e0`, X5 recognizer `61042ee7262f927b90c1b25ba81335a2533c1d0d6e36d531a49e7726385920d7`, S100 detector `d731680b32501989b4f9f052f6c7b60a187a31157ad849bb4ed659f074be0e08` and S100 recognizer `9710f81bf7431953ea473846d17cd0006f211fccb882450eed13ea28bc3691ce`. The X5 digests are the same in both captures. These local digests identify the observed files but do not prove their published origin. The captures report X5 DNN/HBRT `1.24.5_(3.15.55)` with model builder `1.23.8` and a model-build `3.15.54.0` versus runtime `3.15.55.0` warning, and S100 DNN/UCP `3.13.6` with HBRT `4.7.5`. `pyclipper` is present in the X5 8G capture and absent in the X5 4G and S100 capture environments, so dependency packaging is an acceptance item rather than an assumed common host capability.

## Runtime and algorithm comparison

### Entry flow and composition

X5's [`main()`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/main.py) parses detector/recognizer paths, threshold, output path and scheduling flags, constructs one `PaddleOCRConfig`, calls `PaddleOCR.predict()`, then draws boxes and text. The wrapper's [`PaddleOCR.predict`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) runs detection once and loops over boxes, calling `_crop_and_rotate`, `_rec_pre_process`, `_rec_forward` and `_rec_post_process` for each crop.

S's Python [`main()`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/main.py) creates a `PaddleOCRDet`, runs its `pre_process/forward/post_process`, creates a `PaddleOCRRec`, then runs recognition on every returned crop. The public return values are different: detector `predict()` returns `(img_boxes, cropped_images, boxes_list)` and recognizer `predict()` returns one text string. S C++ follows the same sequence in [`main.cpp`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/main.cpp): `pre_process_det -> infer -> post_process_det`, then `pre_process_rec -> infer -> post_process_rec` per crop.

Both paths are sequential per crop and do not batch recognition. The shared concept is a two-stage composition; the public API, lifecycle, visualization and error behavior are not interchangeable.

### Detection input and output

| Contract point | X5 source behavior | S source behavior |
| --- | --- | --- |
| Image resize | `PaddleOCR.pre_process()` [`paddleocr.py:187-210`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) calls `cv2.resize(image, (W,H))`, whose default is linear interpolation. | Python `PaddleOCRDet.pre_process()` [`paddle_ocr.py:258-280`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/paddle_ocr.py) calls `pre_utils.resized_image(..., resize_type=0, interpolation=cv2.INTER_AREA)`; C++ `pre_process_det()` [`paddle_ocr.cpp:381-392`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/paddle_ocr.cpp) uses `INTER_AREA`. |
| NV12 input shape | Calls [`bgr_to_nv12_planes`](../../../platforms/x5/utils/py_utils/preprocess.py) and concatenates Y then UV into one packed `(1, H*3/2, W, 1)` array (`paddleocr.py:204-210`). | Python returns separate Y `(1,H,W,1)` and UV `(1,H/2,W/2,2)` tensors (`paddle_ocr.py:272-280`). C++ `bgr_to_nv12_tensor()` [`platforms/s/utils/c_utils/src/preprocess.cpp:111-153`](../../../platforms/s/utils/c_utils/src/preprocess.cpp) writes the two tensors row-by-row using tensor strides and flushes caches. |
| Metadata use | Reads the first detector input name and takes H/W from indices `[2]`/`[3]`; it uses those values to reshape the packed buffer. It does not bind or validate output shape/dtype metadata, even though the observed X5 artifacts report the concrete contract above. | Python reads plane dimensions from indices `[1]`/`[2]`; C++ `PaddleOCRDet::init()` reads the same plane layout and queries every tensor property. The observed S100 artifact confirms the two U8 planes and F32 map above. |
| Prediction map | `PaddleOCR.post_process()` [`paddleocr.py:225-250`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) reshapes raw output to `(1,H,W)`, thresholds at `det_threshold`, then resizes to the source image. | Python thresholds the returned output after `squeeze()` and resizes it. C++ `process_and_resize_pred()` [`paddle_ocr.cpp:82-123`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/paddle_ocr.cpp) derives H/W from the output tensor and has separate S16+scale and F32 paths before `INTER_LINEAR` resize. |
| Geometry | X5 `_dilate_contours`, `_get_bounding_boxes` and `_crop_and_rotate` are local methods (`paddleocr.py:252-361`). | S Python has local `dilate_contours()` and calls platform S `post_utils.get_bounding_boxes()` / `crop_and_rotate_image()`; S C++ has Clipper/OpenCV equivalents. |

The DB geometry formula is recognizably the same: `area * ratio_prime / perimeter`, round Clipper joins, minimum contour area 100 and a minimum-area rectangle. That establishes a useful candidate mapping, not byte-for-byte equivalence. X5's `np.array(pco.Execute(...))` filtering can reject ragged/multi-polygon output after conversion; S Python filters the solution count before constructing an array. S C++ converts points with `std::round` and rejects invalid rectangles, while the Python paths use NumPy integer conversion. These edge and rounding policies can alter boxes and crops.

### Cropping and recognition input

X5 [`PaddleOCR._crop_and_rotate`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) and S's Python [`post_utils.crop_and_rotate_image`](../../../platforms/s/utils/py_utils/postprocess.py) both use `minAreaRect`, `boxPoints`, perspective warp and a clockwise rotation when `angle >= 45`. X5 explicitly returns the original image for a zero-width/zero-height rectangle; S Python's helper has no corresponding guard. S C++ [`crop_and_rotate_image`](../../../platforms/s/utils/c_utils/src/postprocess.cpp) rejects fewer than four points or a degenerate rectangle. Preserve these target-specific failure policies until a fixture proves the desired unified behavior.

For ordinary Python crops, X5 `_rec_pre_process()` [`paddleocr.py:284-297`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) and S `PaddleOCRRec.pre_process()` [`paddle_ocr.py:454-480`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/paddle_ocr.py) both do direct resize, divide `uint8` pixels by 255, swap BGR to RGB and emit float32 NCHW. The S C++ `pre_process_rec()` [`paddle_ocr.cpp:463-491`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/paddle_ocr.cpp) does the same channel/range/layout conversion but explicitly uses `INTER_AREA`. A shared helper is safe only after a textured, non-square crop fixture checks Python and C++ interpolation and all channel/stride details.

### Recognition model and decoder

X5's [`ALPHABET`](../../../platforms/x5/samples/vision/paddleocr/runtime/python/paddleocr.py) is a fixed 96-character ASCII string (`paddleocr.py:60-61`). `_CTCLabelConverter` appends a blank sentinel and indexes blank as class 0; `PaddleOCRConfig.rec_output_size` defaults to `(40, 97)` (`paddleocr.py:64-83`). `_rec_post_process()` reshapes to that configured shape, takes an argmax per timestep and returns both raw and collapsed text (`paddleocr.py:310-330`). There is no label-file or model-class-count binding.

S's `ctc_greedy_decode()` [`paddle_ocr.py:148-178`](../../../platforms/s/samples/vision/paddle_ocr/runtime/python/paddle_ocr.py) also argmaxes per timestep, removes blank index 0 and collapses consecutive repeats, but it receives a caller-supplied `char_list`. `PaddleOCRRec.__init__()` reads `seq_len` and `num_classes` from model output metadata (`paddle_ocr.py:410-432`), and `main.py` prepends a blank and appends a trailing space to the line-oriented dictionary (`main.py:133-138`). For the checked-in `ppocrv6_dict.txt`, this is 18,708 dictionary entries plus blank and trailing space, or 18,710 classes. The C++ decoder [`ctc_greedy_decode_from_tensor`](../../../platforms/s/samples/vision/paddle_ocr/runtime/cpp/src/paddle_ocr.cpp) is stride-aware and uses the runtime-provided `seq_len`, `num_classes` and `id2token`.

The supplied metadata captures confirm that the X5 recognition output is `[1,40,97,1]` `F32` and the S100 output is `[1,40,18710]` `F32`. The trailing singleton in the X5 layout and the dictionary-sized S class dimension are contract differences, not cosmetic shape variants. The shared blank/repeat-collapse algorithm is a candidate pure function, but its tensor adapter, class-count check, token table and output score semantics must remain explicit per artifact.

Both conversion configs place selected Softmax nodes on the BPU, but that does not establish identical output semantics or class vocabulary. The X5 and S recognition artifacts must each be inspected and decoded with their own model/dictionary contract. A generic CTC decoder can be considered only as a parameterized pure function after exact class-count, blank-index, token-order and output-layout tests.

### Runtime/API and target selection differences

X5 and S Python wrappers both append a relative platform root to `sys.path` before importing `utils.py_utils.*` (`x5/.../paddleocr.py:52-55`, `s/.../paddle_ocr.py:52-64`). This is an old source-tree import convention, not evidence that the two utility trees are interchangeable. X5 scheduling is applied to both models by `PaddleOCR.set_scheduling_params()` (`paddleocr.py:164-185`); S applies it independently to `PaddleOCRDet` and `PaddleOCRRec` (`paddle_ocr.py:237-256`, `434-452`).

S `main.py` and both S shell scripts map S100P and unknown/unreadable SoCs to the S100 path, while X5 has no analogous SoC selection. The S C++ default is compile-time (`SOC_S600` versus the `s100` fallback in `runtime/cpp/src/main.cpp:50-67`). These are legacy behaviors that require an explicit target/support policy in a unified entry; a fallback must not silently turn unknown hardware into published support.

## Conversion and model preparation comparison

| Concern | X5 | S | Must retain |
| --- | --- | --- | --- |
| Source model/version | PTQ notes name `en_PP-OCRv3_det_infer.onnx` and `en_PP-OCRv3_rec_infer.onnx` in [`conversion/ptq_yamls/*.yaml`](../../../platforms/x5/samples/vision/paddleocr/conversion/ptq_yamls/paddleocr_det_config.yaml). | Notes export **PP-OCRv6** with Paddle2ONNX opset 19, then compile [`model_detv6.onnx` / `model_recv6.onnx`](../../../platforms/s/samples/vision/paddle_ocr/conversion/README.md#export-onnx-models-pp-ocrv6). | Model family and graph provenance; neither model may be renamed as the other. |
| Compiler / march | `hb_mapper checker` / `hb_mapper makertbin`, `march: bayes-e`, output prefixes `en_PP-OCRv3_*` ([X5 conversion README](../../../platforms/x5/samples/vision/paddleocr/conversion/README.md#model-check)). | `hb_compile`, with `nash-e`/`nash-m`/`nash-p` documented for S100/S100P/S600, output prefixes `PP-OCRv6_*` ([S conversion README](../../../platforms/s/samples/vision/paddle_ocr/conversion/README.md#multi-platform-compilation)). | Toolchain, target march, compiler options and generated artifact identity. |
| Detector config | `input_type_rt: nv12`, training RGB NCHW, mean/scale, calibration `./calibration_data_rgb_f32`, optimize O3; no source-level output dequantization policy. | `input_name: x`, `1x3x640x640`, same mean/scale, calibration `../calibration_data`, optimize O2; docs explicitly keep trailing Dequantize for an F32 probability map. | Runtime output dtype/quantization must be obtained from each artifact, not inferred from a README. |
| Recognizer config | `input_type_rt/train: featuremap`, NCHW, `norm_type: no_preprocess`, `set_all_nodes_int16`, three Softmax node mappings, output prefix `en_PP-OCRv3_rec...`. | Same broad featuremap/NCHW/no-preprocess and Softmax mappings, but PP-OCRv6 ONNX, calibration `calibration_data_rec_new/cropped_images_npy`, optimize O2. | Calibration domain, class count and graph postprocessing. |
| Model acquisition | [`model/download_model.sh`](../../../platforms/x5/samples/vision/paddleocr/model/download_model.sh) downloads the two X5 `.bin` files next to the sample; `runtime/python/run.sh` invokes it when either is absent. | [`model/download_model.sh`](../../../platforms/s/samples/vision/paddle_ocr/model/download_model.sh) and both `runtime/*/run.sh` select a variant, install missing dependencies and download into `/opt/hobot/model/<soc>/basic/`; S Python `main.py` also calls `download_model_if_needed`. | A new main/runner must use manifest-resolved qualified assets and must not add implicit network activity to normal inference. |

The X5 detector conversion uses compiler normalization (`data_mean_and_scale`) while the runtime supplies NV12. The S detector documentation additionally specifies retaining Dequantize for F32 output; S C++ still contains an explicit S16 legacy path. Those facts are sufficient to require artifact metadata checks, but insufficient to declare a common raw-output semantic.

## Function-level mapping and merge decision

| Existing X5 symbol | Closest S symbol(s) | Decision |
| --- | --- | --- |
| `PaddleOCR.pre_process` | `PaddleOCRDet.pre_process`; C++ `pre_process_det` | **Adapter mapping only.** Same image-to-NV12 intent, different packed versus dual-plane tensor contract and interpolation (`INTER_LINEAR` default versus `INTER_AREA`). |
| `PaddleOCR.forward` | `PaddleOCRDet.forward`; C++ `infer` | **Keep runtime-local.** Python HBM calls and C++ DNN/UCP task/memory APIs are different. |
| `PaddleOCR.post_process` | `PaddleOCRDet.post_process`; C++ `post_process_det` / `process_and_resize_pred` | **Keep per target until metadata and golden-map evidence.** Shape sourcing and S16/F32 handling differ. |
| `PaddleOCR._dilate_contours` | S `dilate_contours`; C++ local `dilate_contours` | **Candidate pure geometry utility, not yet mergeable.** Formula is shared, but ragged-result handling, integer type, diagnostics and Clipper implementation differ. |
| `PaddleOCR._get_bounding_boxes` | S Python `post_utils.get_bounding_boxes`; C++ `get_bounding_boxes` | **Candidate only after exact geometry tests.** Python implementations are close; C++ rounds points and validates rectangle dimensions. |
| `PaddleOCR._crop_and_rotate` | S Python/C++ `crop_and_rotate_image` | **Retain adapters.** Degenerate input behavior and language/runtime types differ. |
| `PaddleOCR._rec_pre_process` | S `PaddleOCRRec.pre_process`; C++ `pre_process_rec` | **Potential shared image policy, target-local writers.** Python X5/S agree on `[0,1]` RGB NCHW; C++ uses explicit `INTER_AREA` and device strides. |
| `PaddleOCR._rec_post_process` | S `ctc_greedy_decode`; C++ `ctc_greedy_decode_from_tensor` | **Parameterize only after proof.** Greedy CTC steps match, but dictionaries, class counts, shape handling and C++ stride access differ. |
| `PaddleOCR.predict` | S `main` composition of `PaddleOCRDet` + `PaddleOCRRec`; C++ `main` | **Share the logical pipeline/result vocabulary, preserve public compatibility wrappers.** Return tuples and visualization outputs are not the same. |
| X5 `draw_boxes_and_texts` | S `vis_utils.draw_polygon_boxes` + `draw_text`; C++ drawing helpers | **Keep target presentation.** X5 uses OpenCV text; S uses Pillow/FreeType for Unicode text and different canvas composition. |

### What can be merged

1. A documented, target-independent **logical pipeline contract**: one BGR image, detector result with ordered pixel boxes/crops, recognizer result per crop, and a final ordered OCR result. Each target adapter must bind its own model names, tensor layout, dtype, stride and output semantics.
2. A pure CTC greedy decoder only if it accepts an explicit token table and tensor adapter and is protected by tests for blank, repeats, empty output, Unicode token, class count and layout. The existing fixed X5 converter must remain available for the PP-OCRv3 compatibility path.
3. Possibly the normal-contour geometry formula after host tests prove identical point ordering, integer conversion, angle handling, minimum area and crop pixels. Until then, retaining small local adapters is lower risk than changing a shared utility.
4. Asset enumeration/preparation may consume the existing manifest reader, but the detector and recognizer remain separate qualified asset records. The X5 references are `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` and `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin`; the currently recorded S references are `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` and `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`.

### What must remain separate

- PP-OCRv3 versus PP-OCRv6 weights, graph provenance, filenames, URLs, hashes and dictionaries.
- X5 packed-NV12 input versus S dual-plane Y/UV input; model metadata binding and device stride writers.
- Detector output shape/dtype/quantization handling, including S C++'s S16 compatibility branch and the S documentation's F32 Dequantize requirement.
- Recognition class count/token order: fixed X5 ASCII 97-class contract versus S model-metadata-driven dictionary contract (18,710 classes with the checked-in PP-OCRv6 dictionary plus blank/trailing space).
- X5 Python-only runtime versus S Python and native C++ entry points, native dependencies and resource lifecycle.
- SoC selection and model acquisition behavior; S100P/S600/unknown claims require explicit assets and evidence. A source fallback is not a support declaration.
- Conversion YAMLs, calibration data and toolchain commands.
- Evaluation and benchmark records. X5 historical PP-OCRv3 throughput values do not validate S PP-OCRv6 and the S evaluator is currently empty.

## Minimum representative model and acceptance gate

The minimum **functional** representative is a detector/recognizer pair, because either component alone cannot exercise crop ordering, dictionary binding or the end-to-end OCR result. For the source-equivalence decision, use one pair per distinct platform/model contract:

1. **X5 PP-OCRv3 pair** — the two X5 manifest assets above, with `paddleocr_test.jpg` as the initial fixed fixture.
2. **S100 PP-OCRv6 pair** — the two S `s100/` manifest assets above, with `gt_2322.jpg` and the checked-in `ppocrv6_dict.txt` as the initial fixture.

This is a minimum comparison set, not blanket support. S600 and S100P must be treated as separate acceptance targets if their artifacts are retained or added to the catalog. The existing source claim that S100P reuses S100 and the script's S600 URL construction do not create manifest support by themselves.

Before calling a unified implementation accepted, collect the following evidence for each claimed target, runtime and pair:

1. **Artifact identity and metadata:** the supplied captures establish the initial X5 8G/4G and S100 contracts above; retain their qualified references and observed local digests in the comparison ([X5 8G](evidence/x5-8g-p2-ocr-metadata.log), [X5 4G](evidence/x5-4g-p2-ocr-metadata.log), [S100](evidence/s100-p2-ocr-metadata.log)). For every additional target or artifact, query every detector/recognizer input and output name, count, shape, layout, dtype, stride and quantization. Confirm X5 packed NV12 versus S Y/UV planes and S recognition `(T,V)` against the dictionary length. A missing or unknown dtype is a failure, not an inferred F32.
2. **Host preprocessing golden tests:** use the same textured, non-square image and at least one rotated/non-degenerate crop. Compare exact X5 packed-NV12 bytes, S Y/UV planes, RGB NCHW float values, resize interpolation, channel order, padding and dtype. Compare Python and S C++ writers when both are claimed.
3. **Detector stage regression:** capture raw detector outputs and final mask/contours/boxes for the old source and candidate path using the same artifact. Check threshold `0.5`, dilation `2.7`, minimum area `100`, box order/coordinates, crop dimensions and crop pixels. Test an S16 output fixture separately from F32 if the artifact exposes it.
4. **Recognition stage regression:** assert dictionary file format, exact token order, blank index 0, repeat collapse and output shape/class count. Run known synthetic logits for blank/repeat/Unicode cases and one real crop; do not substitute the X5 alphabet for the S dictionary or apply an unverified softmax/logit policy.
5. **Full-pipeline result:** on each target, run the fixed image and at least one additional aspect ratio. Compare detection count/boxes, recognized strings, saved result image and a repeated invocation. The supplied legacy captures record six X5 boxes and nine S100 boxes; both include boxes with coordinates outside the image bounds, so a candidate must preserve that observable behavior or document an explicit contract change rather than silently clip. See [X5 8G baseline](evidence/x5-8g-p2-ocr-baseline.log), [X5 4G baseline](evidence/x5-4g-p2-ocr-baseline.log), and [S100 baseline](evidence/s100-p2-ocr-baseline.log). The X5 and S strings need not be identical because the model families/dictionaries differ; each must match its own legacy baseline.
6. **Entry behavior:** verify help/list/dry-run paths remain SDK-free and network-free, model execution fails clearly on target mismatch, and normal inference does not download implicitly. Preserve old CLI wrappers only through explicit compatibility mapping.
7. **Runtime and performance:** run both Python and the S C++ path where claimed; measure detector, recognizer and full-pipeline latency separately on the actual board. Do not carry the X5 README's 158.12/245.68 FPS into S. Record board/OS/SDK/runtime versions and exact commands.
8. **Evaluation:** add a real OCR ground-truth/evaluator path or mark accuracy as `not-run`. Existing screenshots and the empty evaluator README cannot be an acceptance result.

Until those checks are recorded, the recommended status is **source-audited / integration pending / board acceptance pending**. The legacy X5 and S directories should remain untouched while a unified candidate is reviewed and compared against these contracts.

## 中文摘要

X5 的 `paddleocr` 是 PP-OCRv3 英文模型（单个打包 NV12 输入、观测到固定 97 类），S 的 `paddle_ocr` 是 PP-OCRv6 模型（Y/UV 双输入、运行时读取大字典，且保留 C++ 入口）。两者都采用 DB 检测 + CTC 识别，但模型版本、资产、输入布局、输出量化、字典、转换工具链和 API 均不同，不能按目录名或任务名直接替换。元数据证据见 [X5 8G](evidence/x5-8g-p2-ocr-metadata.log)、[X5 4G](evidence/x5-4g-p2-ocr-metadata.log) 和 [S100](evidence/s100-p2-ocr-metadata.log)；未记录发布方哈希，不能据此证明来源。

最小代表应是每个平台各一组“检测 + 识别”资产；当前清单中 S 只记录 `s100/` 资产。后续归并可以共享两阶段流程描述及经过测试的纯 CTC/几何适配器，但必须保留平台绑定、预处理、量化输出、字典、C++ 实现和资产身份。板端、模型元数据、预处理金样、检测框/裁剪、识别字典、连续调用和评估证据完成前，不应宣称 OCR 已完成统一迁移。
