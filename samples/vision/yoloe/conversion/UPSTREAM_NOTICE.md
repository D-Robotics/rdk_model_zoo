# Public YOLOE PF input versions

The following input identities apply to the new 11s/X5 and 26n/S100 conversion
candidates. They do not replace historical model identities, defaults or metrics.

| Input | Official asset | Size | SHA-256 and verification |
| --- | --- | ---: | --- |
| YOLOE-11s segmentation PF | [v8.3.0 release checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt), asset ID 240891586 | 27,948,751 | `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d`; observed download digest, no publisher digest supplied |
| YOLOE-26n segmentation PF | [v8.4.0 release checkpoint](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt), asset ID 570403017 | 16,271,345 | `ae805a9ac11c85214588b32da8248d68a516e1d071465e1e86a03cbfadb66ba7`; matched the publisher's SHA-256 |

Export implementation: Ultralytics 8.4.127, source commit
`8e1eb77948c6c4aa110ad0b4aaacb93fd9855967`. Its original
[AGPL-3.0 license](licenses/ultralytics-AGPL-3.0.txt) is retained from the
[pinned source](https://github.com/ultralytics/ultralytics/blob/8e1eb77948c6c4aa110ad0b4aaacb93fd9855967/LICENSE).
The [publisher's model licensing statement](https://www.ultralytics.com/license)
identifies trained Ultralytics YOLO models as AGPL-3.0 by default. The model
license is recorded per candidate; the sample's Apache-2.0 notices remain.

Both candidates retain the complete, checkpoint-ordered 4,585-name vocabulary,
SHA-256 `1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3`.
All ten floating NHWC outputs are required: three classification heads, three
box heads, three mask-coefficient heads, and all 32 prototype channels. No text
encoder or arbitrary prompt path is substituted for the PF model.

The native exporter keeps its original thresholds: upstream raw/decoded/prototype
comparisons use rtol/atol 1e-4; ONNX CPU comparisons use rtol/atol 2e-3 with
ORT_DISABLE_ALL. E26 additionally requires identical selected anchor/class sets.
Release-candidate validation adds three real images and the native seed-0 random
case to the same graph. These host checks do not establish board accuracy or speed.

## Completed X5 artifact

`11s-native-20261102` was compiled as `yoloe_11s_seg_pf_bayese_640x640_nv12_float_20261102.bin` (12,841,075 bytes; SHA-256 `352e8b8e4b445a8223c2a76a7fdad5594b3317a6ea04e47cd70bdc2066fdc7df`). OE X5 1.2.8 / Mapper 1.24.3 / HBDK 3.49.15 retained all ten FLOAT32 heads. The successful resource retry used two CPU cores and 40 GiB host memory, preserving the same ONNX graph, 200 real calibration images and default calibration/O3/core-1 policy after an 8 GiB attempt was killed by the memory limit. The float export passed the unchanged gates on three frozen COCO images plus seed-0 random input.

Actual PTQ output cosine diagnostics range from 0.798033 to 0.981209; comparison sample count was not stated. These diagnostics are retained without an accuracy claim. Board validation, board performance and dataset accuracy remain not-run. The 26n candidate's verified weights and dependency setup are retained, but export/PTQ/compile remain not-run following the user's pause request.

The new alias uses X5 `input_space_and_range=regular`: the SDK maps this to full-range `YUV444_128`, distinct from explicit `bt601_video`. Its raw calibration loader performs RGB float32 → full-range NV12 (top-left UV sampling, uint8 truncation) → YUV444 → subtract 128. The `scale_value=1/255` is supplied once to the graph preprocess; the loader does not divide the raw calibration pixels. Python and C++ bind this full-range conversion only for `11s-native-20261102`; legacy aliases keep their existing video-range I420 conversion. The compiled graph and model bytes are unchanged.
