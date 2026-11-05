English | [简体中文](README_cn.md)

# YOLOE model artifacts

<a id="artifacts"></a>
## Artifacts

These are the 16 original published artifacts. The Python runtime supports the published S mixed float/SCALE output models directly. Optional local float models can be prepared via [conversion](../conversion/README.md). S100 11m and 11l have no recorded benchmark in this repository.

| Asset ID | Target | Variant | Output route |
| --- | --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | x5 | 11s | floating |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | x5 | 11m | floating |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | x5 | 11l | floating |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11s | native SCALE outputs (Python) |
| `s:yoloe11_seg:yoloe_11m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11m | native SCALE outputs (Python) |
| `s:yoloe11_seg:yoloe_11l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11l | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26n | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26s | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26m | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26l | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26x | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26n | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26s | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26m | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26l | native SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26x | native SCALE outputs (Python) |

## Converted X5 11s (2026-11-02)

The independent variant `11s-native-20261102` selects `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12_float_20261102.bin` (12,841,075 bytes; SHA-256 `352e8b8e4b445a8223c2a76a7fdad5594b3317a6ea04e47cd70bdc2066fdc7df`). It preserves all ten NHWC FLOAT32 heads and the fixed 4585-class vocabulary. OE X5 1.2.8 completed with 200 real COCO calibration images; the host export passed the original gates on three real images and seed-0 random input. Input is full-range packed NV12 matching the SDK calibration transformer (float32 coefficients, uint8 truncation, top-left chroma sampling); the BIN graph descriptor is INT8 NHWC `[1,640,640,3]`, while the source ONNX input is FLOAT32 NCHW `[1,3,640,640]`. Preprocessing keeps native linear letterbox, truncated dimensions, padding 127 and RGB /255 once.

Select it explicitly with `--target x5 --variant 11s-native-20261102` in the downloader or Python/C++ launcher. The original `11s` default and historical measurements remain unchanged. This artifact's model license is AGPL-3.0; [public source and weight identities](../conversion/UPSTREAM_NOTICE.md) retain the original license text.

The complete OE report contains all ten output cosine diagnostics (0.798033–0.981209); the comparison sample count is unknown. These values describe quantization diagnostics, not dataset accuracy. Dataset accuracy, board validation and board performance have not been measured for this artifact.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── vocabulary.py  # Python script
```

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
```

Download or checksum failure exits nonzero; an existing mismatched file is not overwritten. For an offline copy of the original artifact, supply exact `--asset-id` and `--model-path`. The Python entry accepts the original S artifacts and dequantizes their integer heads using runtime SCALE metadata.

```bash
# cwd: repository root; original quantized publication only
bash samples/vision/yoloe/model/download.sh --target s100p --variant 26n
```

<a id="accompanying-files"></a>
## Accompanying Files

Required vocabulary: [classes.names](../test_data/classes.names), 4585 ordered entries aligned with model class IDs; the CLI rejects a changed digest. [office_desk.jpg](../test_data/office_desk.jpg) is replaceable. Editing the vocabulary does not teach the model new classes.

<a id="local-paths"></a>
## Local Paths

Defaults are `model/<target>/<manifest filename>`; S26 keeps the `nash-e/` or `nash-m/` subdirectory. `--output-dir` appends the manifest filename directly. Runtime `--model-path=null` derives this path. Defaults: X5/S100 11s, S100P 26n; an exact asset ID selects its variant when variant is omitted.

A separately converted float HBM requires both `--model-path` and `--local-float-sha256`: the digest identifies the exact local bytes for selection, and `source_asset_id` records the protocol provenance of the conversion. Loading then verifies the target and all ten NHWC float32 output roles, and target compatibility is confirmed by those checks together with a board run.

Use the [conversion preparation guide](../conversion/README.md) for ONNX checks, target-specific calibration and optional compilation. Generated `_float` names express intent; `compiled_unverified` is not a verified runtime artifact.


## X5 model file sizes


| Model | Download | Size (MB) |
| --- | --- | ---: |
| YOLOE-11s-Seg-PF | [yoloe_11s_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11s_seg_pf_bayese_640x640_nv12.bin) | 13.17 |
| YOLOE-11m-Seg-PF | [yoloe_11m_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11m_seg_pf_bayese_640x640_nv12.bin) | 26.73 |
| YOLOE-11l-Seg-PF | [yoloe_11l_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11l_seg_pf_bayese_640x640_nv12.bin) | 32.83 |

<a id="formats-checksums"></a>
## Formats & Checksums

X5 uses BIN and S uses HBM. Known digests come from active manifests. S11 publisher SHA-256 remains unknown; no sibling digest is substituted.

| Asset ID | SHA-256 | Authority |
| --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | `8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | `ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | `a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2` | `docs/release/x5/models.yaml` |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe11_seg:yoloe_11m_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe11_seg:yoloe_11l_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | `bf4a03cb75fcfe67cdf6549ef0dec9541b5ae009cdef05e7879a4af9c75a5439` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | `a27c313dc922b778d3537187c7b55a82f87b826edc66abdef027e15770243ee6` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | `d257d502a5245de823cdb76cc734974a1568e68bff02bece232895db29aceeef` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | `695cd10aba09873bb9cd4f838113882945e979c745514185c7ab6f5cb17690c4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | `119a6b069b32282915948e161f548b7b22136b765b70101c33d22680ac6fef24` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | `58922c669bcccec7a00c11719db6c3b9c4eaa4e0388b58bd120fda914fe9d048` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | `73f838c5d52a73451b8af50002c020d571d8fd75beeb29c54e0a6c956e0c45a5` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | `8621baa458a94d3d1946665609633f1ecdd930efeeebe91d975f822ffd273af4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | `314df1d0d60c839f9ff4248ab0d4ff2adc3cc8e514a8e9cf9acca62527696738` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | `2abf9d30dd487f2d5eb53444b7272a71d7be18265a39bd463cfd5b0cd6c14539` | `docs/release/s/models.yaml` |

## Converted S100 profile (2026-11-03)

`26n-native-cal32-20261103-o1` · S100 / nash-e · OE 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5) · O1.

Select this independent build explicitly. Historical defaults and measurements are retained. Actual HBM outputs are float32.

| Role | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| model | `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1.hbm` | `00340e52995a559be99fce15909babc7cad09e66785f8a596ab71f67bdeb61c4` | 167770984 |

```bash
bash samples/vision/yoloe/model/download.sh --target s100 --variant 26n-native-cal32-20261103-o1
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n-native-cal32-20261103-o1 --test-img /path/to/image.jpg
```

Calibration uses 32 independent COCO training photographs. Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are `not-run`. Source, checkpoint, ONNX, effective configuration, native recipe and complete OE identities are recorded in [converted.json](../conversion/converted.json).

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100 | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |
| s100p | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |
| s600 | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |

```bash
python3 samples/vision/yoloe/model/download.py --target s100 --asset-id s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n-native-cal32-cpuptq-20261103-o1 --asset-id s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm --model-path /path/to/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm --test-img samples/vision/yoloe/test_data/office_desk.jpg
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.

| Target | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| s100 | `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `6582c867d89472f3b5c6aed29944fb15768fcc07adb376965dd831c617a0ed30` | 167799968 |
| s100p | `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `9358de5bc440818e81cd51fbdf669e68b65dbc32060b95fabc7304e5238ad0f2` | 174226456 |
| s600 | `s:yoloe26_seg:nash-p/yoloe_26n_seg_pf_nashp_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `bd831c5607a7091c19d63ba4e8ffa0db41d9b818f3a65139550fb19010253efd` | 12553336 |

<!-- BEGIN CPU S CONVERSIONS 20261104 -->

## Additional CPU PTQ S profiles (2026-11-04)

These exact trained-checkpoint exports passed the source host gate and the full-output CPU source/CPU SDK gate (three frozen images plus deterministic random input, atol=1e-4, rtol=1e-3). Each target has its own INT16 PTQ, compiler metadata, OE report and checksum in `converted_targets_20261103.json`. Compiler estimates describe conversion; board accuracy, Runtime 1/2 concurrency and C++ 1/2 streams remain not run.

Models and intermediate outputs stay on dev-server. The Mac holds code and metadata only; Archive uploads stream from server files without a local model copy.

| Target | Native variant | Manifest filename | SHA-256 | Bytes |
|---|---|---|---|---:|
| s100 | `26n-native-cal32-cpuptq-20261103-o1` | `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `6582c867d89472f3b5c6aed29944fb15768fcc07adb376965dd831c617a0ed30` | 167799968 |
| s100p | `26n-native-cal32-cpuptq-20261103-o1` | `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `9358de5bc440818e81cd51fbdf669e68b65dbc32060b95fabc7304e5238ad0f2` | 174226456 |
| s600 | `26n-native-cal32-cpuptq-20261103-o1` | `s:yoloe26_seg:nash-p/yoloe_26n_seg_pf_nashp_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `bd831c5607a7091c19d63ba4e8ffa0db41d9b818f3a65139550fb19010253efd` | 12553336 |
| s600 | `11s-int16-cal32-cpuptq-s-o1-20261103` | `s:yoloe11_seg:nash-p/yoloe_11s_seg_pf_nashp_640x640_nv12_float_20261103_o1_cpuptq.hbm` | `d0f400a5e1f7d9fe59cbde4b0b0b9841c16b19588d67b660bd7305dfeb5520e5` | 54169136 |

<!-- END CPU S CONVERSIONS 20261104 -->
