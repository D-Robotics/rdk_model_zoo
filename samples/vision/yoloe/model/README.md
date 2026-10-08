English | [简体中文](README_cn.md)

# YOLOE model artifacts

<a id="artifacts"></a>
## Artifacts

These are the 14 original published artifacts. S rows describe the published quantized-output models; for float-output inference on S, prepare a local float model via [conversion](../conversion/README.md) (no float S HBM is published).

| Asset ID | Target | Variant | Output route |
| --- | --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | x5 | 11s | floating |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | x5 | 11m | floating |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | x5 | 11l | floating |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26n | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26m | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26l | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26x | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26n | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26m | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26l | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26x | quantized / float conversion pending |

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

Download or checksum failure exits nonzero; an existing mismatched file is not overwritten. For an offline copy of the original artifact, supply exact `--asset-id` and `--model-path`. S originals may be downloaded explicitly for quantized-output use and source comparison, but cannot run through this float entry.

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
