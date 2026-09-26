[English](README.md) | [简体中文](README_cn.md)

# Model preparation

<a id="artifacts"></a>
## Published artifacts

Twenty target-specific manifest entries are selected by target and variant.
Default variant is n. BIN is X5; HBM belongs to its exact Nash march. S100P has
its own nash-m assets and never reuses nash-e implicitly.

| Target | Variant | Profile | Path relative to this model directory |
|---|---|---|---|
| x5 | n | nv12 | `yolo26n_depth_bayese_768x768_nv12.bin` |
| x5 | s | nv12 | `yolo26s_depth_bayese_768x768_nv12.bin` |
| x5 | m | nv12 | `yolo26m_depth_bayese_768x768_nv12.bin` |
| x5 | l | nv12 | `yolo26l_depth_bayese_768x768_nv12.bin` |
| x5 | x | nv12 | `yolo26x_depth_bayese_768x768_nv12.bin` |
| s100 | n | nv12 | `nash-e/yolo26n_depth_nashe_768x768_nv12.hbm` |
| s100 | s | nv12 | `nash-e/yolo26s_depth_nashe_768x768_nv12.hbm` |
| s100 | m | nv12 | `nash-e/yolo26m_depth_nashe_768x768_nv12.hbm` |
| s100 | l | lite | `nash-e/yolo26l_depth_lite_nashe_768x768.hbm` |
| s100 | x | lite | `nash-e/yolo26x_depth_lite_nashe_768x768.hbm` |
| s100p | n | nv12 | `nash-m/yolo26n_depth_nashm_768x768_nv12.hbm` |
| s100p | s | nv12 | `nash-m/yolo26s_depth_nashm_768x768_nv12.hbm` |
| s100p | m | nv12 | `nash-m/yolo26m_depth_nashm_768x768_nv12.hbm` |
| s100p | l | lite | `nash-m/yolo26l_depth_lite_nashm_768x768.hbm` |
| s100p | x | lite | `nash-m/yolo26x_depth_lite_nashm_768x768.hbm` |
| s600 | n | nv12 | `nash-p/yolo26n_depth_nashp_768x768_nv12.hbm` |
| s600 | s | nv12 | `nash-p/yolo26s_depth_nashp_768x768_nv12.hbm` |
| s600 | m | nv12 | `nash-p/yolo26m_depth_nashp_768x768_nv12.hbm` |
| s600 | l | lite | `nash-p/yolo26l_depth_lite_nashp_768x768.hbm` |
| s600 | x | lite | `nash-p/yolo26x_depth_lite_nashp_768x768.hbm` |

Authoritative filenames, URLs and hashes are in the [X5 manifest](../../../../platforms/x5/docs/release/models.yaml)
and [S manifest](../../../../platforms/s/docs/release/models.yaml). The list command
below prints exact asset IDs and URLs. A declaration of availability in a source
manifest is not a new download or inference verification.

<a id="preparation"></a>
## Explicit preparation

Run from repository root:

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --list-models
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/model/download.sh --target s100p --variant l
```

Each download prepares one artifact. Use the exact target and variant for another
entry, or supply its exact `--asset-id`. `download_model.sh` delegates to the same
canonical downloader; it accepts the same explicit flags, not historical
positional shell arguments. Downloads are separate from inference. No model was
downloaded as part of this host migration.

<a id="accompanying-files"></a>
## Accompanying files

Depth needs no label file. The sample includes only the byte-preserved source
`../test_data/bus.jpg`, not calibration or evaluation data. S lite l/x calibration
coefficients live in the binding module; they are part of the declared checkpoint
contract. Original weights, generated ONNX, BIN/HBM and SUNRGBD data are external.
See [conversion](../conversion/README.md) for their preparation and known gaps.

<a id="local-paths"></a>
## Paths and external copies

Default paths are resolved from this model directory regardless of Python's
current directory. X5 filenames are flat; S retains nash-e/m/p subdirectories.
Downloader `--output-dir` changes only the destination; it does not rewrite
runtime defaults. For example:

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n \
  --output-dir /work/models
python -m samples.vision.yolo26_depth.runtime.python.main --target x5 \
  --asset-id x5:yolo26_depth:yolo26n_depth_bayese_768x768_nv12.bin \
  --model-path /work/models/yolo26n_depth_bayese_768x768_nv12.bin \
  --output /work/depth/external-published
```

For a newly compiled model use explicit custom provenance:

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --target x5 \
  --asset-id x5:yolo26_depth:yolo26n_depth_bayese_768x768_nv12.bin \
  --model-path /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin \
  --converted-model --output /work/depth/custom-x5-n
```

Here the asset ID references the expected input/output contract, not official
byte identity. Reports set `asset_id=null`, `artifact_origin=user-converted` and
retain `contract_reference` plus the observed model digest. Board identity and
metadata checks still apply. Arbitrary shape/boundary changes require a new
binding; a filename cannot make an incompatible model compatible. For S lite,
the runtime coefficients must match the exported checkpoint calibration.

<a id="formats-checksums"></a>
## Formats, checksums and provenance

The five X5 manifest hashes are preserved:

| Variant | Publisher SHA-256 |
|---|---|
| n | `e55091eb594e20e37e6c36a36cce42a94ad80ec651ae893a2143cd2273ed9b0b` |
| s | `0e43958195f504d7a8ac48b1c99f4802cd9a4c3580321bfb251d0e0f892ccf4c` |
| m | `f4f2f1958dc16324932b4492490209c817cf7565c3c29240bcf4f0012f9c0be0` |
| l | `6a5fa40bda20ee56208ca6e594ecfd9781329385d0baf1b15c9eaa9625286d14` |
| x | `61798227fb7e0772a739b483ae5b5acd58a8e785dd7fd9aec5dcac7db0903d91` |

All fifteen S hashes are `null` (unknown). The downloader and runtime report an
observed digest, which identifies local bytes but cannot independently certify
publisher provenance when the expected hash is absent. Native launcher published
mode verifies the X5 hashes before building or executing. Explicit converted mode
never presents the published reference hash as a hash of generated bytes.

No checksum establishes model accuracy or board compatibility. Runtime metadata
is checked against the declared profile when loaded; real artifact metadata and
new board results remain unobserved in this host-only migration.
