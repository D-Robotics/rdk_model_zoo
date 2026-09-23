English | [简体中文](./README_cn.md)

# Model Preparation — R3D-18

<a id="artifacts"></a>
## Artifacts

| Manifest asset | Target | Stage | Format | Source |
| --- | --- | --- | --- | --- |
| `s:3dresnet:s100/r3d_18.hbm` | `s100` | runtime inference | HBM | `docs/release/s/models.yaml` |

The sample has one published artifact. There are no x5, S100P, S600, ONNX, checkpoint, or calibration artifacts in this directory.

<a id="preparation"></a>
## Preparation

From the repository root, prepare the exact manifest row explicitly:

```bash
# cwd: repository root
bash samples/vision/3dresnet/model/download.sh s100
# expect: samples/vision/3dresnet/model/s100/r3d_18.hbm
```

The downloader resolves `s:3dresnet:s100/r3d_18.hbm` from the shared S manifest and uses the shared atomic download/verification helper. The manifest currently has no publisher SHA-256, so the observed digest is reported but cannot independently verify origin. Existing files are verified and are not silently overwritten. `download_model.sh` is a compatibility delegate to the same explicit command. Runtime execution never downloads a model automatically.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Required for | Purpose |
| --- | --- | --- |
| `../test_data/video0.npy` | runtime smoke test | Already normalized RGB float32 clip, shape `(1,3,16,112,112)` |
| `../test_data/kinetics_classnames.json` | CLI label display | 400-entry source `name → id` mapping, decoded into `id → name` with source quote removal |

The screenshot files under `../test_data/readme_img/` are documentation evidence only.

<a id="local-paths"></a>
## Local Paths

After preparation the model is expected at:

```text
samples/vision/3dresnet/model/s100/r3d_18.hbm
```

When `--model-path` is omitted, the runtime resolves this path from the exact selected manifest asset. An external model path is accepted only together with `--asset-id s:3dresnet:s100/r3d_18.hbm`.

<a id="formats-checksums"></a>
## Format and Checksums

| Asset | Format | SHA-256 |
| --- | --- | --- |
| `s100/r3d_18.hbm` | HBM | `sha256: null (unknown)` |

The `null` value is from the active manifest; no checksum is guessed or copied from another artifact.
