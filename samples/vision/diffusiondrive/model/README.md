[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive model assets

<a id="artifacts"></a>
## Published assets

| Target | March | Asset ID / local suffix |
| --- | --- | --- |
| S100P | `nash-m` | `s:diffusiondrive:s100p/diffusiondrive_r34_256x1024_s100p.hbm` |
| S600 | `nash-p` | `s:diffusiondrive:s600/diffusiondrive_r34_256x1024_s600.hbm` |

After `s:diffusiondrive:`, the remainder is the relative path under this directory. Camera resolution is 256×1024; the graph also consumes LiDAR, ego status and diffusion noise. There is no S100/X5 artifact or interchangeable S100P/S600 file. The [S release manifest](../../../../docs/release/s/models.yaml) controls URLs and checksums.

<a id="preparation"></a>
## Explicit preparation

From the repository root, download only the target you need:

```bash
bash samples/vision/diffusiondrive/model/download.sh --target s100p
bash samples/vision/diffusiondrive/model/download.sh --target s600
```

`download_model.sh` forwards the same flags for compatibility. Unlike the source wrapper, downloading requires an explicit target and is never triggered by inference. `CHIP`, `RDK_SOC` and `MODEL_PATH` environment overrides are not artifact selection contracts in the new entry. To use another storage root:

```bash
python3 -m samples.vision.diffusiondrive.model.download --target s600 --output-dir /data/diffusiondrive-models
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--target` | Required | `s100p` or `s600` |
| `--asset-id` | `null` | Inferred target asset; explicit value must match exactly |
| `--output-dir` | This sample's `model` directory | Target-relative filename is appended |

<a id="accompanying-files"></a>
## Accompanying files

[SHA256SUMS](SHA256SUMS) is preserved from the source. No class-name file is required. The [test-data directory](../test_data/README.md) supplies prepared four-input float NPZ archives and reference outputs, not a raw NAVSIM sensor loader. Keep input noise unchanged for comparisons. The [conversion directory](../conversion/README.md) retains both PTQ configs and explains missing export/calibration prerequisites.

<a id="local-paths"></a>
## Local paths and external copies

Defaults resolve to `model/s100p/diffusiondrive_r34_256x1024_s100p.hbm` or `model/s600/diffusiondrive_r34_256x1024_s600.hbm` under this sample. An external path requires exact asset identity:

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --asset-id s:diffusiondrive:s600/diffusiondrive_r34_256x1024_s600.hbm --model-path /data/diffusiondrive-models/s600/diffusiondrive_r34_256x1024_s600.hbm --dry-run
```

Dry-run resolves selection without opening the model. Actual loading checks physical board identity, the published file digest and runtime metadata. `auto` can derive a contract from an explicit asset ID; otherwise it requires recognized local identity and does not fall back to S600 on a development host. Custom converted files with different hashes require explicit new asset/binding integration, not a checksum override.

<a id="formats-checksums"></a>
## Format and integrity

| Target | Published SHA-256 |
| --- | --- |
| S100P | `5829a1a318116e2eba5c3ba313841fbd064fe2d48fe3cd37eb1ec7d9a0ffa8fa` |
| S600 | `78605b5aaa573fbaa2bf4788c6c8922bcab39d90c195e264aa3a8d3a78321c53` |

Downloads and real runtime loading verify these digests. The source records OpenExplorer v3.7.0, INT16-first/max PTQ, a GridSample INT8 exception and BPU-only placement. These historical conversion claims do not establish current board compatibility or physical tensor types; loading validates actual IO and quantization. Model binaries are not committed to Git. This host migration has not downloaded or loaded either HBM.
