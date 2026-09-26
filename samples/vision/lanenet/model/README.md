[English](README.md) | [简体中文](README_cn.md)

# LaneNet model assets

<a id="artifacts"></a>
## Published artifact

| Asset ID | Target | Local filename | Publisher SHA-256 |
| --- | --- | --- | --- |
| `s:lanenet:s100/lanenet256x512.hbm` | S100 / nash-e | `s100/lanenet256x512.hbm` | Unknown |

The authoritative URL and asset identity come from [the S release manifest](../../../../platforms/s/docs/release/models.yaml). This is the only published LaneNet artifact. S100P, S600 and X5 have no asset here; renaming or moving the S100 HBM does not add support. The original entry is retained under [the source model directory](../../../../platforms/s/samples/vision/lanenet/model).

<a id="preparation"></a>
## Explicit preparation

Run from the repository root:

```bash
bash samples/vision/lanenet/model/download.sh --target s100
```

The compatibility wrapper `download_model.sh` delegates to the same downloader. Downloads are never triggered by inference. To choose a different storage root:

```bash
python3 -m samples.vision.lanenet.model.download --target s100 --output-dir /data/lanenet-models
```

| Downloader option | Default | Meaning |
| --- | --- | --- |
| `--target` | `s100` | Only supported published target |
| `--asset-id` | `s:lanenet:s100/lanenet256x512.hbm` | Must match the exact manifest identity |
| `--output-dir` | This `model` directory | Asset-relative `s100/lanenet256x512.hbm` is appended |

<a id="accompanying-files"></a>
## Accompanying files

No class-name file is required. The runtime needs the HBM and an image, with the source image available at [test_data/lane.jpg](../test_data/lane.jpg). The output is not a list of named classes or lane instances. [Conversion](../conversion/README.md) retains the compiler YAML and the historical checkpoint URL; the checkpoint alone is not a deployable model, and the source export script is missing.

<a id="local-paths"></a>
## Local paths and identity

Default inference resolves this directory's `s100/lanenet256x512.hbm`. An external path requires an explicit contract identity:

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/lanenet-models/s100/lanenet256x512.hbm --dry-run
```

Dry-run does not open the HBM. Real loading checks board identity and actual model metadata. A matching asset ID is a caller declaration of the intended contract, not proof that arbitrary supplied bytes are the published model. Locally compiled artifacts also need full metadata and numerical validation; do not report their measured digest as a publisher checksum.

<a id="formats-checksums"></a>
## Format and checksums

The HBM contract has one float32 RGB NCHW `[1,3,256,512]` input with ImageNet normalization performed by the sample. Required outputs are an embedding `[1,3,256,512]` float32 tensor and a discrete binary int64 tensor `[1,1,256,512]` or `[1,256,512]`. Python binds `instance_seg_logits` and `binary_seg_pred`; native code binds unique shape/type roles. Additional observed outputs are retained without inventing a third name from source prose.

The downloader prints the observed SHA-256. Because the manifest has no publisher SHA-256, this can track byte identity across hosts but cannot independently authenticate the download. Runtime reports retain observed digests for later comparison. No model download or board loading was performed as part of the host migration validation.
