English | [简体中文](README_cn.md)

# PointNet model files

<a id="artifacts"></a>
## Artifacts

| Asset ID | Target | File | Source |
| --- | --- | --- | --- |
| `s:pointnet:s100/pointnet.hbm` | s100 | `s100/pointnet.hbm` | public download |

The [release manifest](../../../../docs/release/s/models.yaml) provides
[this HBM URL](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointNet/pointnet.hbm).
No X5, S100P or S600 asset is published for this sample. It is a chair-only part
segmenter; the architecture's broader capabilities are not extra variants.

<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root
bash samples/vision/pointnet/model/download.sh --target s100
python3 samples/vision/pointnet/runtime/python/main.py --list-models
```

Download runs on host or board with network access and PyYAML. It writes a
partial file before atomic installation. Existing files are not overwritten;
without a publisher digest an existing file cannot be independently authenticated.
If download fails, retry the explicit command after restoring network access or
obtain the exact HBM from the publisher and place it at the default path. A file
extension or a renamed model cannot establish target compatibility.

<a id="accompanying-files"></a>
## Accompanying files

`../test_data/chair.pts` is the example XYZ input, not a calibration set or model
parameter. Labels are fixed in the CLI as back/seat/leg/arm, with IDs 0/1/2/3;
there is no external class dictionary, MVN file or configuration to download.

<a id="local-paths"></a>
## Local paths

Default: `samples/vision/pointnet/model/s100/pointnet.hbm`, resolved relative to
the sample location, independent of cwd. `--output-dir` on the downloader changes
the parent of `s100/pointnet.hbm`. To use an alternate exact artifact path:

```bash
# cwd: repository root; copy the published S100 HBM to /tmp/pointnet.hbm first
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --asset-id s:pointnet:s100/pointnet.hbm --model-path /tmp/pointnet.hbm
```

<a id="formats-checksums"></a>
## Format and checksum

Format: `.hbm`. `sha256: null (unknown)` in the publisher manifest. The downloader
prints an observed SHA-256 for tracking; that digest is not a publisher checksum.
Loading validates one model, float32 `(1,3,N)` input and `(1,N,4)` output. Output
dtype is verified from metadata; integer logits require valid SCALE parameters.
These checks do not replace the pending board run or certify arbitrary HBM files.
