# ResNet18 model artifacts

The model directory contains no checked-in binaries; artifacts are fetched
explicitly from the platform release manifests by the canonical downloader.

<a id="artifacts"></a>
## Artifacts

| File | Target | Stage | Source |
| --- | --- | --- | --- |
| `resnet18_224x224_nv12.bin` | x5 | single | download (`x5:resnet:resnet18_224x224_nv12.bin`) |
| `s100/resnet18_224x224_nv12.hbm` | s100 | single | download (`s:resnet18:s100/resnet18_224x224_nv12.hbm`) |
| `s600/resnet18_224x224_nv12.hbm` | s600 | single | download (`s:resnet18:s600/resnet18_224x224_nv12.hbm`) |

Each reference is an exact row of `platforms/x5/docs/release/models.yaml` or
`platforms/s/docs/release/models.yaml`; the manifest is the authority for
URL and format. X5 consumes one packed NV12 tensor; S100/S600 consume
separate Y and UV tensors — a bare filename cannot select the protocol, so
the runtime always pairs `--model-path` with the exact reference.
S100P has no ResNet18 row and cannot be satisfied by reusing the S100 file.

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/resnet/model/download.sh x5    # or s100 / s600
```

The Python form is equivalent:
`python3 samples/vision/resnet/model/download.py --target s100`. The
downloader writes through a same-directory temporary file, checks the
content length and the recorded publisher SHA-256 when present, and installs
atomically without overwriting an existing file (a failed verification
preserves the file for investigation). The current rows carry no publisher
SHA-256, so the downloader prints the observed digest as local evidence and
states that origin is unproven. Downloading is explicit and never happens
during inference.

<a id="accompanying-files"></a>
## Accompanying files

The classifier additionally needs a one-label-per-line ImageNet class file
at run time: `platforms/x5/datasets/imagenet/imagenet_classes.names` (X5
runs) or `platforms/s/datasets/imagenet/imagenet_classes.names` (S runs).
Both are checked into the repository; no download is required.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live at `model/resnet18_224x224_nv12.bin`
(x5), `model/s100/resnet18_224x224_nv12.hbm` (s100), and
`model/s600/resnet18_224x224_nv12.hbm` (s600) relative to the sample root;
the canonical `--model-path` default examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations. The C++ launcher defaults to `model/s100` as well.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `resnet18_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input, F32 `[1,1000,1,1]` output | null (unknown) |
| `s100/resnet18_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input, F32 `[1,1000]` output | null (unknown) |
| `s600/resnet18_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input, F32 `[1,1000]` output | null (unknown) |

The manifests record no publisher SHA-256 for these rows; unknown values
stay `null (unknown)` and are never copied across artifacts. The downloader
prints the observed digest on every download for local evidence.
