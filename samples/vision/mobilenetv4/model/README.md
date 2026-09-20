# MobileNetV4 model artifacts

The model directory contains no checked-in binaries; artifacts are fetched
explicitly from the platform release manifests by the canonical downloader.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `MobileNetV4_conv_small_224x224_nv12.bin` | x5 | small | single | download (`x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin`) |
| `MobileNetV4_conv_medium_224x224_nv12.bin` | x5 | medium | single | download (`x5:mobilenetv4:MobileNetV4_conv_medium_224x224_nv12.bin`) |
| `s100/mobilenetv4_small_224x224_nv12.hbm` | s100 | small | single | download (`s:mobilenetv4:s100/mobilenetv4_small_224x224_nv12.hbm`) |
| `s600/mobilenetv4_small_224x224_nv12.hbm` | s600 | small | single | download (`s:mobilenetv4:s600/mobilenetv4_small_224x224_nv12.hbm`) |
| `s100/mobilenetv4_medium_256x256_nv12.hbm` | s100 | medium | single | download (`s:mobilenetv4:s100/mobilenetv4_medium_256x256_nv12.hbm`) |
| `s600/mobilenetv4_medium_256x256_nv12.hbm` | s600 | medium | single | download (`s:mobilenetv4:s600/mobilenetv4_medium_256x256_nv12.hbm`) |

Each reference is an exact row of `docs/release/x5/models.yaml` or
`docs/release/s/models.yaml`; the manifest is the authority for URL and
format. X5 consumes one packed NV12 tensor; S100/S600 consume separate Y
and UV tensors — a bare filename cannot select the protocol, so the runtime
always pairs `--model-path` with the exact reference. S100P has no asset
row and cannot be satisfied by reusing the S100 file.

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target/variant — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/mobilenetv4/model/download.sh x5 small
bash samples/vision/mobilenetv4/model/download.sh x5 medium
bash samples/vision/mobilenetv4/model/download.sh s100 medium
```

The Python form is equivalent:
`python3 samples/vision/mobilenetv4/model/download.py --target s100 --variant small`.
The downloader writes through a same-directory temporary file, checks the
content length and the recorded publisher SHA-256 when present, and installs
atomically without overwriting an existing file (a failed verification
preserves the file for investigation). The current rows carry no publisher
SHA-256, so the downloader prints the observed digest as local evidence and
states that origin is unproven. Downloading is explicit and never happens
during inference.

<a id="accompanying-files"></a>
## Accompanying files

The classifier additionally needs a one-label-per-line ImageNet class file
at run time: `datasets/imagenet/imagenet_classes.names` (shared by X5 and
S runs). It is checked into the repository; no download is required. The
label copies inside `test_data/` are source-branch leftovers; the canonical
path is the root `datasets/` one.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live under the sample's `model/` directory
(X5 flat; S-series under `model/s100/` and `model/s600/`), relative to the
sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `MobileNetV4_conv_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input, F32 `[1,1000,1,1]` logits output | null (unknown) |
| `MobileNetV4_conv_medium_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input, F32 `[1,1000,1,1]` logits output | null (unknown) |
| `s100/mobilenetv4_small_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s600/mobilenetv4_small_224x224_nv12.hbm` | nash `.hbm`, split Y/UV input (224x224), F32 `[1,1000]` logits output | null (unknown) |
| `s100/mobilenetv4_medium_256x256_nv12.hbm` | nash `.hbm`, split Y/UV input (256x256), F32 `[1,1000]` logits output | null (unknown) |
| `s600/mobilenetv4_medium_256x256_nv12.hbm` | nash `.hbm`, split Y/UV input (256x256), F32 `[1,1000]` logits output | null (unknown) |

The manifests record no publisher SHA-256 for these rows; unknown values
stay `null (unknown)` and are never copied across artifacts. The downloader
prints the observed digest on every download for local evidence.
