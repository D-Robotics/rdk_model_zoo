# FastViT model artifacts

The model directory contains no checked-in binaries; artifacts are fetched
explicitly from the platform release manifests by the canonical downloader.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `FastViT_S12_224x224_nv12.bin` | x5 | S | single | download (`x5:fastvit:FastViT_S12_224x224_nv12.bin`) |
| `FastViT_SA12_224x224_nv12.bin` | x5 | T0 | single | download (`x5:fastvit:FastViT_SA12_224x224_nv12.bin`) |
| `FastViT_T12_224x224_nv12.bin` | x5 | T1 | single | download (`x5:fastvit:FastViT_T12_224x224_nv12.bin`) |
| `FastViT_T8_224x224_nv12.bin` | x5 | T2 | single | download (`x5:fastvit:FastViT_T8_224x224_nv12.bin`) |

Each reference is an exact row of `docs/release/x5/models.yaml`; the
manifest is the authority for URL and format. X5 consumes one packed NV12
tensor, so the runtime always pairs `--model-path` with the exact
reference. No S-series artifact exists for this model: S100/S100P/S600
selection is an explicit no-published-asset error, and no X5 file may be
reused on an S board.

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target/variant — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/fastvit/model/download.sh x5 s12
bash samples/vision/fastvit/model/download.sh x5 t8
```

The Python form is equivalent:
`python3 samples/vision/fastvit/model/download.py --target x5 --variant t12`.
The downloader writes through a same-directory temporary file, checks the
content length and the recorded publisher SHA-256 when present, and installs
atomically without overwriting an existing file (a failed verification
preserves the file for investigation). The current rows carry no publisher
SHA-256, so the downloader prints the observed digest as local evidence and
states that origin is unproven. Downloading is explicit and never happens
during inference. The legacy source `download.sh` fetched both files with
`wget` and verified nothing; it is replaced by this manifest-backed path.

<a id="accompanying-files"></a>
## Accompanying files

The classifier additionally needs a one-label-per-line ImageNet class file
at run time: `datasets/imagenet/imagenet_classes.names`. It is checked into
the repository; no download is required.

<a id="local-paths"></a>
## Local paths

After preparation, artifacts live flat under the sample's `model/`
directory, relative to the sample root; the `--model-path` examples in
[runtime/python/README.md](../runtime/python/README.md) point at these
locations.

<a id="formats-checksums"></a>
## Formats and checksums

| File | Format | SHA-256 |
| --- | --- | --- |
| `FastViT_S12_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FastViT_SA12_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FastViT_T12_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FastViT_T8_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest records no publisher SHA-256 for these rows; unknown values
stay `null (unknown)` and are never copied across artifacts. The downloader
prints the observed digest on every download for local evidence.
