# FasterNet model artifacts

Prepare the model artifact with the canonical downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `FasterNet_S_224x224_nv12.bin` | x5 | S | single | download (`x5:fasternet:FasterNet_S_224x224_nv12.bin`) |
| `FasterNet_T0_224x224_nv12.bin` | x5 | T0 | single | download (`x5:fasternet:FasterNet_T0_224x224_nv12.bin`) |
| `FasterNet_T1_224x224_nv12.bin` | x5 | T1 | single | download (`x5:fasternet:FasterNet_T1_224x224_nv12.bin`) |
| `FasterNet_T2_224x224_nv12.bin` | x5 | T2 | single | download (`x5:fasternet:FasterNet_T2_224x224_nv12.bin`) |

Each reference is an exact row of `docs/release/x5/models.yaml`; the
manifest is the authority for URL and format. X5 consumes one packed NV12
tensor, so pair `--model-path` with its exact manifest reference. Choose a target and variant listed in the sample support matrix, then prepare that artifact.

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target/variant — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/fasternet/model/download.sh x5 s
bash samples/vision/fasternet/model/download.sh x5 t2
```

The Python form is equivalent:
`python3 samples/vision/fasternet/model/download.py --target x5 --variant t1`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. It prints the computed digest after transfer; the manifest SHA-256 is `null (unknown)` for these rows. Run the downloader before inference to place the selected artifact under this directory.

<a id="accompanying-files"></a>
## Accompanying files

Use the checked-in one-label-per-line ImageNet class file at run time: `datasets/imagenet/imagenet_classes.names`.

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
| `FasterNet_S_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FasterNet_T0_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FasterNet_T1_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `FasterNet_T2_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest SHA-256 fields are `null (unknown)`. The downloader prints each artifact's computed digest after transfer; keep that digest with its artifact identity.
