# EdgeNeXt model artifacts

Prepare the model artifact with the canonical downloader, which resolves its URL and format from the platform release manifest.

<a id="artifacts"></a>
## Artifacts

| File | Target | Variant | Stage | Source |
| --- | --- | --- | --- | --- |
| `EdgeNeXt_base_224x224_nv12.bin` | x5 | base | single | download (`x5:edgenext:EdgeNeXt_base_224x224_nv12.bin`) |
| `EdgeNeXt_small_224x224_nv12.bin` | x5 | small | single | download (`x5:edgenext:EdgeNeXt_small_224x224_nv12.bin`) |
| `EdgeNeXt_x_small_224x224_nv12.bin` | x5 | x_small | single | download (`x5:edgenext:EdgeNeXt_x_small_224x224_nv12.bin`) |
| `EdgeNeXt_xx_small_224x224_nv12.bin` | x5 | xx_small | single | download (`x5:edgenext:EdgeNeXt_xx_small_224x224_nv12.bin`) |

Each reference is an exact row of `docs/release/x5/models.yaml`; the
manifest is the authority for URL and format. X5 consumes one packed NV12
tensor, so pair `--model-path` with its exact manifest reference. Choose a target and variant listed in the sample support matrix, then prepare that artifact.

<a id="preparation"></a>
## Preparation

From the repository root:

```bash
# input: manifest row for the target/variant — output: file under this directory
# success: exit 0, observed digest printed; no partial files left behind
bash samples/vision/edgenext/model/download.sh x5 base
bash samples/vision/edgenext/model/download.sh x5 xx_small
```

The Python form is equivalent:
`python3 samples/vision/edgenext/model/download.py --target x5 --variant small`.
The downloader checks content length and any manifest SHA-256, then installs the artifact atomically without overwriting an existing file. Run it before inference; it prints the computed digest after transfer. The manifest SHA-256 is `null (unknown)` for this row.

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
| `EdgeNeXt_base_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_x_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |
| `EdgeNeXt_xx_small_224x224_nv12.bin` | bayes-e `.bin`, packed NV12 input (224x224), F32 `[1,1000,1,1]` logits output | null (unknown) |

The manifest SHA-256 fields are `null (unknown)`. The downloader prints each artifact's computed digest after transfer; keep that digest with its artifact identity.
