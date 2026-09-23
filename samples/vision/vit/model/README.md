# ViT models

<a id="artifacts"></a>
## Artifacts

| Variant | Target | Stage | File | Format |
| --- | --- | --- | --- | --- |
| int8 | s100 | classifier | `s100/vit_cifar10_batch1_int8.hbm` | hbm |
| int16 | s100 | classifier | `s100/vit_cifar10_batch1_int16.hbm` | hbm |

Authoritative source: [S manifest](../../../../docs/release/s/models.yaml), sample `vit`.

- `int8`: [download](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ViT/vit_cifar10_batch1_int8.hbm)
- `int16`: [download](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ViT/vit_cifar10_batch1_int16.hbm)


<a id="preparation"></a>
## Preparation

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
python3 samples/vision/vit/model/download.py --target s100 --variant int16
```

Defaults: target s100, variant int8. download.sh accepts positional target/variant; download.py uses named options. Network/filesystem failures return 2. For offline use transfer the exact artifact into the same path, retain its observed digest and source URL; do not use another target build.

<a id="accompanying-files"></a>
## Accompanying files

`../test_data/cifar10_classes.names` is an integer-keyed Python dictionary literal, parsed safely with literal_eval. Ten labels map IDs 0–9. `airplane_0000.png` is the default functional image. Neither is downloaded during inference.

<a id="local-paths"></a>
## Local paths

The runtime resolves models under this directory, independent of cwd. An external `--model-path` additionally requires its exact `--asset-id`; a matching filename alone is insufficient.

```bash
python3 samples/vision/vit/runtime/python/main.py --dry-run --target s100 --asset-id s:vit:s100/vit_cifar10_batch1_int16.hbm --model-path /path/to/vit_cifar10_batch1_int16.hbm
```

<a id="formats-checksums"></a>
## Formats and checksums

Both artifacts: `sha256: null (unknown)` in the manifest. The downloader prints an observed SHA-256; this records local bytes, not independent publisher verification. It preserves existing files under the shared downloader policy. Runtime rejects incompatible tensor metadata before inference.
