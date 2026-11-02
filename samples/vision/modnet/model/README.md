English | [简体中文](README_cn.md)

# MODNet model

<a id="artifacts"></a>
## Artifacts

| target | stage | asset ID | filename | availability |
|---|---|---|---|---|
| X5 | matting | `x5:modnet:modnet_512x512_rgb.bin` | `modnet_512x512_rgb.bin` | manual external asset |

No S100/S100P/S600 MODNet asset is published.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
├── download.sh  # Model preparation command
└── download_model.sh  # Shell command
```

<a id="preparation"></a>
## Preparation

The historical manual asset has no public URL. Obtain the external model through the authorized model owner, verify its identity as `x5:modnet:modnet_512x512_rgb.bin`, and place it at `samples/vision/modnet/model/modnet_512x512_rgb.bin`. The preparation command below reports where to place the model:

```bash
# cwd: repository root
python3 -m samples.vision.modnet.model.download \
  --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin \
  --output-dir samples/vision/modnet/model
```

It prints the manual requirement and exits `2`; this is the expected behavior because no URL exists.

### Public photographic artifact

The new profile uses `x5:modnet:modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin` and filename `modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`. After its publication is registered, download it explicitly below. URL and SHA-256 come from the platform manifest; existing files are verified and the legacy model is preserved.

```bash
python3 -m samples.vision.modnet.model.download --target x5 \
  --asset-id x5:modnet:modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin
```

The official checkpoint/model is Apache-2.0. See [conversion](../conversion/README.md) for the pinned source, checkpoint digest and preprocessing. Board performance and matting accuracy remain pending.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/person.jpg`: source input image.
- `../test_data/bg.jpg`: optional source background for composition.

<a id="local-paths"></a>
## Local paths

The default runtime path is `samples/vision/modnet/model/modnet_512x512_rgb.bin`. An external path takes the matching `--asset-id x5:modnet:modnet_512x512_rgb.bin`, which binds the protocol identity.

<a id="formats-checksums"></a>
## Format and checksums

The expected artifact is an X5 `bin` deployment model. The legacy manifest row records `url: null` and `sha256: null (unknown)`. No local observation is treated as publisher authentication.
