# MODNet model

<a id="artifacts"></a>
## Artifacts

| target | stage | asset ID | filename | availability |
|---|---|---|---|---|
| X5 | matting | `x5:modnet:modnet_512x512_rgb.bin` | `modnet_512x512_rgb.bin` | manual external asset |

No S100/S100P/S600 MODNet asset is published.

<a id="preparation"></a>
## Preparation

The active manifest has no public URL. Obtain the external model through the authorized model owner, verify its identity as `x5:modnet:modnet_512x512_rgb.bin`, and place it at `samples/vision/modnet/model/modnet_512x512_rgb.bin`. The repository helper is deliberately non-downloading:

```bash
# cwd: repository root
python3 -m samples.vision.modnet.model.download \
  --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin \
  --output-dir samples/vision/modnet/model
```

It prints the manual requirement and exits `2`; this is the expected behavior because no URL exists.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/person.jpg`: source input image.
- `../test_data/bg.jpg`: optional source background for composition.

<a id="local-paths"></a>
## Local paths

The default runtime path is `samples/vision/modnet/model/modnet_512x512_rgb.bin`. An external path must be paired with the exact `--asset-id x5:modnet:modnet_512x512_rgb.bin`; a filename alone does not establish protocol identity.

<a id="formats-checksums"></a>
## Format and checksums

The expected artifact is an X5 `bin` deployment model. The manifest records `url: null` and `sha256: null (unknown)`. No local observation is treated as publisher authentication.
