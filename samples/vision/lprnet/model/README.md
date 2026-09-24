# LPRNet model

<a id="artifacts"></a>
## Artifacts

| target | stage | asset ID | filename | availability |
|---|---|---|---|---|
| X5 | recognition | `x5:lprnet:lpr.bin` | `lpr.bin` | published download |

The S100/S100P/S600 manifests contain no LPRNet asset. The bundled `test_data/test_input.dat` is an input artifact, not a model.

<a id="preparation"></a>
## Preparation

From the repository root, run the explicit downloader in an environment with network access:

```bash
python3 -m samples.vision.lprnet.model.download \
  --target x5 --asset-id x5:lprnet:lpr.bin \
  --output-dir samples/vision/lprnet/model
```

Success means `samples/vision/lprnet/model/lpr.bin` exists and the script prints `Prepared ...`. The documented module command was executed on the 2026-09-24 board runs (one X5 8GB and one X5 4GB, both rc=0; observed SHA-256 below). Runtime never downloads or overwrites the file.

<a id="accompanying-files"></a>
## Accompanying files

- `../test_data/test_input.dat`: source-provided float32 input tensor, 27072 bytes, reshaped to `1x3x24x94`.
- `../test_data/example.jpg`: source visual reference only; it is not passed through the runtime.

<a id="local-paths"></a>
## Local paths

The default runtime path is `samples/vision/lprnet/model/lpr.bin`. An external path is allowed only with the exact `--asset-id x5:lprnet:lpr.bin`; this preserves identity even though the publisher checksum is unknown.

<a id="formats-checksums"></a>
## Format and checksums

`lpr.bin` is an X5 `bin` deployment artifact. The active manifest URL is `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/LPRNet/lpr.bin`; its publisher field is `sha256: null (unknown)`. Both 2026-09-24 board runs observed SHA-256 `f480391507b6d15274bfff90900acc3271f4e46ac9afd4bf7d0fee4aa50f91bc` ([8GB recheck](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-binding-recheck/), [4GB run](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/)); a locally observed digest is evidence of the bytes used in that run, not publisher authentication.
