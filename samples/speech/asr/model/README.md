# ASR model preparation

English | [简体中文](README_cn.md)

<a id="artifacts"></a>
## Artifacts

| Target | Exact identity | Sample-relative local path |
| --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | `model/s100/asr.hbm` |
| S600 | `s:asr:s600/asr.hbm` | `model/s600/asr.hbm` |

Both URLs come from the [active S manifest](../../../../docs/release/s/models.yaml),
which has no publisher SHA-256 field. There is no X5/S100P artifact. A local
SHA-256 digest identifies the downloaded bytes.

<a id="preparation"></a>
## Explicit download

From the repository root, choose your actual target:

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/model/download.sh --target s600
```

Run only the command for the required target unless deliberately preparing both files. The downloader requires `--target`; optional `--asset-id` must match it. `--output-dir /path/to/models` stores `<target>/asr.hbm` below that directory. `PYTHON` selects the interpreter; failures return 2.

<a id="accompanying-files"></a>
## Vocabulary and audio

`test_data/vocab.json` contains 3503 token-to-ID entries, contiguous 0..3502, with blank `<pad>` at 0. The runtime verifies its exact source digest and builds the ordered token list. A differently ordered vocabulary can silently change text despite matching output width, so arbitrary replacement is rejected. See [input hashes](../test_data/README.md). The bundled WAV is demonstration data, not a calibration or labeled accuracy corpus.

<a id="local-paths"></a>
## Existing model

Example on S600 using an already prepared file:

```sh
bash samples/speech/asr/runtime/python/run.sh --target s600 \
  --asset-id s:asr:s600/asr.hbm --model-path /path/to/asr.hbm \
  --output-dir outputs/asr-s600-run1
```

The exact ID is required with a path override. The default path is sample-relative;
supplied relative paths are caller-relative. The runtime checks board identity
before constructing the SDK.

<a id="formats-checksums"></a>
## Runtime contract

A model must expose exactly one named model, one float32 `[1,30000]` input and one `[1,T,3503]` output with T positive. Tensor names and T come from real SDK metadata. Float32 logits remain raw; integer logits need valid SCALE metadata and shared dequantization before argmax. NaN/Inf or malformed tensors fail. The runtime does not add softmax because it cannot change finite float argmax.

Prepare an HBM deployment model with the tensor contract above. For ONNX/checkpoint conversion, prepare the exporter and calibration inputs listed in the [conversion guide](../conversion/README.md).

## Native model identity

The [native launcher](../runtime/cpp/README.md) resolves the same manifest
identities and checks the file before building or running. It passes the local
SHA-256 to the binary, which checks board, model and vocabulary identity before
SDK calls. The native interface takes unquantized FLOAT32 input and output
directly; integer SCALE outputs are supported by the Python runtime. Check the
model's SDK metadata on the target board before inference.
