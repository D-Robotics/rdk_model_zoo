English | [简体中文](./README_cn.md)

# Model Artifacts — CLIP X5 encoder pair

<a id="artifacts"></a>
## Artifacts

The sample requires two independent assets: a BPU image encoder and a CPU ONNX text encoder. Their asset IDs and paths are separate so an external path cannot silently replace a different published file.

| Artifact | Format | Target | Stage(s) | Source |
| --- | --- | --- | --- | --- |
| `img_encoder.bin` | BIN | x5 | image encoder | download |
| `text_encoder.onnx` | ONNX | x5 | text encoder | download |

Exact manifest URLs:

| Asset ID | Release URL |
| --- | --- |
| `x5:clip:img_encoder.bin` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/img_encoder.bin> |
| `x5:clip:text_encoder.onnx` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/text_encoder.onnx> |

<a id="preparation"></a>
## Preparation

Run from the repository root. The target is a named argument and only `x5` is accepted; no variant argument exists. The downloader does not run implicitly from `runtime/python/run.sh`.

```bash
# cwd: repository root; source: exact URLs above
python3 samples/vision/clip/model/download.py --target x5
# expect: samples/vision/clip/model/img_encoder.bin and text_encoder.onnx

# cwd: repository root; equivalent shell wrapper and explicit destination
bash samples/vision/clip/model/download.sh --target x5 --output-dir /tmp/clip-model
# expect: /tmp/clip-model/img_encoder.bin and /tmp/clip-model/text_encoder.onnx
```

The manifest records both SHA-256 values as unknown. The downloader prints observed digests and reports that they do not independently verify publisher origin. I/O or selection errors exit 2. This migration did not download either file.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `download.py` | Downloads both exact manifest assets. | Yes for scripted preparation; no when both files already exist. |
| `download.sh` | Named-argument wrapper for `download.py`. | No. |
| `../runtime/python/bpe_simple_vocab_16e6.txt.gz` | Source BPE vocabulary used by the text tokenizer. | Yes for text encoding. |

<a id="local-paths"></a>
## Local Paths

- Default image path: `samples/vision/clip/model/img_encoder.bin`.
- Default text path: `samples/vision/clip/model/text_encoder.onnx`.
- `--image-model-path` must pair with `--image-asset-id x5:clip:img_encoder.bin` when supplied.
- `--text-model-path` must pair with `--text-asset-id x5:clip:text_encoder.onnx` when supplied.

<a id="formats-checksums"></a>
## Formats & Checksums

| Artifact | Format | SHA-256 | Source of value |
| --- | --- | --- | --- |
| `img_encoder.bin` | BIN | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |
| `text_encoder.onnx` | ONNX | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |

## License

Preparation code follows the repository [LICENSE](../../../../LICENSE), Apache-2.0. The source model assets retain the publication's provenance and license metadata; no additional license is invented here.
