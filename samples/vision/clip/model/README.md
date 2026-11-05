English | [简体中文](README_cn.md)

# Model Artifacts — CLIP encoder pair

<a id="artifacts"></a>
## Artifacts

The sample requires two independent assets: a BPU image encoder and a CPU ONNX text encoder. Supply each encoder’s corresponding asset ID and model path.

| Artifact | Format | Target | Stage(s) | Source |
| --- | --- | --- | --- | --- |
| `img_encoder.bin` | BIN | x5 | image encoder | download |
| `text_encoder.onnx` | ONNX | x5 | text encoder | download |

Exact manifest URLs:

| Asset ID | Release URL |
| --- | --- |
| `x5:clip:img_encoder.bin` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/img_encoder.bin> |
| `x5:clip:text_encoder.onnx` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/text_encoder.onnx> |

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
└── download.sh  # Model preparation command
```

<a id="preparation"></a>
## Preparation

Run from the repository root. The target is a named argument and only `x5` is accepted; no variant argument exists. Run the downloader explicitly — `runtime/python/run.sh` never triggers it implicitly.

```bash
# cwd: repository root; source: exact URLs above
python3 samples/vision/clip/model/download.py --target x5
# expect: samples/vision/clip/model/img_encoder.bin and text_encoder.onnx

# cwd: repository root; equivalent shell wrapper and explicit destination
bash samples/vision/clip/model/download.sh --target x5 --output-dir /tmp/clip-model
# expect: /tmp/clip-model/img_encoder.bin and /tmp/clip-model/text_encoder.onnx
```

The manifest records both SHA-256 values as unknown. The downloader prints observed digests; no published digest exists for comparison. I/O or selection errors exit 2.

<a id="accompanying-files"></a>
## Accompanying Files

| File | Role | Required |
| --- | --- | --- |
| `download.py` | Downloads both exact manifest assets. | Yes for scripted preparation; no when both files already exist. |
| `download.sh` | Named-argument helper for `download.py`. | No. |
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

Preparation code follows the repository [LICENSE](../../../../LICENSE), Apache-2.0. The model assets retain the publication's provenance and license metadata.

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

These independent profiles passed conversion, complete OE collection and actual HBM metadata binding. Source weights, graphs, per-target recipes, calibration counts and digests are in [converted_targets_20261103.json](../conversion/converted_targets_20261103.json).

| Target | Native variant | Components | Calibration photographs |
| --- | --- | --- | ---: |
| s100 | `vit-b32-native-int16-o1-20261103` | image, text | 200 |
| s100p | `vit-b32-native-int16-o1-20261103` | image, text | 200 |
| s600 | `vit-b32-native-int16-o1-20261103` | image, text | 200 |

```bash
python3 samples/vision/clip/model/download.py --target s100 --variant vit-b32-native-int16-o1-20261103
python3 samples/vision/clip/runtime/python/main.py --target s100 --variant vit-b32-native-int16-o1-20261103 --image-model-path /path/to/clip_vitb32_image_nashe_int16_20261103_o1.hbm --image-asset-id s:clip:s100/clip_vitb32_image_nashe_int16_20261103_o1.hbm --text-model-path /path/to/clip_vitb32_text_20261103.onnx --text-asset-id s:clip:s100/clip_vitb32_text_20261103.onnx --test-img samples/vision/clip/test_data/dog.jpg --texts 'a diagram,a dog'
```

Substitute the actual file paths. Multi-component tasks require every component together. Historical defaults and measurements retain their original identities. Board smoke, dataset accuracy, Runtime 1/2 and C++ 1/2 are `not-run` for these profiles.
