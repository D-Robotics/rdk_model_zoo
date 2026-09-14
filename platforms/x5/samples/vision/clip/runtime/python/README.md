# CLIP Python Sample

This sample demonstrates image-text similarity matching with a BPU image encoder and an ONNX text encoder.

## Directory Structure

```text
.
|-- bpe_simple_vocab_16e6.txt.gz
|-- clip_retrieval.py
|-- main.py
|-- README.md
|-- README_cn.md
|-- run.sh
`-- simple_tokenizer.py
```

## Parameter Description

| Argument | Description | Default |
| --- | --- | --- |
| `--image-model-path` | Path to the BPU image encoder `.bin` model. | `../../model/img_encoder.bin` |
| `--text-model-path` | Path to the ONNX text encoder model. | `../../model/text_encoder.onnx` |
| `--test-img` | Path to the test input image. | `../../test_data/dog.jpg` |
| `--texts` | Comma-separated text prompts for matching. | `a diagram,a dog` |
| `--img-save-path` | Path to save the output visualization image. | `../../test_data/inference.png` |
| `--priority` | Image encoder priority in the range `0~255`. | `0` |
| `--bpu-cores` | BPU core indexes used by the image encoder. | `0` |

## Quick Run

This sample requires `onnxruntime`, `ftfy`, and `regex` for the ONNX text
encoder and tokenizer. If the board image does not provide them, install them
before running:

```bash
python3 -m pip install --user onnxruntime ftfy regex
```

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

Prepare the default models before running `main.py` directly. You can either run
`./run.sh` once, or run `../../model/download.sh` from this directory to download
`../../model/img_encoder.bin` and `../../model/text_encoder.onnx`.

- Run with default parameters:

```bash
python3 main.py
```

- Run with explicit parameters:

```bash
python3 main.py \
    --image-model-path ../../model/img_encoder.bin \
    --text-model-path ../../model/text_encoder.onnx \
    --test-img ../../test_data/dog.jpg \
    --texts "a diagram,a dog" \
    --img-save-path ../../test_data/inference.png
```

## Interface Description

- **CLIPConfig**: Encapsulates image encoder, text encoder, and tokenizer parameters.
- **CLIPMatcher**: Implements image preprocessing, BPU image encoding, ONNX text encoding, and cosine similarity matching.
