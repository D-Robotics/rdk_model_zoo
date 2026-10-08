English | [简体中文](README_cn.md)

# Example inputs and demonstrations


This directory provides four input images and four example result images.
It contains no models, calibration dataset or golden tensors.

| Path | Purpose |
| --- | --- |
| `image1.jpg`–`image4.jpg` | Example inputs for single-image VLM questions; users may supply their own images |
| `results/image.jpg` | Source project demonstration image |
| `results/test1.jpg`, `test2.jpg`, `test3.jpg` | Example runtime session screenshots illustrating demo runs |

## Directory structure

```text
test_data/
├── results/  # Files for results
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

## Use

After [model preparation](../model/README.md) and [native build](../runtime/cpp/README.md#build), start from the repository root:

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 demo vlm --image_path ../../test_data/image1.jpg --prompt "Describe this image"
```

In interactive `main`, load the image first with `/image ../../test_data/image1.jpg`, then ask a question.
Relative paths resolve against the process working directory, not the image directory.
Generated text appears in the terminal. The `results/` directory contains example runtime screenshots.

## Interpretation

Use these images for single-image questions; generated wording depends on the prompt and model output. For dataset accuracy, prepare labeled examples and compare predictions with references using the [evaluator](../evaluator/README.md). The conversion tutorial describes the COCO calibration data.
