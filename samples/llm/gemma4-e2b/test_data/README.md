# Example inputs and historical demonstrations

[简体中文](README_cn.md) | **English**

This directory preserves four input images and four historical result images from pinned S source `380e1a2`.
It contains no models, calibration dataset or golden tensors.

| Path | Purpose |
| --- | --- |
| `image1.jpg`–`image4.jpg` | Example inputs for single-image VLM questions; users may supply their own images |
| `results/image.jpg` | Source project demonstration image |
| `results/test1.jpg`, `test2.jpg`, `test3.jpg` | Historical source runtime screenshots used in the README |

## Use

After [model preparation](../model/README.md) and [native build](../runtime/cpp/README.md#build), start from the repository root:

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 demo vlm --image_path ../../test_data/image1.jpg --prompt "Describe this image"
```

In interactive `main`, use `/image ../../test_data/image1.jpg` followed by a question.
Relative paths resolve against the process working directory, not the image directory.
Generated text goes to the terminal; execution does not replace the historical screenshots in `results/`.

## Interpretation

These four images demonstrate functionality rather than dataset-level accuracy. Generated wording may differ;
matching screenshot text exactly is not the acceptance criterion. They are also not the COCO calibration set required by the conversion tutorial.
Board golden alignment needs separate internal data; see [evaluation](../evaluator/README.md).
This migration connected to no board and regenerated no screenshots.
