[English](./README.md) | [简体中文](./README_cn.md)

# Model Conversion

This directory provides the conversion guide for the PaddleOCR sample, including PTQ configuration files, OpenExplorer build environment requirements, model compilation commands, and verification steps.

## Directory Structure

```text
conversion/
|-- README.md
|-- README_cn.md
`-- ptq_yamls/
    |-- paddleocr_det_config.yaml
    `-- paddleocr_rec_config.yaml
```

## Conversion Workflow

```text
PaddleOCR model -> ONNX export -> calibration data preparation -> hb_mapper checker -> hb_mapper makertbin -> .bin model
```

## Build Environment

Use the official OpenExplorer Docker image or an equivalent OE package environment for model checking and compilation.

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Or download the offline Docker image from the [D-Robotics Developer Community](https://forum.d-robotics.cc/t/topic/28035).

The conversion commands below should be executed inside the Docker container or OE compilation environment, not on the RDK board.

## Existing PTQ Configurations

| File | Model | Runtime Input | Output Model Prefix |
| --- | --- | --- | --- |
| `ptq_yamls/paddleocr_det_config.yaml` | OCR detection | `nv12` | `en_PP-OCRv3_det_infer-deploy_640x640_nv12` |
| `ptq_yamls/paddleocr_rec_config.yaml` | OCR recognition | `featuremap` | `en_PP-OCRv3_rec_infer-deploy_48x320_rgb` |

The YAML files are reference PTQ configurations for `hb_mapper makertbin`. Update `onnx_model`, calibration data paths, and output prefixes according to your local model files.

## Calibration Data

Prepare representative calibration data before compiling the model.

- Detection model: images should match the preprocessing protocol used by `paddleocr_det_config.yaml`.
- Recognition model: featuremap calibration data should match the recognition model input protocol in `paddleocr_rec_config.yaml`.
- Recommended sample count depends on the dataset and model stability; use representative training-domain samples.

## Model Check

Run `hb_mapper checker` before compilation to verify operator support.

```bash
hb_mapper checker --model-type onnx --march bayes-e --model en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e --model en_PP-OCRv3_rec_infer.onnx
```

## Model Compilation

Use the provided PTQ YAML files as the compilation entry.

```bash
hb_mapper makertbin --model-type onnx --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx --config ptq_yamls/paddleocr_rec_config.yaml
```

After compilation, the generated `.bin` files are written to the `working_dir` configured in the YAML files.

## Verification

Inspect model metadata:

```bash
hrt_model_exec model_info --model_file model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

Run a performance check:

```bash
hb_perf model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hb_perf model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

## Runtime Model Format

The runtime sample uses `.bin` models on `RDK X5`.

Prebuilt models can be downloaded from the [model](../model/README.md) directory. Recompile only when you need to regenerate models from your own ONNX files or calibration data.

## License

Tools and documents in this directory follow the repository top-level license.
