English | [简体中文](./README_cn.md)

# ConvNeXt Model Description

This directory describes the complete workflow of ConvNeXt in this Model Zoo, including: algorithm introduction, model conversion, runtime inference (Python), reusable pre/post-processing interfaces, and model evaluation steps.

---

## Algorithm Overview

ConvNeXt is a convolutional neural network that starts from the original ResNet and gradually improves the model by borrowing the design of Swin Transformer.

- **Paper**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- **Official Implementation**: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

### Algorithm Functionality

ConvNeXt can complete the following tasks:

- ImageNet 1000-class image classification
- Output Top-K classes and their confidence scores

### Algorithm Features

ConvNeXt uses larger convolution kernels (7x7), replaces ReLU with GELU activation functions, has fewer activation functions, uses LayerNorm instead of BatchNorm, and reduces the frequency of downsampling.

- **Large Kernel Convolution**: Uses large kernel (7x7) convolution instead of traditional 3x3 convolution to help expand the receptive field
- **Depthwise Separable Convolution**: Similar to MobileNet and EfficientNet, it significantly reduces parameters and computational costs
- **Normalization Layer**: Replaces traditional BatchNorm with LayerNorm, making it more suitable for small batch data
- **Simplified Residual Connections**: Simplifies the design of fully connected layers and removes the bottleneck structure in ResNet

![ConvNeXt Block](./test_data/ConvNeXt_Block.png)

---

## Directory Structure

This directory contains:

```bash
.
├── conversion                          # Model conversion process
│   ├── ConvNeXt_atto.yaml              # ConvNeXt atto PTQ configuration
│   ├── ConvNeXt_femto.yaml             # ConvNeXt femto PTQ configuration
│   ├── ConvNeXt_nano.yaml              # ConvNeXt nano PTQ configuration
│   ├── ConvNeXt_pico.yaml              # ConvNeXt pico PTQ configuration
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── evaluator                           # Model evaluation
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── model                               # Model files and download scripts
│   ├── download.sh                     # BIN model download script
│   ├── README.md                       # Documentation (English)
│   └── README_cn.md                    # Documentation (Chinese)
├── runtime                             # Inference samples
│   ├── cpp                             # C++ inference implementation (TODO)
│   └── python                          # Python inference sample
│       ├── main.py                     # Python entry script
│       ├── convnext.py                 # ConvNeXt wrapper implementation
│       ├── run.sh                      # One-click execution script
│       ├── README.md                   # Documentation (English)
│       └── README_cn.md                # Documentation (Chinese)
├── test_data                           # Inference results and sample data
│   ├── cheetah.JPEG                    # Sample input image
│   ├── ConvNeXt_Block.png              # Algorithm architecture diagram
│   └── inference.png                   # Sample inference result
└── README.md                           # ConvNeXt overview and quickstart
```

---

## QuickStart

For a quick experience, each model provides a `run.sh` script that allows you to run the corresponding model with one click.

### Python

- Go to the `python` directory under `runtime` and run the `run.sh` script:
    ```bash
    cd runtime/python/
    chmod +x run.sh
    ./run.sh
    ```
- For detailed usage of the `python` code, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Model Conversion

- ModelZoo provides pre-adapted BIN model files. Users can directly run the `download.sh` script in the `model` directory to download and use them. If you are not concerned about the model conversion process, **you can skip this section**.

- If you need to customize model conversion parameters or understand the complete conversion process, please refer to [conversion/README.md](./conversion/README.md).

---

## Runtime Inference

ConvNeXt model inference sample provides Python implementation.

### Python Version

- Provided in script form, suitable for rapid verification of model effects and algorithm flows
- The sample demonstrates the complete process of model loading, inference execution, post-processing, and result visualization
- For detailed usage, parameter descriptions, and interface specifications, please refer to [runtime/python/README.md](./runtime/python/README.md)

---

## Evaluator

`evaluator/` is used for model accuracy, performance, and numerical consistency evaluation. Please refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Performance Data

The following table shows the actual test performance data of ConvNeXt series models on the RDK X5 platform.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt_nano | 224x224 | 1000 | 15.59 | 77.37% | 71.75% | 5.71 | 200+ |
| ConvNeXt_pico | 224x224 | 1000 | 9.04 | 77.25% | 71.03% | 3.37 | 364+ |
| ConvNeXt_femto | 224x224 | 1000 | 5.22 | 73.75% | 72.25% | 2.46 | 556+ |
| ConvNeXt_atto | 224x224 | 1000 | 3.69 | 73.25% | 69.75% | 1.96 | 732+ |

**Notes:**
1. Test platform: RDK X5, CPU 8xA55@1.8G, BPU 1xBayes-e@1G (10TOPS INT8)
2. Latency data is for single-frame, single-thread, single-BPU-core ideal conditions
3. FPS data is for 4-thread concurrent scenarios, achieving 100% BPU utilization

![Inference Result](./test_data/inference.png)

---

## License

Follows the Model Zoo top-level License.
