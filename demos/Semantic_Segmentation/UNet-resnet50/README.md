[English](./README.md) | 简体中文

# UNet-resnet50 Model Description

UNet-resnet50 is a semantic segmentation model based on the UNet architecture with a ResNet-50 backbone. It performs pixel-level classification to assign a class label to each pixel in the input image, making it suitable for scene understanding, medical imaging, and autonomous driving perception tasks.

## Algorithm Overview

UNet adopts a classic encoder-decoder architecture:

- **Encoder**: Uses convolution and pooling operations to progressively extract features and reduce spatial dimensions, capturing contextual information.
- **Decoder**: Restores spatial resolution through upsampling and convolution operations for precise localization.
- **Skip Connections**: Concatenates high-resolution features from the encoder with features in the decoder, preserving fine details and improving segmentation accuracy.
- **ResNet-50 Backbone**: Provides powerful feature extraction capabilities with residual connections, enabling deeper networks to train effectively.

For the original implementation and training details, please refer to:

- [Pytorch-UNet](https://github.com/bubbliiiing/unet-pytorch)

## Directory Structure

```bash
.
|-- conversion/          # Model conversion workflow (ONNX → HBM)
|-- model/               # Model files and download instructions
|-- runtime/             # Inference examples (Python)
|   -- python/
|-- evaluator/           # Model evaluation tools and guides
|-- test_data/           # Sample input images and results
|-- README.md            # This file (model overview)
|-- README_cn.md         # Chinese version of this document
```

## QuickStart

Pre-converted BPU models are provided. Ordinary users can skip the conversion step and run inference directly.

### Python

```bash
cd runtime/python
bash run.sh
```

The `run.sh` script will:
- Locate or prompt for the model file
- Run the inference with default test data

For more details, please refer to [runtime/python/README.md](./runtime/python/README.md).

## Model Conversion

Pre-converted models are already provided. If you need to convert from ONNX or retrain the model, please refer to the conversion documentation:

[conversion/README.md](./conversion/README.md)

## Runtime

Both Python inference examples are provided in this repository.

- **Python**: See [runtime/python/README.md](./runtime/python/README.md) for environment setup, parameter descriptions, and running instructions.

## Model Evaluation

For evaluation metrics (mIoU, Pixel Accuracy, etc.) and evaluation scripts, please refer to:

[evaluator/README.md](./evaluator/README.md)

## Inference Results

Below is an example of the UNet-resnet50 segmentation output on a sample image:

![](imgs/unet_result.jpg)

*Input image and segmentation visualization with class overlay.*

## Performance Data

### RDK X5 & RDK X5 Module

Semantic Segmentation (VOC)

| Model         | Size (Pixels) | Classes | Parameters | Throughput (1 thread) <br/> Throughput (Multi-thread) | Post-processing Time (Python) |
| ------------- | ------------- | ------- | ---------- | ---------------------------------------------------- | ----------------------------- |
| UNet-resnet50 | 512×512       | 20      | 43.93 M    | 11.23 FPS (1 thread) <br/> 13.23 FPS (2 threads) <br/> 13.23 (8 threads) | 267.08 ms |

### RDK X3 & RDK X3 Module

Semantic Segmentation (VOC)

| Model         | Size (Pixels) | Classes | Parameters | Throughput (1 thread) <br/> Throughput (Multi-thread) | Post-processing Time (Python) |
| ------------- | ------------- | ------- | ---------- | ---------------------------------------------------- | ----------------------------- |
| UNet-resnet50 | 512×512       | 20      | 43.93 M    | 2.61 FPS (1 thread) <br/> 5.17 FPS (2 threads) <br/> 5.21 FPS (4 threads) | 361.96 ms |

Test boards are all in optimal condition:

- **X5 optimal status**: CPU is 8 × A55@1.8G, all cores in Performance mode, BPU is 1 × Bayes-e@10TOPS.
- **X3 optimal status**: CPU is 4 × A53@1.8G, all cores in Performance mode, BPU is 2 × Bernoulli2@5TOPS.

Regarding post-processing: Currently, on the X5, the Python-based post-processing (including Softmax and Argmax) can be completed in approximately 3-5 ms using a single core and single thread. Post-processing does not become a bottleneck.

## License

This model sample follows the license of the ModelZoo repository. Please refer to the top-level LICENSE file for details.
