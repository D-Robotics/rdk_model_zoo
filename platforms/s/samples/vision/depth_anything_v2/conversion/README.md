English | [简体中文](./README_cn.md)

# Depth Anything V2 Model Conversion Guide

This directory records the ONNX structure, quantization notes, and OE toolchain entry points for the Depth Anything V2 monocular depth estimation model.

## ONNX Model

The ONNX model has one input and one output:

- Input: image tensor shaped `1x3x518x686`
- Output: predicted depth map shaped `1x518x686`

![Depth Anything V2 ONNX input output](../test_data/readme_img/image-3.png)
![Depth Anything V2 ONNX graph](../test_data/readme_img/image-1.png)

The model mainly contains common operators such as `Add`, `Conv`, `Mul`, and `MatMul`. Since the network contains Transformer attention modules, it also includes quantization-sensitive operators such as `Softmax`.

## Quantization Notes

The ONNX model is converted with the RDK S100 toolchain and quantized with int16. The quantization record shows that most operator accuracies are greater than 0.99, and the final quantization accuracy is about 0.999.

![Depth Anything V2 quantization](../test_data/readme_img/image-4.png)

## Model Artifact

After conversion, the `.hbm` heterogeneous model can run on RDK S100/S100P with BPU acceleration. The runtime uses the model file path below:

```text
samples/vision/depth_anything_v2/model/s100/depth_any.hbm
```

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
