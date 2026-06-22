English | [简体中文](./README_cn.md)

# ViT Model Conversion

This directory provides the OpenExplore conversion configuration, calibration
data requirements, and HBM compilation steps for the CIFAR-10 ViT model.

## Files

| File | Description |
| --- | --- |
| `config_vit_nv12.yaml` | OpenExplore conversion configuration for the CIFAR-10 ViT NV12 HBM model. |
| `hb_compile.log` | Reference compilation log for the published configuration. |

## Source Model

The source model is a CIFAR-10 Vision Transformer exported from a PyTorch
implementation.

- Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- Project reference: <https://github.com/xiongqi123123/ViT_PyTorch.git>
- ONNX model name: `vit_cifar10_batch1.onnx`

The PyTorch training environment is used to train or fine-tune the CIFAR-10 ViT
model and export `vit_cifar10_batch1.onnx`. Place the exported ONNX file in this
directory before running OpenExplore compilation.

## Calibration Data

The configuration uses CIFAR-10 RGB calibration data. Prepare the calibration
images as `float32` NumPy files and place them in:

```text
./calibration_data_rgb
```

The released reference model uses PTQ with 50 calibration images. The runtime
input is NV12, while the training input is RGB in NCHW layout.

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore
environment. Model conversion is not intended to run on the board.

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

Install Docker by following the official Docker documentation, then verify the
installation:

```bash
sudo docker --version
sudo docker run --rm hello-world
```

Download the OpenExplore CPU Docker image for RDK S100/S100P from the OE
resource entry point, then load the actual image file:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

Start the container with the repository mounted and enough shared memory for
compilation:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

## Compile

Run the following command in this directory:

```bash
hb_compile --config config_vit_nv12.yaml
```

The expected output prefix is:

```text
vit_cifar10_batch1
```

The key configuration is:

```yaml
model_parameters:
  onnx_model: './vit_cifar10_batch1.onnx'
  march: "nash-e"
  layer_out_dump: False
  working_dir: './vit_cifar10_batch1'
  output_model_file_prefix: 'vit_cifar10_batch1'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_mean_and_scale'
  mean_value: 0.4914 0.4822 0.4465
  scale_value: 4.943153707865546 5.014042553191489 4.975124378109453
calibration_parameters:
  cal_data_dir: './calibration_data_rgb'
  cal_data_type: 'float32'
  quant_config: {"op_config": {"softmax": {"qtype": "int32"}}}
compiler_parameters:
  extra_params: {'input_no_padding': True, 'output_no_padding': True}
  compile_mode: 'latency'
  debug: False
  jobs: 8
  optimize_level: 'O2'
  advice: 1
```

## Runtime Interface

The compiled HBM model uses two NV12 input tensors:

- Y plane
- UV plane

Keep this input protocol unchanged when regenerating the model, because the
Python runtime prepares and feeds the two planes explicitly.

## Reference Accuracy

The CIFAR-10 reference accuracy for the published configuration is:

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | `74.54%` | `98.36%` |
| HBM | `72.62%` | `98.03%` |
