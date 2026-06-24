English | [简体中文](./README_cn.md)

# PaddleOCR Model Conversion Guide

This directory provides the quantization YAML configurations and full conversion workflow notes for PaddleOCR (detection + recognition) on RDK S100 / S100P / S600. The shipped configs target **PP-OCRv6**; switch the `march` field to retarget another platform.

## Environment Setup

```bash
# Clone PaddleOCR and install
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3 -m pip install -e .

# Install Paddle2ONNX and ONNXRuntime
python3 -m pip install paddle2onnx
python3 -m pip install onnxruntime
```

## Export ONNX Models (PP-OCRv6)

Refer to the official PaddleOCR documentation to obtain the PP-OCRv6 detection and recognition inference models, then convert to ONNX with `paddle2onnx`:

```bash
# Detection model
paddle2onnx --model_dir ./inference/PP-OCRv6_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model_detv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

# Recognition model
paddle2onnx --model_dir ./inference/PP-OCRv6_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model_recv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

## Dataset Preparation

Use the [ICDAR2019-LSVT dataset](https://ai.baidu.com/broad/introduction?dataset=lsvt):

- 450,000 Chinese street view images
- Full annotation (bbox + text): 50,000 images
- Weak annotation (text only): 400,000 images

Download: <https://ai.baidu.com/broad/download?dataset=lsvt>

Generate calibration data for detection and recognition (run on the target board if you need true on-device statistics):

```bash
python get_det_calibration_data.py
python get_rec_calbration_data.py
```

## Detection Model Compilation

Configuration file: [`paddleocr_det_configs.yaml`](./paddleocr_det_configs.yaml)

Key fields:

```yaml
model_parameters:
  onnx_model: '../onnx/model_detv6.onnx'
  march: "nash-e"          # nash-e (S100) / nash-m (S100P) / nash-p (S600)
  output_model_file_prefix: 'PP-OCRv6_det_infer-deploy_640x640_nv12'
  # NOTE: do NOT set remove_node_type: "Dequantize" — keeping the trailing
  # Dequantize node makes the runtime output float32 probability maps so the
  # standard `prob > threshold` comparison works directly.

input_parameters:
  input_name: "x"
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: '1x3x640x640'
  norm_type: 'data_mean_and_scale'
  mean_value: 123.675 116.28 103.53
  scale_value: 0.01712475 0.017507 0.01742919
```

```bash
hb_compile -c paddleocr_det_configs.yaml
```

## Recognition Model Compilation

Configuration file: [`paddleocr_rec_configs.yaml`](./paddleocr_rec_configs.yaml)

Key fields:

```yaml
model_parameters:
  onnx_model: '../onnx/model_recv6.onnx'
  march: "nash-e"          # nash-e (S100) / nash-m (S100P) / nash-p (S600)
  output_model_file_prefix: 'PP-OCRv6_rec_infer-deploy_48x320_rgb'
  node_info:
    "p2o.Softmax.0": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }
    "p2o.Softmax.1": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }
    "p2o.Softmax.2": { 'ON': 'BPU', 'InputType': 'int16', 'OutputType': 'int16' }

input_parameters:
  input_type_rt: 'featuremap'
  input_layout_rt: 'NCHW'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_shape: '1x3x48x320'
  norm_type: 'no_preprocess'

calibration_parameters:
  optimization: "set_all_nodes_int16"
```

```bash
hb_compile -c paddleocr_rec_configs.yaml
```

## Multi-Platform Compilation

To generate the artifacts published under `archive.d-robotics.cc/.../rdk_<soc>/paddle_ocr/`, compile each YAML with the matching `march`:

| Target Platform | `march`   | Output prefix unchanged |
|-----------------|-----------|-------------------------|
| RDK S100        | `nash-e`  | `PP-OCRv6_*-deploy_*`   |
| RDK S100P       | `nash-m`  | `PP-OCRv6_*-deploy_*`   |
| RDK S600        | `nash-p`  | `PP-OCRv6_*-deploy_*`   |

The runtime samples expect filenames `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` and `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`.

## Performance Reference

```bash
hrt_model_exec perf --model_file PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec perf --model_file PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

Numbers vary by SOC; run the command above on the target board to obtain the per-platform latency / FPS.

## OE Resources

Run model conversion on an x86 Linux host with the matching RDK OpenExplore environment.

- OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
