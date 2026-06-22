English | [简体中文](./README_cn.md)

# PaddleOCR Model Conversion Guide

This directory provides the quantization YAML configurations and full conversion workflow notes for PaddleOCR (detection + recognition) on RDK S100.

## Environment Setup

```bash
# Clone PaddleOCR and install
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3 -m pip install -e .

# Install Paddle2ONNX and ONNXRuntime
python3 -m pip install paddle2onnx
python3 -m pip install onnxruntime
```

## Export ONNX Models (PP-OCRv3)

Download models:

```bash
# Detection model
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_det_infer.tar && cd ..

# Recognition model
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf ch_PP-OCRv3_rec_infer.tar && cd ..
```

Convert to ONNX:

```bash
# Detection model
paddle2onnx --model_dir ./inference/ch_PP-OCRv3_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

# Recognition model
paddle2onnx --model_dir ./inference/ch_PP-OCRv3_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

## Dataset Preparation

Use the [ICDAR2019-LSVT dataset](https://ai.baidu.com/broad/introduction?dataset=lsvt):

- 450,000 Chinese street view images
- Full annotation (bbox + text): 50,000 images
- Weak annotation (text only): 400,000 images

Download: <https://ai.baidu.com/broad/download?dataset=lsvt>

Generate calibration data for detection:

```bash
python get_det_calibration_data.py
```

Generate calibration data for recognition (run on S100):

```bash
python get_rec_calbration_data.py
```

## Detection Model Compilation

Configuration file: `paddleocr_det_configs100.yaml`

```yaml
model_parameters:
  onnx_model: './../PaddleOCR/inference/det_onnx/modelv3.onnx'
  march: "nash-e"
  working_dir: 'model_output'
  output_model_file_prefix: 'cn_PP-OCRv3_det_infer-deploy_640x640_nv12'
  remove_node_type: "Dequantize"

input_parameters:
  input_name: "x"
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: '1x3x640x640'
  norm_type: 'data_mean_and_scale'
  mean_value: 123.675 116.28 103.53
  scale_value: 0.01712475 0.017507 0.01742919

calibration_parameters:
  cal_data_dir: './../calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
```

```bash
hb_compile -c paddleocr_det_configs100.yaml
```

## Recognition Model Compilation

Configuration file: `paddleocr_rec_configs100.yaml`

```yaml
model_parameters:
  onnx_model: './../PaddleOCR/inference/rec_onnx/model_recv3.onnx'
  march: "nash-e"
  working_dir: 'model_output'
  output_model_file_prefix: 'cn_PP-OCRv3_rec_infer-deploy_48x320_rgb'
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
  cal_data_dir: './../calibration_data_rec'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: "set_all_nodes_int16"

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
```

```bash
hb_compile -c paddleocr_rec_configs100.yaml
```

## Performance Reference

### Detection Model

```bash
hrt_model_exec perf --model_file cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm
```

- Average Latency: 1.219 ms
- FPS: 798.575

### Recognition Model

```bash
hrt_model_exec perf --model_file cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm
```

- Average Latency: 2.588 ms
- FPS: 380.525

## Accuracy Reference

Cosine similarity images after quantization:

- Detection model: ![det cosine](../test_data/readme_img/image1.png)
- Recognition model: ![rec cosine](../test_data/readme_img/image2.png)

## OE Resources

Run model conversion on an x86 Linux host with the RDK S100 OpenExplore environment.

- OE resource entry point: <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
