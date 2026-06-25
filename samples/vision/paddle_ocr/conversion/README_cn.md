[English](./README.md) | 简体中文

# PaddleOCR 模型转换说明

本目录提供 PaddleOCR（检测 + 识别）在 RDK S100 / S100P / S600 上的量化 YAML 配置和完整转换流程说明。配置针对 **PP-OCRv6**；切换 `march` 字段即可适配其他平台。

## 环境搭建

```bash
# Clone PaddleOCR and install
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3 -m pip install -e .

# Install Paddle2ONNX and ONNXRuntime
python3 -m pip install paddle2onnx
python3 -m pip install onnxruntime
```

## 导出 ONNX 模型（PP-OCRv6）

参考 PaddleOCR 官方文档获取 PP-OCRv6 检测和识别推理模型，然后用 `paddle2onnx` 转换：

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

## 数据集准备

使用 [ICDAR2019-LSVT 数据集](https://ai.baidu.com/broad/introduction?dataset=lsvt)：

- 45 万张中文街景图片
- 完整标注（bbox + 文本）：5 万张
- 弱标注（仅文本）：40 万张

下载地址：<https://ai.baidu.com/broad/download?dataset=lsvt>

生成检测和识别的校准数据（如需在板端获取真实数据统计，可在目标板上运行）：

```bash
python get_det_calibration_data.py
python get_rec_calbration_data.py
```

## 检测模型编译

配置文件：[`paddleocr_det_configs.yaml`](./paddleocr_det_configs.yaml)

关键字段：

```yaml
model_parameters:
  onnx_model: '../onnx/model_detv6.onnx'
  march: "nash-e"          # nash-e (S100) / nash-m (S100P) / nash-p (S600)
  output_model_file_prefix: 'PP-OCRv6_det_infer-deploy_640x640_nv12'
  # 注意：请勿设置 remove_node_type: "Dequantize" —— 保留末尾的 Dequantize 节点
  # 可使运行时输出 float32 概率图，直接用 prob > threshold 比较即可

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

## 识别模型编译

配置文件：[`paddleocr_rec_configs.yaml`](./paddleocr_rec_configs.yaml)

关键字段：

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

## 多平台编译

要生成 `archive.d-robotics.cc/.../rdk_<soc>/paddle_ocr/` 下发布的产物，请使用对应的 `march` 编译：

| 目标平台 | `march` | 输出前缀不变 |
|---------|---------|-------------|
| RDK S100 | `nash-e` | `PP-OCRv6_*-deploy_*` |
| RDK S100P | `nash-m` | `PP-OCRv6_*-deploy_*` |
| RDK S600 | `nash-p` | `PP-OCRv6_*-deploy_*` |

运行时示例期望的文件名为 `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` 和 `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm`。

## 性能参考

```bash
hrt_model_exec perf --model_file PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec perf --model_file PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

数据因 SOC 而异；在目标板上运行上述命令获取实际延迟 / FPS。

## OE 资源

模型转换请在 x86 Linux 主机上配合 RDK OpenExplore 环境完成。

- OE 资源入口：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

## License

本目录基于 [Apache 2.0 License](../../../../LICENSE) 授权。