[English](./README.md) | 简体中文

# PaddleOCR 模型转换

本目录提供 sample 使用的两组模型资产的维护版转换入口。请按目标板选择
完整的一组：

| 目标 | 模型族 | 工具链/march | 运行时检测输入 | 运行时识别输出 | 文件 |
| --- | --- | --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 英文 | `hb_mapper`、`bayes-e` | 一个打包 NV12 tensor | `[1,40,97,1]` F32，固定 96 字符表加 blank | `x5/ptq_yamls/*.yaml` |
| RDK S100 | PP-OCRv6 | `hb_compile`、`nash-e` | 分离的 `x_y`/`x_uv` NV12 tensor | `[1,40,18710]` F32，仓库 UTF-8 字典加 blank/空格 | `s100/*_configs.yaml` |

模型图、输出名、字典和输入协议都是目标相关的。不要把 X5 检测模型与 S100
识别模型混用，也不要互换字典。仓库没有已审计的 S100P 或 S600 PaddleOCR
制品行，因此本目录不宣称这些目标的转换支持。

## 转换环境

模型转换应在 x86 Linux 主机的对应 OpenExplorer 环境中完成，不要在 RDK 板端
运行。X5 原说明使用 OE v1.2.8，S 系列原说明使用 S OpenExplore 工具链。
`hb_mapper`、`hb_compile`、`hrt_model_exec` 由对应工具链环境提供。

S100 PP-OCRv6 的 ONNX 导出沿用原说明中的 Paddle2ONNX opset 19：

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python3 -m pip install -e .
python3 -m pip install paddle2onnx onnxruntime

paddle2onnx --model_dir ./inference/PP-OCRv6_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model_detv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

paddle2onnx --model_dir ./inference/PP-OCRv6_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model_recv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

将生成文件复制到 S100 YAML 预期的路径，或只修改对应的 `onnx_model` 字段：

```text
conversion/onnx/model_detv6.onnx
conversion/onnx/model_recv6.onnx
```

仓库没有 PP-OCRv3 exporter。X5 配方要求名为
`en_PP-OCRv3_det_infer.onnx` 与 `en_PP-OCRv3_rec_infer.onnx` 的上游/导出文
件；请从确定版本的 PP-OCRv3 发布物获取，并在运行 `hb_mapper` 前检查图输入。
原始审计没有建立仓库自有的 X5 导出命令，因此本 sample 不会把猜测命令标为
受支持的 X5 导出配方。

## 校准数据

下面的辅助脚本把 BGR 图片目录转换为已提交配方所需的输入 tensor。它不执行
模型转换，也不会联网：

```bash
cd samples/vision/paddle_ocr/conversion

# X5 hb_mapper：使用两个空目录，避免检测和识别 tensor 互相覆盖或混合。
# 编译前将检测 YAML 的 cal_data_dir 设为
# './calibration_data_detector'，将识别 YAML 的 cal_data_dir 设为
# './calibration_data_recognizer'。
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./x5/calibration_data_detector \
  --target x5 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops --output ./x5/calibration_data_recognizer \
  --target x5 --stage recognizer

# S100 hb_compile：以下路径对应已提交 YAML 中的
# ../calibration_data 和 ../calibration_data_rec_new/cropped_images_npy。
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./calibration_data \
  --target s100 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops \
  --output ./calibration_data_rec_new/cropped_images_npy \
  --target s100 --stage recognizer
```

辅助脚本要求每个输出目录为空，第二个阶段不能悄悄复用残留 tensor。其他目
标或阶段请使用新目录。检测 tensor 缩放到 640×640，在编译器
`data_mean_and_scale` 字段生效前保存 RGB float32；识别 tensor 缩放到 320×48，
保存与运行时 `no_preprocess` 相同的 `[0,1]` RGB float32。X5 输出扩展名是
`.rgbchw`，S100 输出扩展名是 `.npy`。请使用模型训练域中的代表性图片/裁剪
图；脚本不决定数据量，也不推导准确率。
检测 tensor 缩放到 640×640，先保存应用编译器 `data_mean_and_scale` 字段前
的 RGB float32 值；识别 tensor 缩放到 320×48，保存与运行时
`no_preprocess` 相同的 `[0,1]` RGB float32 值。X5 输出扩展名是 `.rgbchw`，
S100 输出扩展名是 `.npy`。请使用模型训练域中的代表性图片/裁剪图；脚本不
决定数据量，也不推导准确率。

## X5 PP-OCRv3（`hb_mapper`）

将两个 ONNX 文件和校准目录放到 `conversion/x5` 后，在 OE X5 v1.2.8 环境中
执行：

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_rec_infer.onnx

hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_rec_config.yaml
```

YAML 会生成：

```text
model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

复制到 X5 板端前检查生成制品的元数据：

```bash
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

检测 YAML 保留审计到的 RGB mean/scale、`nv12` 运行时输入、`bayes-e` 和 O3
编译级别。识别 YAML 保留 `featuremap` NCHW/no-preprocess 以及三个已观察到
的 `p2o.Softmax.*` int16 节点映射。

## S100 PP-OCRv6（`hb_compile`）

在 `conversion/s100` 目录的 S OpenExplore 环境执行：

```bash
hb_compile -c paddleocr_det_configs.yaml
hb_compile -c paddleocr_rec_configs.yaml
```

YAML 会生成：

```text
model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

检测配方保留末尾 `Dequantize`。已审计运行时把 `fetch_name_0` 当作 F32 概率
图直接阈值处理；删除该节点会改变观察到的输出契约。识别配方保留三个
`p2o.Softmax.*` 映射及 `set_all_nodes_int16`；输出类别数必须与
`test_data/s100/ppocrv6_dict.txt` 加 blank 和末尾空格类保持一致。编译后检
查制品：

```bash
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

提交的 S100 配方使用 `nash-e`。工具链文档中虽有其他 SoC 的 `nash-m`、
`nash-p` 名称，但仓库没有对应的已审计 manifest 制品或板测证据；修改
`march` 属于新的转换试验，不是本 sample 的已支持变体。

## 交给运行时前

将 model-info 中的 tensor 名、形状、类型和输出名与上级 README 的目标契约
逐项比较。只注册同时满足精确限定 manifest 引用和元数据的制品。Python 入口
通过 `--det-model-path`/`--rec-model-path` 与两条限定资产引用接收制品，不会
从文件名推断目标。
