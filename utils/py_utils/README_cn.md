[English](README.md) | 简体中文

# Python 推理公共函数


本目录为 Model Zoo Sample 提供板端 SDK 调用、图像与张量处理、标签读取及可视化函数。模型专有的预处理和后处理由各 Sample 实现。

## 目录结构

| 模块 | 职责 |
| --- | --- |
| `runtime.py` | 加载板端模型并调用 `hbm_runtime` |
| `model_runner.py`、`single_array_runner.py` | 分类及命名数组模型的输入输出绑定 |
| `platforms.py`、`platform_profile.py` | 板卡识别与平台能力 |
| `assets.py`、`cls_binding.py`、`sam_binding.py` | 模型清单与张量契约 |
| `image.py`、`file_io.py`、`preprocess.py`、`tensor_io.py` | 文件/图像读取、缩放及输入转换 |
| `labels.py`、`classification.py` | 标签加载与校验、分类结果处理 |
| `runtime_meta.py`、`quantization.py`、`nn_math.py`、`postprocess.py` | 元数据、反量化与输出处理 |
| `sam_runner.py`、`sam_stages.py`、`sam_tensor_io.py`、`sam_evaluator.py` | SAM 编码器/解码器运行与评估 |
| `yoloe26_decode.py`、`yoloe26_geometry.py` | YOLOE26 无提示分割解码与坐标还原 |
| `visualize.py`、`inspect.py` | 结果绘制与张量信息查看 |
| `text_metrics.py` | 文本编辑距离与字符错误率 |
| `tests/` | 单元测试 |

## 使用方式

从仓库根目录运行 Python，按对应 Sample 的 Runtime 文档安装依赖。

```python
from pathlib import Path
from utils.py_utils.image import read_bgr_image, bgr_to_nv12_planes
from utils.py_utils.labels import load_labels, validate_labels

image = read_bgr_image("samples/vision/resnet/test_data/zebra_cls.jpg")
labels = load_labels(Path("datasets/imagenet/imagenet_classes.names"))
validate_labels(labels, 1000)
```

`read_bgr_image` 返回 `(H,W,3)` 的 BGR `uint8` 数组。将图像缩放至模型要求的偶数宽高后，`bgr_to_nv12_planes` 返回连续的 `uint8` 数组：Y 为 `(1,H,W,1)`，UV 为 `(1,H/2,W/2,2)`。`validate_labels` 接收完整标签序列或索引到名称的映射。

### Runtime 调用

通过 `RuntimeSession(model_path, target=...)` 创建会话，在对应板卡调用 `load()`，然后将 SDK 的命名输入映射传给 `run(inputs)`。模型文件及依赖按 Sample 的 model 和 runtime 文档准备。

本地分类模型可通过 `RuntimeModelRunner.from_file(model_path, target=..., input_size=(height, width), class_count=...)` 创建运行器，调用 `load()` 后读取实际模型元数据。发布模型使用 `RuntimeModelRunner(selection, table=BINDING_TABLE)`，绑定表由 Sample 提供。

完整模型调用示例见 [ResNet](../../samples/vision/resnet/runtime/python/README_cn.md) 和 [YOLO](../../samples/vision/ultralytics_yolo/runtime/python/README_cn.md)。

### 模型选择与下载

`assets.py` 读取 `docs/release/{x5,s}/models.yaml`。模型引用由分组、Sample 和文件标识构成，例如 `x5:resnet:resnet18_224x224_nv12.bin`。使用对应 Sample 的下载命令准备模型。板卡标识与别名由 `docs/release/platforms.json` 定义。

### 任务公共函数

SAM 函数负责编码器特征与解码器提示的绑定。YOLOE26 函数接收 4585 类无提示协议的十个 NHWC float32 输出，并根据图像缩放信息还原框和掩码。浮点输出模型的准备见 [YOLOE 转换文档](../../samples/vision/yoloe/conversion/README_cn.md)。

`text_metrics.py` 根据已保存的转写文本计算 Unicode 字符错误率，保留空格、大小写和标点。输入格式见 [ASR 评估文档](../../samples/speech/asr/evaluator/README_cn.md)。

函数签名及张量说明见[源码参考文档](../../docs/source_reference/README.md)。
