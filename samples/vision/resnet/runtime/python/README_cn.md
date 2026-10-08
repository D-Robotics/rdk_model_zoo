# ResNet Python 运行时

[`main.py`](main.py) 提供命令行，构造分类器、调用 `predict()` 并展示结果。
[`classify.py`](classify.py) 包含 `ResNetClassifier` 的模型初始化和前处理、推理、后处理。
[`cli.py`](cli.py) 集中管理参数、发布模型选择和结果展示。
图片读取、标签校验及 Runtime 加载由 `utils/py_utils/` 提供。

<a id="overview"></a>
## Python 推理

本目录提供Python 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── classify.py  # 分类前处理、推理与后处理
├── cli.py  # 参数与结果展示
├── main.py  # 命令行入口
└── run.sh  # 运行示例
```

<a id="environment"></a>
## 环境

目标板卡需要匹配的 `hbm_runtime`、NumPy、OpenCV-Python 和 PyYAML。

| 板卡 | 发布模型 | 输入 |
| --- | --- | --- |
| RDK X5 | ResNet18 | packed NV12 |
| RDK S100 | ResNet18、ResNet50、ResNet152 | split NV12 |
| RDK S600 | ResNet18、ResNet50、ResNet152 | split NV12 |

模型准备见 [模型文件](../../model/README_cn.md)，ONNX 导出和编译见
[模型转换](../../conversion/README_cn.md)。

<a id="usage"></a>
## 使用

从仓库根目录执行。查看可用模型和目标配置：

```bash
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
python3 samples/vision/resnet/runtime/python/main.py --dry-run --target x5
```

列表和 dry-run 可在开发机运行，不加载板卡 SDK。准备模型后，在板卡上运行：

```bash
python3 samples/vision/resnet/runtime/python/main.py --target x5
python3 samples/vision/resnet/runtime/python/main.py --target s100 --variant resnet50
python3 samples/vision/resnet/runtime/python/main.py --target s600 --variant resnet152
```

`--target auto` 根据本机硬件选择板卡，默认模型是 ResNet18。
显式 `--target` 必须与实际板卡匹配。指定模型路径、图片和标签：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件匹配 |
| `--asset-id` | string | null | Manifest 中完整的 `group:sample:filename` 引用 |
| `--variant` | choice | null | 模型变体（`resnet18` 支持 x5/s100/s600；`resnet50`/`resnet152` 仅 s100/s600） |
| `--model-path` | string | null | 已存在的 `.bin`/`.hbm`；必须与 `--asset-id` 配对；缺省时按所解析引用的 `model/` 位置查找 |
| `--test-img` | string | samples/vision/resnet/test_data/white_wolf.JPEG | BGR 输入图像 |
| `--label-file` | string | null | 逐行一个类别的标签文件；默认使用内置 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的别名 |
| `--resize-type` | int | null | `0` 直接拉伸或 `1` letterbox（BGR 127 填充）；默认使用模型配置 |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核索引 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 无板卡访问列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型或 SDK，仅解析/检查选择 |

<a id="results"></a>
## 结果

命令输出 Top-K 类别 ID、标签和分数。添加 `--img-save-path result.jpg`
保存标注图片。库接口返回 `ClassificationResult(class_ids, scores, labels)`。

发布模型输入尺寸为 224×224，默认 letterbox，填充值为 BGR 127。
X5 输入是 75,264 字节的一维 uint8 NV12 数组；S100/S600 输入为
Y `(1,224,224,1)` 和 UV `(1,112,112,2)` 两个 uint8 数组。
发布模型输出为 F32 分数向量，经过 softmax 后按稳定降序取 Top-K。

<a id="integration-example"></a>
## 集成示例

在目标板卡上导入分类器，实例可重复使用：

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier(
    "samples/vision/resnet/model/resnet18_224x224_nv12.bin",
    target="x5", top_k=5,
)
result = model.predict("samples/vision/resnet/test_data/white_wolf.JPEG")
print(result.class_ids, result.scores, result.labels)
```

命令行用 `--target s100 --variant resnet50` 选择 S100 的 ResNet50；
库接口直接传对应的 `.hbm` 路径和 `target="s100"`。
`predict` 接受图片路径或 BGR uint8 NumPy 数组。标签可通过 `labels` 传入；
未传标签时 `result.labels` 是类别 ID 字符串。预测不修改输入数组。

<a id="custom-model"></a>
## 自训练模型

使用编译产物对应的板卡、输入尺寸和类别数构造分类器：

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier(
    "mymodels/my_resnet_4class.bin", target="x5",
    input_size=(224, 224), class_count=4, top_k=2,
    labels=["cat", "dog", "bus", "ship"],
)
result = model.predict("samples/vision/resnet/test_data/white_wolf.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`ResNetClassifier` 默认按 F32 输出执行 softmax。输出已经是概率时可设
`score_policy="none"`；量化输出设置对应的 `output_transform="dequant"`。
运行时依据声明校验实际张量的形状、类型和类别数。标签序列长度应与类别数一致。

<a id="stage-io"></a>
## 三阶段接口

| 方法 | 输入 | 输出 |
| --- | --- | --- |
| `preprocess` | 图片路径或 BGR uint8 数组 | `PreparedInput`：NV12 张量与本次图片缩放信息 |
| `infer` | `PreparedInput` | 原始输出张量字典 |
| `postprocess` | 原始输出张量字典 | Top-K `ClassificationResult` |
| `predict` | 图片路径或 BGR uint8 数组 | 顺序执行三阶段，返回 `ClassificationResult` |

`infer` 经共享 `RuntimeModelRunner` 调用 `hbm_runtime`；模型加载和板卡识别
由共享 `RuntimeSession` 完成。应用需要控制中间数据时可直接调用三个阶段。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 处理 |
| --- | --- |
| `Cannot identify this board` | 在支持的板卡上执行推理；开发机查看配置使用 `--dry-run --target x5`。 |
| `model file not found` | 按模型文件文档准备对应制品，或指定 `--asset-id` 和 `--model-path`。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整模型引用。 |
| S100P 提示没有发布模型 | 当前发布模型支持 X5、S100、S600；S100P 自训练模型使用 `ResNetClassifier` 指定契约。 |
| 输入或输出形状、类型不匹配 | 核对编译目标、输入协议、输出类别数与选择的模型配置。 |
