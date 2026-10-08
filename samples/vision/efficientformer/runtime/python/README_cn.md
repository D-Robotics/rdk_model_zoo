[English](README.md) | 简体中文

# EfficientFormer Python 运行时

<a id="overview"></a>
## Python 推理

[`main.py`](main.py) 解析参数，显式构造 `EfficientFormerClassifier`，调用 `predict` 并展示结果。
[`classify.py`](classify.py) 包含模型初始化、前处理、推理和后处理；
[`cli.py`](cli.py) 集中管理命令行参数、发布模型选择和结果展示。
图片读取、标签校验和 SDK 会话复用 `utils/py_utils/`。

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

在目标板卡的 Python 环境中运行，需要与板卡匹配的 `hbm_runtime`、
NumPy 和 OpenCV-Python；读取 Manifest 需要 PyYAML。`hbm_runtime`
只存在于板端镜像中，且为懒加载——`--help`、`--list-models`、
`--dry-run` 和主机 unittest 套件都不需要它。主机侧测试依赖见 sample
的 `requirements-host.txt`。

<a id="usage"></a>
## 用法

cwd：仓库根目录。最小的不依赖 SDK 的调用是列表模式：

```bash
# 成功判据：打印两条发布引用，退出码 0，不加载 SDK
python3 samples/vision/efficientformer/runtime/python/main.py --list-models --target auto
```

在准备好的 X5 板卡上完整运行（缺省变体为 `l3`，保持源入口的默认
模型）：

```bash
# 前置：bash samples/vision/efficientformer/model/download.sh x5 l3
# 成功判据：退出码 0 并打印 Top-5 列表
python3 samples/vision/efficientformer/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientformer:EfficientFormer_l3_224x224_nv12.bin \
  --model-path samples/vision/efficientformer/model/EfficientFormer_l3_224x224_nv12.bin \
  --test-img samples/vision/efficientformer/test_data/bittern.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`l1` 换用自己的引用与路径。`--dry-run --target x5` 不接触板卡、不加载
模型、不下载即可解析选择；本目录的 `run.sh` 原样转发参数给 `main.py`。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件一致 |
| `--asset-id` | string | null | Manifest 中的完整 `group:sample:filename` 引用 |
| `--variant` | choice | null | 模型变体（省略时默认 `l3`；发布组合见 `--list-models`） |
| `--model-path` | string | null | 已存在的 `.bin`；必须与 `--asset-id` 配对；省略时默认取解析引用在 `model/` 下的位置 |
| `--test-img` | string | samples/vision/efficientformer/test_data/bittern.JPEG | BGR 输入图像 |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | 每行一个类别的 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的别名 |
| `--resize-type` | int | null | `0` 直接拉伸或 `1` letterbox（BGR 127 填充）；默认跟随所绑定的源（1） |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核编号 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 无需板卡列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型与 SDK 解析/检查一个选择 |

上表默认值即 [`cli.py`](cli.py) 中 `build_parser` 定义的值。

<a id="results"></a>
## 结果

命令打印稳定的 Top-K：类别 ID、分数与标签
（`ClassificationResult(class_ids, scores, labels)`）；仅当给出
`--img-save-path` 时才写标注图像。X5 收到 packed NV12 的规范化 flat
1-D uint8 数组，长度 `H*W*3/2` 字节（224x224 -> 75,264 字节）。发布制品
返回原始 logits；任务在 Top-K 前施加稳定 softmax。默认 resize 为
letterbox（type 1）加线性插值，与模型参考预处理一致。输出形状遵循 rank
规则：任何 squeeze 后为 `(1000,)` 的单批次/空间拼写都可绑定（发布制品
声明 `raw_f32` 变换；量化制品需要显式声明的 `dequant` 契约）。

<a id="integration-example"></a>
## 集成示例

前提：制品已准备（见 [model/README_cn.md](../../model/README_cn.md)）且
OpenCV-Python 可导入。示例中每个输入变量都有定义：

```python
from samples.vision.efficientformer.runtime.python.classify import EfficientFormerClassifier
from samples.vision.efficientformer.runtime.python.cli import resolve_selection

selection = resolve_selection(
    "x5",
    asset_id="x5:efficientformer:EfficientFormer_l1_224x224_nv12.bin",
    model_path="samples/vision/efficientformer/model/EfficientFormer_l1_224x224_nv12.bin",
)
contract = selection.contract
model = EfficientFormerClassifier(
    selection.model_path, target=selection.target,
    input_size=(contract.input_height, contract.input_width),
    class_count=contract.class_count, top_k=5,
    resize_type=contract.resize_type,
    resize_interpolation=contract.resize_interpolation,
    score_policy=contract.output_score_policy,
    output_transform=contract.output_transform,
)
result = model.predict("samples/vision/efficientformer/test_data/bittern.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` 接受本地图像路径或 BGR `uint8` NumPy 数组，不会原地修改
数组。三个阶段也可以显式驱动：`prepared = model.preprocess(source)`、
`outputs = model.infer(prepared)`、`result = model.postprocess(outputs)`
——`predict` 恰好串联这些步骤。

<a id="stage-io"></a>
## 阶段 I/O

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `preprocess`（`pre_process`） | 图像路径或一张任意尺寸的 BGR `uint8` 数组 | `PreparedInput.tensors`（按契约成形的 NV12 张量）+ `PreparedInput.transform`（每次调用冻结的 resize 上下文） |
| `infer`（`forward`） | `PreparedInput` | 原始输出 dict（X5 F32 `[1,1000,1,1]`）——与 runner 输出逐位一致，不做解码 |
| `postprocess`（`post_process`） | 原始输出（无 context：分类不消费几何） | `ClassificationResult(class_ids, scores, labels)`，按声明的分数策略稳定降序 Top-K |
| `predict` | 图像路径或 BGR `uint8` 数组 | 串联三阶段，返回同一 `ClassificationResult` |

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 检查 |
| --- | --- |
| `Cannot identify this board` | 按支持矩阵选择目标，并在匹配的板卡上运行推理。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整限定引用；不要使用裸文件名。 |
| 输入形状或 dtype 不匹配 | 核对制品引用与运行时 metadata；不得跨平台复用 X5 制品。 |
| 输出与旧实现不同 | 先固定同一制品、图像、resize 模式、Top-K 并对比原始输出，再考虑分数语义。 |
