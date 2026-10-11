[English](README.md) | 简体中文

# MobileNetV2 Python 运行

<a id="overview"></a>
## Python 推理

[`main.py`](main.py) 解析参数，显式构造 `MobileNetV2Classifier`，调用 `predict` 并展示结果。
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

在目标板卡的 Python 环境运行，需要与板卡匹配的 `hbm_runtime`、NumPy、
OpenCV-Python 和 Pillow；读取 Manifest 需要 PyYAML。Pillow 用于已发布模型
所需的“短边缩放 + 中心裁剪”前处理中的抗锯齿双三次缩放（板卡镜像缺少时用
`python3 -m pip install pillow` 安装）。`hbm_runtime` 只存在于板卡镜像
中并被懒加载——`--help`、`--list-models`、`--dry-run` 与主机 unittest 套件
均不需要它。主机侧测试依赖见 sample 的 `requirements-host.txt`。

<a id="usage"></a>
## 用法

cwd：仓库根目录。默认命令（除模式标志外零参数，在没有板卡与制品时不可执行，
因此最小的免 SDK 调用是列清单模式）：

```bash
# 成功判据：打印全部发布引用，退出码 0，不加载 SDK
python3 samples/vision/mobilenetv2/runtime/python/main.py --list-models --target auto
```

在已准备制品的 X5 板卡上的完整运行：

```bash
# 前置：bash samples/vision/mobilenetv2/model/download.sh x5 100
# 成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv2/model/mobilenetv2_100_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100、S100P、S600 替换为对应的 `s:` 引用（例如
`s:mobilenetv2:s100p/mobilenetv2_100_nashm_224x224_nv12.hbm`）与
`s100/`、`s100p/` 或 `s600/` 制品路径；标签文件共用。`--dry-run --target x5` 在无板卡访问、无模型加载、无下载的情况下解析
选择。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--variant` | choice | null | 模型变体：`100`、`140`（省略时默认 `100`；发布组合见 `--list-models`） |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件一致 |
| `--asset-id` | string | null | 完整的 `group:sample:filename` Manifest 引用 |
| `--model-path` | string | null | 已存在的 `.bin`/`.hbm`；必须与 `--asset-id` 配对；省略时默认取解析引用对应的 `model/` 位置 |
| `--test-img` | string | samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG | BGR 输入图像 |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | 逐行一个类别的 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的别名 |
| `--resize-type` | int | null | `0` 直接缩放、`1` letterbox（BGR 127 填充）或 `2` 短边缩放 + 中心裁剪（需要 Pillow）；默认跟随绑定的模型（`2`） |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核编号 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 无板卡访问列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型或 SDK 地解析/检查选择 |

上面的默认值即 [`cli.py`](cli.py) 中 `build_parser` 定义的值。

<a id="results"></a>
## 结果

命令打印稳定的 Top-K（`ClassificationResult(class_ids, scores, labels)`），
仅当给出 `--img-save-path` 时写出标注图像。X5 收到 packed NV12 的 一维 uint8 缓冲（`H*W*3/2` 字节；224x224 即 75,264 字节）；S100/S100P/S600 收到 Y `(1,224,224,1)` 与 UV
`(1,112,112,2)` uint8 数组（所有变体在所有平台上都是 224x224）。
发布制品返回原始 logits；任务在 Top-K 前施加数值稳定的 softmax（两个平台一致）。
输出 shape 遵循 rank 规则：任何能 squeeze 到 `(1000,)` 的单例批次/空间拼写
均可绑定（发布制品声明 `raw_f32` 变换；量化制品需要显式 `dequant` 契约）。

<a id="integration-example"></a>
## 集成示例

前提：制品已准备（见 [model/README_cn.md](../../model/README_cn.md)），且
OpenCV-Python 与 Pillow 可导入。示例中每个输入变量都有定义：

```python
from samples.vision.mobilenetv2.runtime.python.classify import MobileNetV2Classifier
from samples.vision.mobilenetv2.runtime.python.cli import resolve_selection

selection = resolve_selection(
    "x5",
    asset_id="x5:mobilenetv2:mobilenetv2_100_bayese_224x224_nv12.bin",
    model_path="samples/vision/mobilenetv2/model/mobilenetv2_100_bayese_224x224_nv12.bin",
)
contract = selection.contract
model = MobileNetV2Classifier(
    selection.model_path, target=selection.target,
    input_size=(contract.input_height, contract.input_width),
    class_count=contract.class_count, top_k=5,
    resize_type=contract.resize_type,
    resize_interpolation=contract.resize_interpolation,
    resize_shorter=contract.resize_shorter,
    score_policy=contract.output_score_policy,
    output_transform=contract.output_transform,
)
result = model.predict("samples/vision/mobilenetv2/test_data/Scottish_deerhound.JPEG")
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
| `preprocess`（`pre_process`） | 图像路径或一张任意尺寸的 BGR `uint8` 图 | `PreparedInput.tensors`（目标形状的 NV12 张量）+ `PreparedInput.transform`（本次调用的冻结缩放上下文，包含裁剪偏移） |
| `infer`（`forward`） | `PreparedInput` | 原始输出字典（X5 F32 `[1,1000,1,1]`；S F32 `[1,1000]`），保留 SDK 原始张量 |
| `postprocess`（`post_process`） | 原始输出（分类不消耗几何上下文） | `ClassificationResult(class_ids, scores, labels)`，按声明的分数策略做稳定降序 Top-K |
| `predict` | 图像路径或 BGR `uint8` 图 | 串联三阶段，返回同一 `ClassificationResult` |

<a id="troubleshooting"></a>
## 故障排查

| 症状 | 检查 |
| --- | --- |
| `Pillow is required for resize_type 2` | 在板卡的 Python 环境安装 Pillow（`python3 -m pip install pillow`）。 |
| `Cannot identify this board` | 先用显式 target 做 dry-run，再只在匹配的板卡上执行；显式 target 不是硬件证据。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整引用；不要用裸文件名。 |
| 输入 shape 或 dtype 不匹配 | 核对制品引用与运行时 metadata；不要互换 X5 packed 与 S split 制品。 |
| 输出与旧实现不一致 | 先固定相同制品、图像、缩放方式、Top-K 比较 raw 输出，再考虑分数语义。 |
