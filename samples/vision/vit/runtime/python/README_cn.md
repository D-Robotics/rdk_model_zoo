# ViT Python 运行

<a id="overview"></a>
## Python 推理

[`main.py`](main.py) 解析参数，显式构造 `ViTClassifier`，调用 `predict` 并展示结果。
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

需要完整仓库。S100 推理需要板端提供的 `hbm_runtime`；使用包含匹配 SDK 和 Python 环境的 S100 板端镜像。磁盘需容纳仓库、所选 HBM 和输出。只有重新转换需要 OE。

```bash
# cwd: repository root
python3 -m venv .venv-vit
source .venv-vit/bin/activate
python3 -m pip install -r samples/vision/vit/requirements-host.txt
```

<a id="usage"></a>
## 使用

按根 Quick Start 准备默认模型后，S100 上零参数入口自动识别板卡。下列命令 cwd 均为仓库根；退出 0 且打印 Top-5 表示成功。

```bash
python3 samples/vision/vit/runtime/python/main.py
python3 samples/vision/vit/runtime/python/main.py --target s100 --variant int8 --test-img samples/vision/vit/test_data/airplane_0000.png --label-file samples/vision/vit/test_data/cifar10_classes.names --top-k 5
```

纯主机检查：

```bash
python3 samples/vision/vit/runtime/python/main.py --list-models
python3 samples/vision/vit/runtime/python/main.py --dry-run --target s100 --variant int16
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | auto/x5/s100/s100p/s600；仅 s100 有制品 |
| `--asset-id` | str | None | 准确 manifest 引用 |
| `--variant / --model-variant` | choice | None | int8/int16；解析后默认 int8 |
| `--model-path` | str | None | 外部路径需要 asset-id |
| `--test-img` | str | samples/vision/vit/test_data/airplane_0000.png | BGR 图像 |
| `--label-file` | str | samples/vision/vit/test_data/cifar10_classes.names | 字典或每行一个标签 |
| `--top-k / --topk` | int | 5 | 1..10 |
| `--resize-type` | choice | None | 0 nearest 直接缩放（发布模型默认）；1 linear letterbox |
| `--priority` | int | 0 | 0..255 |
| `--bpu-cores` | int list | [0] | 板端核心编号 |
| `--img-save-path` | str | None | 可选标注图片 |
| `--list-models` | flag | false | 不加载 SDK 或下载 |
| `--dry-run` | flag | false | 仅解析选择，不推理 |

图片/标签默认解析为当前仓库绝对路径。variant/resize 的 None 表示使用所选模型配置，并非无配置。list/dry-run 互斥。

<a id="results"></a>
## 结果

ClassificationResult 包含 class_ids（整数数组）、scores（softmax 概率）和 labels（元组）。ID 0–9 按随附 CIFAR 标签映射。CLI 打印结果，可选图片路径保存标注副本；默认不写文件。

<a id="integration-example"></a>
## 集成示例

板端示例，先准备 int8；导入不下载任何内容。

```python
# cwd: repository root on S100; prepare int8 first
from pathlib import Path
from samples.vision.vit.runtime.python.classify import ViTClassifier
from samples.vision.vit.runtime.python.cli import resolve_selection
from utils.py_utils.labels import load_labels

selection = resolve_selection("s100", variant="int8")
labels = load_labels(Path("samples/vision/vit/test_data/cifar10_classes.names"))
contract = selection.contract
model = ViTClassifier(
    selection.model_path, target=selection.target,
    input_size=(contract.input_height, contract.input_width),
    class_count=contract.class_count, top_k=5, labels=labels,
    resize_type=contract.resize_type,
    resize_interpolation=contract.resize_interpolation,
    score_policy=contract.output_score_policy,
    output_transform=contract.output_transform,
)
model.set_scheduling_params(priority=0, bpu_cores=[0])
result = model.predict("samples/vision/vit/test_data/airplane_0000.png")
print(result.class_ids, result.scores, result.labels)
```

`predict` 接受本地图像路径或 BGR `uint8` 数组。阶段也可显式驱动
（`preprocess`/`infer`/`postprocess`）。<a id="stage-io"></a>
## 三阶段 I/O

preprocess（pre_process）：图像路径或 BGR U8 H×W×3 → PreparedInput，含 Y U8 [1,224,224,1]、UV U8 [1,112,112,2] 和每次调用独立几何信息。直接缩放用 nearest，letterbox 用 linear、127填充。infer 只调用运行器并保留 raw 映射。postprocess 压缩 F32 十类向量，稳定 softmax 后取 Top-K。predict 组合三阶段。运行前核对张量形状与 dtype；量化输出需要 output_transform="dequant" 和 SDK 量化元数据。

<a id="troubleshooting"></a>
## 排错

HBM 缺失：显式准备模型。目标不匹配：在 S100 运行，不覆盖 S100P 身份。类别数/形状/dtype 错误：核对制品引用和 metadata，改名不能绕过检查。图片不可读或 Top-K 非法：修正输入与 1..10 参数。外部模型路径需准确 asset-id。
