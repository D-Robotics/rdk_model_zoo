# ResNet18 Python 运行时

`main.py` 是 canonical 的用户命令。它从发布 Manifest 解析唯一模型引用，
校验检测到的板卡，懒加载 `hbm_runtime`，执行一次 `ClassificationTask`
流程。模型准备是显式操作；本运行时不下载、不安装任何包。

<a id="environment"></a>
## 环境

在目标板卡的 Python 环境运行，需要与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；读取 Manifest 需要 PyYAML。`hbm_runtime` 仅存在于板端镜像
且为懒加载——`--help`、`--list-models`、`--dry-run` 与主机 unittest 均不
需要它。主机测试依赖见 sample 的 `requirements-host.txt`。

<a id="usage"></a>
## 使用

cwd：仓库根目录。默认命令（无板卡与制品时零参数不可执行，最小 SDK-free
调用是列出示例模式）：

```bash
# 成功判据：打印全部已发布引用，退出码 0，不加载 SDK
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
```

在已准备制品的 X5 板上的完整运行：

```bash
# 前置：bash samples/vision/resnet/model/download.sh x5
# 成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names
```

S100/S600 替换为 `s:resnet18:<target>/...` 引用、`s100/`/`s600/` 制品路径
与 `platforms/s/...` 标签。`--dry-run --target x5` 在无板卡访问、无模型
加载、无下载的情况下解析选择。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件匹配 |
| `--asset-id` | string | null | Manifest 中完整的 `group:sample:filename` 引用 |
| `--variant` | choice | null | 模型变体（仅发布 `resnet18`） |
| `--model-path` | string | null | 已存在的 `.bin`/`.hbm`；必须与 `--asset-id` 配对；缺省时按所解析引用的 `model/` 位置查找 |
| `--test-img` | string | samples/vision/resnet/test_data/white_wolf.JPEG | BGR 输入图像 |
| `--label-file` | string | platforms/x5/datasets/imagenet/imagenet_classes.names | 逐行一个类别的 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的旧拼写 |
| `--resize-type` | int | null | `0` 直接拉伸或 `1` letterbox（BGR 127 填充）；默认跟随绑定源 |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核索引 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 无板卡访问列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型或 SDK，仅解析/检查选择 |

上表默认值由 Q3 检查器对照 `build_parser()` 机器核对。

<a id="results"></a>
## 结果

命令打印稳定的 Top-K：类别 ID、分数与标签
（`ClassificationResult(class_ids, scores, labels)`）；仅当给定
`--img-save-path` 时写标注图。X5 以 canonical 的扁平 1-D uint8 数组接收
packed NV12 缓冲（`H*W*3/2` 字节；224x224 即 75,264 字节，与旧
`(1,336,224,1)` 视图字节相同）；S100/S600 接收 Y `(1,224,224,1)` 与 UV
`(1,112,112,2)` uint8 数组。审计的两个源 wrapper 都对返回分数向量做
softmax；canonical 契约将其记录为 `unverified_score_vector` 上的
`legacy_softmax`，不宣称新的输出语义。输出形状按 rank 规则校验：凡可
squeeze 成 `(1000,)` 的单批次/单空间维拼写均可绑定（已发布制品声明
`raw_f32` 变换；量化制品需声明 `dequant` 契约）。

<a id="integration-example"></a>
## 集成示例

前置条件：制品已准备（见 [model/README_cn.md](../../model/README_cn.md)），
OpenCV-Python 可导入。示例内所有输入变量均有定义：

```python
import cv2

from samples.vision.resnet.runtime.python.classification import ClassificationTask
from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "x5",
    asset_id="x5:resnet:resnet18_224x224_nv12.bin",
    model_path="samples/vision/resnet/model/resnet18_224x224_nv12.bin",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/resnet/test_data/white_wolf.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

三阶段也可显式驱动：`prepared = task.pre_process(image)`、
`outputs = task.forward(prepared.tensors)`、
`result = task.post_process(outputs)` —— `predict` 恰好串联这三步（由
stage-contract 测试验证）。

<a id="stage-io"></a>
## 三阶段 I/O

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `pre_process` | 一张 BGR `uint8` 数组（任意尺寸） | `PreparedInput.tensors`（按 target 成形的 NV12 张量）+ `PreparedInput.transform`（本次调用冻结的缩放上下文） |
| `forward` | `prepared.tensors` | 原始 `{'prob': ndarray}`（X5，F32 `[1,1000,1,1]`）或 `{'output': ndarray}`（S，F32 `[1,1000]`）——与 runner 输出逐位一致，无解码 |
| `post_process` | 原始输出（无 context：分类不消费几何信息） | `ClassificationResult(class_ids, scores, labels)`，`legacy_softmax` 后稳定降序 Top-K |
| `predict` | BGR `uint8` 数组 | 串联三阶段，返回同一 `ClassificationResult` |

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 检查 |
| --- | --- |
| `Cannot identify this board` | dry-run 可用显式目标；真实执行只能在对应板卡上进行，显式目标本身不是硬件证据。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整限定引用，不要只传文件名。 |
| S100P 报 `No published ... asset` | Manifest 没有 ResNet18 S100P 行，只能在与板卡匹配的 S100/S600 制品上运行。 |
| 输入形状或类型不匹配 | 确认制品引用与运行时元数据，不要交叉使用 X5 packed 与 S split 制品。 |
| 输出与旧运行不同 | 先比较相同制品、图片、缩放模式、Top-K 与 raw output，再改变 score 语义。 |

主机检查（仓库根目录）：
`python3 -m unittest discover -s samples/vision/resnet/tests -v`。
