# ConvNeXt Python 运行时

`main.py` 是面向用户的规范化入口。它从发布 Manifest 解析唯一的模型引用，
核对被检测的板卡，懒加载 `hbm_runtime`，并执行一个
`ClassificationTask` 流程。模型准备是显式动作；本运行时绝不下载模型或
安装软件包。

<a id="environment"></a>
## 环境

在目标板卡的 Python 环境中运行，需要与镜像匹配的 `hbm_runtime`、NumPy
和 OpenCV-Python；读取 Manifest 还需要 PyYAML。`hbm_runtime` 只存在于
板端镜像，采用懒加载——`--help`、`--list-models`、`--dry-run` 和主机
unittest 套件都不需要它。主机侧测试依赖见 sample 根的
`requirements-host.txt`。

<a id="usage"></a>
## 用法

cwd：仓库根目录。最小无 SDK 调用是清单模式：

```bash
# 成功判据：打印已发布引用，退出码 0，不加载 SDK
python3 samples/vision/convnext/runtime/python/main.py --list-models --target auto
```

在已准备好的 X5 板卡上的完整运行（缺省变体为 `atto`，唯一已发布变体，
保持源入口的默认模型）：

```bash
# 前置：bash samples/vision/convnext/model/download.sh x5
# 成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`--dry-run --target x5` 在不接触板卡、不加载模型、不下载的前提下解析
选择；本目录的 `run.sh` 原样转发参数给 `main.py`（源 `run.sh` 会优先
使用 `/opt/hobot/model/x5/basic/` 下的系统副本再回退——该隐式回退已
移除；制品来自上面的显式下载）。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件一致 |
| `--asset-id` | string | null | Manifest 中的完整 `group:sample:filename` 引用 |
| `--variant` | choice | null | 模型变体（缺省 `atto`；已发布组合见 `--list-models`） |
| `--model-path` | string | null | 已存在的 `.bin`；须与 `--asset-id` 配对；缺省为解析引用对应的 `model/` 位置 |
| `--test-img` | string | samples/vision/convnext/test_data/cheetah.JPEG | BGR 输入图像 |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | 每行一个 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的旧拼写（源入口的旗标） |
| `--resize-type` | int | null | `0` 直接拉伸或 `1` letterbox（BGR 127 填充）；缺省跟随绑定的源（1） |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核心索引 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 不接触板卡列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型与 SDK 解析/检查选择 |

以上默认值由 Q3 检查器对照 `build_parser()` 机器核对。

<a id="results"></a>
## 结果

命令以类别 ID、分数、标签打印稳定的 Top-K
（`ClassificationResult(class_ids, scores, labels)`），仅当给出
`--img-save-path` 时写标注图（源入口总会写 `test_data/result.jpg`）。
X5 侧以规范化的 flat 1-D uint8 数组（`H*W*3/2` 字节，224x224 → 75,264
字节）接收 packed NV12 缓冲。已发布制品返回 raw logits；任务在 Top-K 前
施加稳定 softmax。缺省 resize 为 letterbox（类型 1）加线性插值，与源
预处理一致。输出形状遵循 rank 规则：任何可 squeeze 到 `(1000,)` 的单例
批次/空间拼写均可绑定（已发布制品声明 `raw_f32` 变换；量化制品需要显式
`dequant` 契约）。

<a id="integration-example"></a>
## 集成示例

前提：制品已准备（见 [model/README_cn.md](../../model/README_cn.md)）且
OpenCV-Python 可导入。示例中每个输入变量都有定义：

```python
import cv2

from samples.vision.convnext.runtime.python.classification import ClassificationTask
from samples.vision.convnext.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.convnext.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "x5",
    asset_id="x5:convnext:ConvNeXt_atto_224x224_nv12.bin",
    model_path="samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/convnext/test_data/cheetah.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

三个阶段也可以显式驱动：`prepared = task.pre_process(image)`、
`outputs = task.forward(prepared.tensors)`、
`result = task.post_process(outputs)`——`predict` 恰好串联这些阶段（由
阶段契约测试验证）。

<a id="stage-io"></a>
## 阶段 I/O

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `pre_process` | 一张任意尺寸的 BGR `uint8` 数组 | `PreparedInput.tensors`（目标形状的 NV12 张量）+ `PreparedInput.transform`（每次调用冻结的 resize 上下文） |
| `forward` | `prepared.tensors` | 原始输出字典（X5 F32 `[1,1000,1,1]`）——与 runner 输出逐位一致，无解码 |
| `post_process` | 原始输出（无上下文：分类不消费几何） | `ClassificationResult(class_ids, scores, labels)`，声明分数策略下稳定降序 Top-K |
| `predict` | BGR `uint8` 数组 | 串联三阶段，同一 `ClassificationResult` |

<a id="troubleshooting"></a>
## 故障排查

| 症状 | 检查 |
| --- | --- |
| `Cannot identify this board` | 先用显式 target 做 dry-run，再只在该匹配板卡上执行；显式 target 不是硬件证据。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整限定引用；不要使用裸文件名。 |
| S 目标的 `No published ... asset` | S Manifest 没有 ConvNeXt 行；本 sample 是发布意义上的 X5-only，不是遗漏。 |
| 输入形状或 dtype 不匹配 | 核对制品引用与运行时 metadata；不得跨平台复用 X5 制品。 |
| 输出与 legacy 运行不一致 | 先在同一制品、图像、resize 模式、Top-K 和原始输出上对照，再考虑分数语义。 |

主机检查（仓库根目录）：
`python3 -m unittest discover -s samples/vision/convnext/tests -v`。
