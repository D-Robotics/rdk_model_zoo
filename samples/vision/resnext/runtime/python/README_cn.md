# ResNeXt Python 运行

<a id="environment"></a>
## 环境

X5 需要匹配的 `hbm_runtime`、NumPy、OpenCV 与 PyYAML。主机导入/help/list/dry-run 无需板端 SDK。主机测试依赖见 `samples/vision/resnext/requirements-host.txt`。

[完整前提与已验证主机版本](../../README_cn.md#prerequisites)。板端镜像与 SDK 版本尚待实测登记。

<a id="usage"></a>
## 用法

所有命令 cwd 为仓库根。主机可执行：

```bash
python3 samples/vision/resnext/runtime/python/main.py --list-models
python3 samples/vision/resnext/runtime/python/main.py --dry-run --target x5
```

预期：1 个引用或默认 `50_32x4d` 契约，退出码 0。准备好的 X5 上执行：

```bash
# cwd: repository root
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py \
  --target x5 --variant 50_32x4d \
  --test-img samples/vision/resnext/test_data/bee_eater.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`run.sh` 原样转发运行参数；下载是单独动作。

模型按上方命令准备好后，X5 上也可直接运行默认入口（自动识别目标）：

```bash
# cwd: repository root
python3 samples/vision/resnext/runtime/python/main.py
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | auto/x5/s100/s100p/s600；必须匹配实际板卡，发布支持范围见根支持矩阵 |
| `--asset-id` | string | null | 精确的发布清单制品引用 |
| `--variant` | choice | null | 50_32x4d；省略时选择 50_32x4d |
| `--model-path` | string | null | 已有模型路径，需配合 asset-id；省略时由清单解析 |
| `--test-img` | string | samples/vision/resnext/test_data/bee_eater.JPEG | BGR 图像路径 |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | ImageNet 标签，每行一项 |
| `--top-k` | int | 5 | 输出结果数 |
| `--topk` | int | 5 | top-k 的旧拼写 |
| `--resize-type` | int | null | 0 直接缩放，1 线性 letterbox；省略沿用源默认 1 |
| `--priority` | int | 0 | 运行调度优先级 0–255 |
| `--bpu-cores` | int list | [0] | 运行使用的 BPU 核编号 |
| `--img-save-path` | string | null | 可选标注输出路径 |
| `--list-models` | flag | false | 只列出发布制品，不加载 SDK |
| `--dry-run` | flag | false | 仅解析契约，不加载模型或 SDK |

<a id="results"></a>
## 结果

打印 Top-K 类别 ID/分数/标签；`ClassificationResult` 包含 int64 ID、float32 分数和标签 tuple。未指定 `--img-save-path` 不写文件。源声明为 logits 加 softmax，实际板端 metadata 待复验。完全平局使用 ID 升序，可能不同于旧 NumPy 的平局排序。

<a id="integration-example"></a>
## 集成示例

cwd：X5 仓库根，先下载变体 50_32x4d。API 不下载模型。本例省略可选标签，因此标签值为类别 ID 字符串。

```python
import cv2
from samples.vision.resnext.runtime.python.classification import ClassificationTask
from samples.vision.resnext.runtime.python.model_binding import resolve_selection
from samples.vision.resnext.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection("x5", variant="50_32x4d")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/resnext/test_data/bee_eater.JPEG")
if image is None:
    raise FileNotFoundError("bee_eater.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

<a id="stage-io"></a>
## 阶段输入输出

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| pre_process | BGR uint8 H×W×3 | PreparedInput：命名的扁平 NV12 uint8 张量（75264 字节）及不可变缩放上下文 |
| forward | prepared.tensors | 原样返回 runner 原始输出，不做 softmax 或排序 |
| post_process | squeeze 后为 (1000,) 的 F32 分数 | softmax 与稳定 Top-K ClassificationResult |
| predict | BGR 图像 | 串联上述三个阶段 |

默认前处理为线性 letterbox，填充值 BGR 127。分类后处理不消费几何上下文。runner 负责 SDK 加载、调度、metadata 检查；文件/标签读取与绘图留在 main.py。

<a id="troubleshooting"></a>
## 排障

缺模型：先显式准备。`model_path requires --asset-id`：补全精确引用。未知板卡或目标不匹配：主机用 `--dry-run --target x5`，真实推理只在匹配 X5 上执行。S 选择失败：没有发布制品。张量不匹配：留存实际 metadata 并核对制品身份，不能绕过绑定强行运行。
