# RepVGG Python 运行

`main.py` 是面向用户的规范化入口：解析参数、构造模型、调用 `predict`、
展示结果。完整的分类流程在 [`classify.py`](classify.py) 中：
`RepVGGClassifier` 在一个可读文件里展示初始化、`preprocess`、`infer`、
`postprocess` 与 `predict`，复用共享的 NV12 打包、Top-K 数学与懒加载
runner。

<a id="environment"></a>
## 环境

X5 需要匹配的 `hbm_runtime`、NumPy、OpenCV 与 PyYAML。主机导入/help/list/dry-run 无需板端 SDK。主机测试依赖见 `samples/vision/repvgg/requirements-host.txt`。

[完整前提与已验证主机版本](../../README_cn.md#prerequisites)。板端镜像与 SDK 版本尚待实测登记。

<a id="usage"></a>
## 用法

所有命令 cwd 为仓库根。主机可执行：

```bash
python3 samples/vision/repvgg/runtime/python/main.py --list-models
python3 samples/vision/repvgg/runtime/python/main.py --dry-run --target x5
```

预期：6 个引用或默认 `a0` 契约，退出码 0。准备好的 X5 上执行：

```bash
# cwd: repository root
bash samples/vision/repvgg/model/download.sh x5 a0
python3 samples/vision/repvgg/runtime/python/main.py \
  --target x5 --variant a0 \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`run.sh` 原样转发运行参数；下载是单独动作。

模型按上方命令准备好后，X5 上也可直接运行默认入口（自动识别目标）：

```bash
# cwd: repository root
python3 samples/vision/repvgg/runtime/python/main.py
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | auto/x5/s100/s100p/s600；必须匹配实际板卡，发布支持范围见根支持矩阵 |
| `--asset-id` | string | null | 精确的发布清单制品引用 |
| `--variant` | choice | null | a0/a1/a2/b0/b1g2/b1g4；省略时选择 a0 |
| `--model-path` | string | null | 已有模型路径，需配合 asset-id；省略时由清单解析 |
| `--test-img` | string | samples/vision/repvgg/test_data/gooze.JPEG | BGR 图像路径 |
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

cwd：仓库根目录，先准备制品。API 不下载模型。本例省略可选标签，因此
标签值为类别 ID 字符串。

```python
from samples.vision.repvgg.runtime.python.classify import RepVGGClassifier
from samples.vision.repvgg.runtime.python.model_binding import resolve_selection

selection = resolve_selection("x5", variant="a0")
model = RepVGGClassifier(selection, top_k=5)
result = model.predict("samples/vision/repvgg/test_data/gooze.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` 接受本地图像路径或 BGR `uint8` 数组，且绝不原地修改数组。
既有的 `pre_process` / `forward` / `post_process` 拼写保持为薄别名，
共享的 `ClassificationTask` 流程仍可从
[`classification.py`](classification.py) 导入。

<a id="stage-io"></a>
## 阶段输入输出

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| preprocess（pre_process） | 图像路径或 BGR uint8 H×W×3 | PreparedInput：命名的扁平 NV12 uint8 张量（75264 字节）及不可变缩放上下文 |
| infer（forward） | PreparedInput | 原样返回 runner 原始输出，不做 softmax 或排序 |
| postprocess（post_process） | squeeze 后为 (1000,) 的 F32 分数 | softmax 与稳定 Top-K ClassificationResult |
| predict | 图像路径或 BGR 图像 | 串联上述三个阶段 |

默认前处理为线性 letterbox，填充值 BGR 127。分类后处理不消费几何上下文。runner 负责 SDK 加载、调度、metadata 检查；文件/标签读取与绘图留在 CLI 层（`cli.py`）。

<a id="troubleshooting"></a>
## 排障

缺模型：先显式准备。`model_path requires --asset-id`：补全精确引用。未知板卡或目标不匹配：主机用 `--dry-run --target x5`，真实推理只在匹配 X5 上执行。S 选择失败：没有发布制品。张量不匹配：留存实际 metadata 并核对制品身份，不能绕过绑定强行运行。
