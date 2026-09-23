[English](./README.md) | 简体中文

# Python 运行 — CLIP 图文匹配

<a id="environment"></a>
## 环境

- 运行目标：RDK X5，图像 encoder 使用 `hbm_runtime`，文本 encoder 使用带 `CPUExecutionProvider` 的 `onnxruntime`。板端镜像和固件版本未核验。
- 主机检查：Python 3.14.7、NumPy、OpenCV、PyYAML、`ftfy==6.3.1`、`regex==2026.9.10`。主机测试注入两个 runtime，不需要 ONNX Runtime。
- 运行时制品：`PromptTokenizer` 加载 `bpe_simple_vocab_16e6.txt.gz`；图像和文本模型分开准备。保留源 BPE 清洗规则和 token ID。

<a id="usage"></a>
## 使用

按 [`../../model/README_cn.md`](../../model/README_cn.md) 准备两个制品，再从仓库根目录在 X5 上运行：

```bash
# cwd：仓库根目录；模型：已准备 img_encoder.bin、text_encoder.onnx
python3 samples/vision/clip/runtime/python/main.py --target x5
# 成功：退出码 0，打印 JSON scores/order 和标注后的 inference.png
```

选择图片、prompt 和可视化保存路径：

```bash
# cwd：仓库根目录；模型：samples/vision/clip/model/；BPE：内置词表
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-inference.png
# 成功：退出码 0，打印 JSON scores/order，并生成 /tmp/clip-inference.png
```

默认 `--img-save-path` 是受版本控制的 `test_data/inference.png`，默认运行会覆盖该文件。`run.sh` 只转发参数到 `main.py`，不会下载模型。

模型、图片和可视化输出的默认路径均由 sample 位置生成绝对路径；下表展示其仓库内位置。用户显式传入的相对路径按当前工作目录解释。

<a id="parameters"></a>
## 参数

`--list-models` 与 `--dry-run` 互斥。制品缺失、prompt 无效、target 不支持和 runtime 错误退出 2。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | `auto`、`x5`、`s100`、`s100p`、`s600`；只有 X5 有发布制品对。 |
| `--image-asset-id` | str | `None` | 精确图像 asset identity；与 `--image-model-path` 一起使用时必需。 |
| `--text-asset-id` | str | `None` | 精确文本 asset identity；与 `--text-model-path` 一起使用时必需。 |
| `--image-model-path` | str | `None` | 显式本地图像 `.bin` 路径；默认使用 manifest `img_encoder.bin`。 |
| `--text-model-path` | str | `None` | 显式本地文本 `.onnx` 路径；默认使用 manifest `text_encoder.onnx`。 |
| `--test-img` | str | `samples/vision/clip/test_data/dog.jpg` | 输入 BGR 图片；parser 从 sample 文件解析内置默认路径。 |
| `--texts` | str | `a diagram,a dog` | 逗号分隔 prompt；空项会移除。 |
| `--img-save-path` | str | `samples/vision/clip/test_data/inference.png` | 精确标注图片路径；默认会覆盖 source fixture。 |
| `--priority` | int | `0` | BPU 图像 encoder 优先级，限制 0–255。 |
| `--bpu-cores` | int 列表 | `[0]` | 非负 BPU 图像 encoder 核索引。 |
| `--list-models` | flag | `false` | 不加载两个 runtime，列出精确制品对。 |
| `--dry-run` | flag | `false` | 显式 `--target x5` 下解析制品对，不连接板卡或加载 SDK。 |

<a id="results"></a>
## 结果

CLI JSON 包含 `target`、`prompts`、`scores`、`order`、`image_saved`。`scores` 按 prompt 顺序排列，`order` 是降序索引数组。`MatchResult` 具有相同的 `scores`、`order` 字段。图像输出是按 `--img-save-path` 精确写出的 BGR PNG/JPEG。Cosine 使用 float32 特征、带 `1e-12` 稳定项的 L2 范数，不执行 softmax。

<a id="integration-example"></a>
## 集成示例

前置：准备 X5 制品对并在 X5 板端运行。源 BPE 词表从本地内置路径加载。示例定义全部路径、ID、输入、`CLIPTask(runner, binding, PromptTokenizer())`、`explicit_result`、`composed_result`，并比较 `MatchResult` 的每个字段。

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.clip.runtime.python.matching import CLIPTask
from samples.vision.clip.runtime.python.model_binding import resolve_selection
from samples.vision.clip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.clip.runtime.python.tokenization import PromptTokenizer

repo = Path.cwd()
image_path = repo / "samples/vision/clip/test_data/dog.jpg"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "x5"
image_asset_id = None
text_asset_id = None
image_model_path = None
text_model_path = None
texts = ["a diagram", "a dog"]
priority = 0
bpu_cores = [0]
selection = resolve_selection(
    target,
    image_asset_id=image_asset_id,
    text_asset_id=text_asset_id,
    image_model_path=image_model_path,
    text_model_path=text_model_path,
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
task = CLIPTask(runner, binding, PromptTokenizer())

prepared = task.pre_process(image, texts)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(image, texts)
np.testing.assert_array_equal(explicit_result.scores, composed_result.scores)
np.testing.assert_array_equal(explicit_result.order, composed_result.order)
print({"scores": composed_result.scores.tolist(),
       "order": composed_result.order.tolist()})
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process`：BGR `uint8` `H×W×3` 加非空文本序列 → `PreparedInput`。图像转 RGB，将短边固定为 224、长边按比例四舍五入后进行 bicubic resize，中心 crop 224，除以 255，生成 contiguous float32 `image` `(1,3,224,224)`。不使用 CLIP mean/std。真实 BPE 词表生成 contiguous int32 `texts` `(N,77)`，包含源 SOT/EOT ID。`context` 保存几何信息和文本。
- `forward`：语义 `image`/`texts` tensor → `image_feature` float32 `(1,512)`、`text_features` float32 `(N,512)` 原始 mapping。runner 将语义 key 适配到动态 image metadata 名称和 ONNX 文本名称；文本 metadata 必须为 I32 `[N,77]` 输入、F32 `[N,512]` 输出。
- `post_process`：原始特征 → `MatchResult(scores, order)`。计算 cosine similarity 和降序 `argsort`，不返回 softmax 或特征 L2 变换。
- `predict(image, texts)` 严格串联三阶段。词表读取与初始化、绘图和文件写入在 task 外部；pre_process 委托注入的 tokenizer 编码。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `Model not found: ...; prepare the pair with model/download.sh.` | 一个 encoder 制品缺失。 | 执行显式双制品下载，或为两个路径传入精确 asset ID。 |
| `No published CLIP encoder pair for s100` | manifest 只有 X5 制品对。 | 使用 X5 target。 |
| `External img_encoder.bin path requires its exact asset-id.` | 自定义图像路径没有身份。 | 添加 `--image-asset-id x5:clip:img_encoder.bin`。 |
| `External text_encoder.onnx path requires its exact asset-id.` | 自定义文本路径没有身份。 | 添加 `--text-asset-id x5:clip:text_encoder.onnx`。 |
| `Input text is too long for context length 77` | 源 BPE token 超过固定上下文且未启用截断。 | 缩短 prompt；CLI 保持源不截断行为。 |
| `CLIP text ... metadata` | ONNX 输入/输出名称、dtype 或宽度不兼容。 | 使用发布文本制品并检查动态 metadata；输入 I32 `[N,77]`，输出 F32 `[N,512]`。 |

## 许可

运行时代码遵循仓库 [LICENSE](../../../../../LICENSE) 的 Apache-2.0。源 BPE 词表和模型制品沿用其发布来源。
