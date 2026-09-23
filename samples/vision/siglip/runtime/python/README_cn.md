[English](./README.md) | 简体中文

# Python 运行 — SigLIP 视觉特征

<a id="environment"></a>
## 环境

- 运行目标：RDK S100（Nash-E）或 S100P（Nash-M），板端镜像需提供 `hbm_runtime`。板端镜像和固件版本未核验；主机仅使用注入 fixture 做契约测试。
- 主机准备：Python 3.14.7，以及 `../../requirements-host.txt` 中的 `numpy`、`opencv-python`、`PyYAML`。
- `hbm_runtime` 只存在于板端镜像。`--help`、`--list-models` 和显式 target 的 `--dry-run` 有意不导入 SDK、不加载模型。

<a id="usage"></a>
## 使用

先按 [`../../model/README_cn.md`](../../model/README_cn.md) 准备 HBM，再从仓库根目录在板端运行：

```bash
# cwd：仓库根目录；模型：samples/vision/siglip/model/ 下已准备的默认 HBM
python3 samples/vision/siglip/runtime/python/main.py
# 成功：退出码 0，打印 pooler_output 的 JSON 统计
```

下面的自定义命令选择 target、variant、patch 特征，并保存输出。文件按给定名称保存；NumPy 格式不会自动追加扩展名。

```bash
# cwd：仓库根目录；模型：已准备的 so400m-patch14-384 HBM；输入：内置 dog.jpg
python3 samples/vision/siglip/runtime/python/main.py \
  --target s100p --variant so400m-patch14-384 \
  --submodel last_hidden_state --output-file /tmp/siglip-features.npy
# 成功：退出码 0，打印 JSON 摘要，且 /tmp/siglip-features.npy 含完整原始数组
```

无需 SDK 或模型加载时，可使用 `--list-models`，或使用 `--dry-run --target s100`（或 `s100p`）。主机上 `--target auto` 的 dry-run 会因缺少显式 target 退出 2。

<a id="parameters"></a>
## 参数

`--list-models` 与 `--dry-run` 互斥。运行时错误、文件缺失、选择错误和不支持的 target 均退出 2。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；SigLIP 只支持 S100/S100P。 |
| `--variant` | str | `None` | 八个 `VARIANTS` 之一；省略时默认 `base-patch16-224`，除非 `--asset-id` 指向其他 variant。 |
| `--asset-id` | str | `None` | 精确限定 manifest 引用，例如 `s:siglip:s100/bpu-siglip-base-patch16-224.hbm`。 |
| `--model-path` | str | `None` | 本地 HBM 路径；必须与精确 `--asset-id` 一起使用。 |
| `--test-img` | str | `samples/vision/siglip/test_data/dog.jpg` | 输入 BGR 图片路径。默认值由 sample 位置生成绝对路径；用户显式传入的相对路径按当前工作目录解释。 |
| `--image-size` | int | `None` | 可选断言，必须等于所选 variant 的固定尺寸。 |
| `--submodel` | str | `pooler_output` | 选择打包子模型：`pooler_output` 或 `last_hidden_state`。 |
| `--priority` | int | `0` | runtime 优先级，限制在 0–255。 |
| `--bpu-cores` | int 列表 | `[0]` | 一个或多个非负 BPU 核心索引。 |
| `--output-file` | str | `None` | 可选 NumPy 格式完整原始特征路径；创建父目录，不追加扩展名。 |
| `--list-models` | flag | `false` | 不使用 SDK、板端检测或模型加载，列出八个唯一 manifest 制品。 |
| `--dry-run` | flag | `false` | 不使用 SDK 或模型加载，解析 target/variant/metadata 契约并打印 JSON；主机必须显式给出 S100/S100P target。 |

<a id="results"></a>
## 结果

CLI 打印 JSON 字段 `submodel`、`shape`、`dtype`、`mean`、`std`、`min`、`max`、`l2_norm`，它们是所选原始特征 tensor 的统计。传入 `--output-file` 时，完整 owned NumPy 数组写到给定路径。输出 dtype、shape 来自绑定 metadata：`pooler_output` 允许 `(1,D)` 或 `(1,1,D)`；`last_hidden_state` 为 `(1,N,D)`。不执行反量化、softmax、归一化、squeeze 或激活。

固定尺寸和特征维度来自 `VARIANTS`：

| Variant | 输入尺寸 | D | `last_hidden_state` 的 N |
| --- | ---: | ---: | ---: |
| `base-patch16-224` | 224 | 768 | 196 |
| `base-patch16-384` | 384 | 768 | 576 |
| `base-patch16-512` | 512 | 768 | 1024 |
| `large-patch16-256` | 256 | 1024 | 256 |
| `large-patch16-384` | 384 | 1024 | 576 |
| `so400m-patch14-224` | 224 | 1152 | 256 |
| `so400m-patch14-384` | 384 | 1152 | 729 |
| `so400m-patch16-256-i18n` | 256 | 1152 | 256 |

<a id="integration-example"></a>
## 集成示例

前置：按 [`model/README_cn.md`](../../model/README_cn.md) 将 `bpu-siglip-base-patch16-224.hbm` 放到默认路径；在 S100/S100P 板端运行。以下片段定义仓库路径、图片路径、target、variant、子模型、调度值、selection、runner、binding、tensor、原始输出，以及显式三步和组合调用。

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.siglip.runtime.python.model_binding import resolve_selection
from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.siglip.runtime.python.embedding import SigLIPTask

repo = Path.cwd()
image_path = repo / "samples/vision/siglip/test_data/dog.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "s100"
variant = "base-patch16-224"
submodel = "pooler_output"
priority = 0
bpu_cores = [0]
selection = resolve_selection(target, variant=variant, submodel=submodel)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
task = SigLIPTask(runner, binding)

prepared = task.pre_process(image)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(image)
assert np.array_equal(explicit_result, composed_result)
print({"shape": composed_result.shape, "dtype": str(composed_result.dtype)})
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process`：BGR `uint8` `H×W×3` → `PreparedInput`；`_input_0` 是 owned contiguous RGB `float32` `(1,3,size,size)`，范围 `[-1,1]`，`context` 保存原图/缩放尺寸及 `(top,bottom,left,right)` padding。
- `forward`：`{"_input_0": tensor}` → 所选打包子模型的原始 `{"_output_0": ndarray}`。`model_runner` 校验 metadata 和容器，但保留原生数值 dtype 和值。
- `post_process`：原始输出 → metadata 绑定 shape/dtype 的 owned ndarray；错误 shape/dtype 和 NaN/Inf 会报错。本视觉特征任务不消费几何 context。
- `predict(image)` 严格串联 pre-process → forward → post-process；不下载、保存、激活、归一化或评估结果。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `Model not found: ...; prepare it explicitly with model/download.sh.` | 解析路径没有 HBM。 | 在 `model/` 准备对应 variant，或同时传入精确 `--asset-id` 和 `--model-path`。 |
| `No published SigLIP support for x5/s600` | target 不在 S100/S100P 发布范围。 | 使用 S100 或 S100P 板卡/target。 |
| `Host dry-run requires --target s100 or --target s100p` | 主机 dry-run 无法从 `auto` 推断板卡。 | 显式传入 `--target`。 |
| `image-size must be ...` | 显式尺寸与 variant 固定几何尺寸不一致。 | 省略 `--image-size` 或使用 variant 对应尺寸。 |
| `SigLIP output shape/dtype differs from bound metadata.` | 制品 metadata 不匹配所选子模型契约。 | 检查精确 HBM 并选择匹配制品；不要 reshape 或 cast 输出。 |

## 许可

运行时代码遵循仓库 [LICENSE](../../../../../LICENSE) 的 Apache-2.0。
