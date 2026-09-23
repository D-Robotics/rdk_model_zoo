[English](./README.md) | 简体中文

# Python 运行 — DINOv2 视觉特征

<a id="environment"></a>
## 环境

- 板端目标：RDK S100（`nash-e`）、S100P（`nash-m`）、S600（`nash-p`），板端镜像需提供 `hbm_runtime`。板端镜像和固件版本未核验。
- 主机契约检查：Python 3.14.7，以及 `../../requirements-host.txt` 中的 `numpy`、`opencv-python`、`PyYAML`。主机 `--help`、`--list-models` 和显式 target 的 `--dry-run` 不导入或加载 SDK。
- runtime 使用一个包含 `cls_feat`、`patch_feat` 的目标 HBM。未声明 runtime 实例和 SDK 线程安全。

<a id="usage"></a>
## 使用

按 [`../../model/README_cn.md`](../../model/README_cn.md) 准备制品，再从仓库根目录在目标板运行：

```bash
# cwd：仓库根目录；模型：samples/vision/dinov2/model/ 下已准备的目标 HBM
python3 samples/vision/dinov2/runtime/python/main.py
# 成功：退出码 0，打印 cls_feat 摘要和 bus.jpg 的 cosine similarity
```

选择 patch 特征并按精确路径保存完整返回 tensor：

```bash
# cwd：仓库根目录；模型：已准备 S100P 制品；输入：内置 dog.jpg、bus.jpg
python3 samples/vision/dinov2/runtime/python/main.py \
  --target s100p --output patch_feat \
  --output-file /tmp/dinov2-patch.npy
# 成功：退出码 0，JSON shape 为 [1,256,384]，生成 /tmp/dinov2-patch.npy；不追加扩展名
```

`run.sh` 接收历史位置输出参数（`cls_feat` 或 `patch_feat`），后面可接命名参数，不会下载。`--list-models` 和显式 target 的 `--dry-run` 不加载 SDK；主机使用 `--target auto` 做 dry-run 会退出 2。

<a id="parameters"></a>
## 参数

`--list-models` 与 `--dry-run` 互斥。选择、target、输入或制品错误均退出 2。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | `auto`、`s100`、`s100p`、`s600`；执行时 `auto` 只用于检测具体板卡。 |
| `--asset-id` | str | `None` | 精确限定 manifest 引用，例如 `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`。 |
| `--model-path` | str | `None` | 本地 HBM 路径；必须和精确 `--asset-id` 一起使用。 |
| `--test-img` | str | `samples/vision/dinov2/test_data/dog.jpg` | 第一张 BGR 图片；parser 从 sample 文件解析内置默认路径。 |
| `--second-img` | str | `samples/vision/dinov2/test_data/bus.jpg` | 用于 cosine similarity 的第二张 BGR 图片；文件缺失会报告，不会使运行失败。 |
| `--output` | str | `cls_feat` | 选择并摘要的特征：`cls_feat` 或 `patch_feat`。 |
| `--priority` | int | `0` | runtime 优先级，限制 0–255。 |
| `--bpu-cores` | int 列表 | `[0]` | 一个或多个非负 BPU 核索引。 |
| `--output-file` | str | `None` | 可选精确路径，保存返回 float32 NumPy 数组；创建父目录，不追加扩展名。 |
| `--list-models` | flag | `false` | 不加载模型，列出精确 manifest 制品。 |
| `--dry-run` | flag | `false` | 显式 target 下解析并打印输入/输出源契约，不加载 SDK。 |

<a id="results"></a>
## 结果

CLI 打印 JSON 字段 `output`、`shape`、`dtype`、`mean`、`std`、`min`、`max`、`l2_norm`。使用第二张图时还打印 `second_image`、`cosine_similarity`；文件缺失时状态为 `skipped_missing`。`cls_feat` 为 `(1,384)`，`patch_feat` 为 `(1,256,384)`，结果均为 owned float32。整数 HBM 输出依据 output quantization metadata 反量化；不执行 softmax、L2 归一化、patch pooling 或其他特征变换。

<a id="integration-example"></a>
## 集成示例

前置：按 [`../../model/README_cn.md`](../../model/README_cn.md) 准备 S100 制品，并在 S100 板端运行。使用其他板卡时将 `target = "s100"` 改为 `"s100p"` 或 `"s600"`，selection 会解析对应的独立 manifest HBM。示例定义全部路径、输入、target、asset identity、调度值、runner、binding、task、`explicit_result`、`composed_result`。`DINOv2Task.post_process` 返回所选 ndarray；将 `output="patch_feat"` 运行同一片段即可对照另一个 output key。

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
from samples.vision.dinov2.runtime.python.model_binding import resolve_selection
from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

repo = Path.cwd()
image_path = repo / "samples/vision/dinov2/test_data/dog.jpg"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "s100"
asset_id = None
model_path = None
priority = 0
bpu_cores = [0]
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
output = "cls_feat"
task = DINOv2Task(runner, binding, output)
prepared = task.pre_process(image)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(image)
np.testing.assert_array_equal(explicit_result, composed_result)
print({"output": output, "shape": composed_result.shape,
       "dtype": str(composed_result.dtype)})
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process`：BGR `uint8` `H×W×3` → `PreparedInput`；OpenCV 转 RGB，bicubic 将短边 resize 到 256，中心 crop 224，执行 `/255` 和 ImageNet mean/std，生成 owned contiguous float32 `{"input": (1,3,224,224)}`。`context` 保存原图/缩放尺寸和 crop 起点。
- `forward`：输入 mapping → 精确包含 `cls_feat` `(1,384)` 和 `patch_feat` `(1,256,384)` 的原始 mapping。runner 校验名称、shape 和原生 metadata dtype，原样返回。
- `post_process`：原始双输出 mapping → 所选 output 的 owned float32 ndarray。float32 输出保持 raw；整数输出只依据绑定 quantization metadata 反量化。不执行 softmax 或 L2 归一化。
- `predict(image)` 为所选 output 串联三个阶段；不会下载、写文件或评估。

<a id="troubleshooting"></a>
## 故障排查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `Model not found: ...; prepare it explicitly with model/download.sh.` | 缺少目标对应 HBM。 | 执行 `model/download.py --target ...`，或传入精确 asset ID 和模型路径。 |
| `No published DINOv2 support for x5` | target 超出三个 S 系列发布范围。 | 使用 `s100`、`s100p` 或 `s600`。 |
| `Host dry-run requires --target s100, s100p, or s600` | 主机无法为 dry-run 推断板卡。 | 显式传入 target。 |
| `An external model-path requires the exact manifest asset-id.` | 传入自定义路径但没有发布身份。 | 添加匹配的 `--asset-id`。 |
| `DINOv2 ... differs from bound metadata` | HBM I/O metadata 不符合固定契约。 | 使用准确的目标制品；不要 reshape 或 cast tensor。 |

## 许可

运行时代码遵循仓库 [LICENSE](../../../../../LICENSE) 的 Apache-2.0。
