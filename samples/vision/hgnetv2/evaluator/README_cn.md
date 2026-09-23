# HGNetV2 评测

<a id="dataset"></a>
## 数据集

自行准备 ImageNet-1k 验证集（通常 50,000 张），仓库不附带数据或下载。评测器递归扫描 JPEG/PNG，CSV 路径相对 --image-path，保留子目录。准备 UTF-8 CSV，表头 image:file,category，标签从 0 到 999。示例布局：

```text
/data/imagenet-val/n01440764/example.JPEG
/data/imagenet-labels.csv:
image:file,category
n01440764/example.JPEG,0
```

示例必须换成真实数据标签。CSV 缺字段、非法或冲突类别会报错，路径反斜杠转换为斜杠。CSV 中未出现的图片计为未匹配，不从目录名猜标签。

<a id="environment"></a>
## 环境

eval.py 在 X5 复用统一 HGNetV2 ClassificationTask 与 RuntimeModelRunner，依赖相同板端 SDK、NumPy、OpenCV、PyYAML，无额外推理框架。--help 及纯数据/指标测试不需 SDK。SciPy 仅用于源对照主机测试。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/hgnetv2/tests -v
```

板端功能检查（耗时未测）：

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

5 个变体与 X5 两种内存板位分别执行。使用下方保留的源任务 API 进行同进程对照。模型字节、图像、resize、Top-K 和调度须一致。原 CLI 仍可使用，但格式化分数不如这里保存的原始数组精确。

```bash
# cwd: repository root on X5, prepare variant b0 first
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/hgnetv2/runtime/python" python3 - <<'PYTHON'
import cv2
import numpy as np
from hgnetv2 import HGNetV2, HGNetV2Config
from samples.vision.hgnetv2.runtime.python.model_binding import resolve_selection
from samples.vision.hgnetv2.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.hgnetv2.runtime.python.classification import ClassificationTask

model_path = "samples/vision/hgnetv2/model/hgnetv2_b0_224x224_nv12.bin"
image = cv2.imread("samples/vision/hgnetv2/test_data/sandbar.JPEG")
if image is None:
    raise FileNotFoundError("sandbar.JPEG")
legacy = HGNetV2(HGNetV2Config(model_path, resize_type=1, topk=5))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_outputs = legacy.forward(legacy.pre_process(image))
old_ids, old_scores, _ = legacy.post_process(old_outputs)
selection = resolve_selection("x5", variant="b0")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5, resize_type=1)
new_outputs = task.forward(task.pre_process(image).tensors)
result = task.post_process(new_outputs)
# Preserve full vectors in a new output directory; do not overwrite old evidence.
from pathlib import Path
from datetime import datetime, timezone
out = Path("outputs") / ("hgnetv2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
out.mkdir(parents=True, exist_ok=False)
np.save(out / "legacy.npy", old_outputs[legacy.output_names[0]])
np.save(out / "unified.npy", new_outputs[binding.output_name])
print("legacy", old_ids.tolist(), old_scores.tolist())
print("unified", result.class_ids.tolist(), result.scores.tolist())
print("raw outputs:", out)
np.testing.assert_array_equal(result.class_ids, old_ids)
np.testing.assert_allclose(result.scores, old_scores, rtol=0, atol=1e-5)
PYTHON
```

这是尚未执行的板端对照配方，不是验证收据。遇到平局导致断言失败时核对逐 ID 分数，不放宽容差。数组之外还需记录“输出”节列出的身份信息。

数据集评测（cwd：X5 仓库根）。先下载 b0，将数据路径改为实际数据集。耗时取决于图片数，本轮未测。成功返回 0 并写 JSON，出现图片失败或没有成功推理时返回 2；未标注图片可产生退出码 0 的 partial 报告，须检查覆盖率字段。

```bash
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/evaluator/eval.py \
  --target x5 --variant b0 \
  --image-path /data/imagenet-val --val-csv /data/imagenet-labels.csv \
  --resize-type 0 --top-k 5 --limit 0 \
  --json-save-path outputs/hgnetv2-b0-evaluation.json
```

| Option | Default | 含义 |
| --- | --- | --- |
| `--target` | auto | 执行目标，仅匹配 X5 才能运行 |
| `--variant` | null | 默认 b0，其余显式指定 |
| `--asset-id` | null | 外部模型路径对应的准确清单引用 |
| `--model-path` | null | 已准备文件，需配合 --asset-id，不自动下载 |
| `--image-path` | required | 验证图像根目录，递归扫描 |
| `--val-csv` | required | 相对路径/类别 CSV |
| `--label-file` | empty string | 可选显示标签，不是 ground truth |
| `--json-save-path` | hgnetv2_cls_results.json | JSON 输出，会覆盖指定文件 |
| `--limit` | 0 | 匹配标签前取排序前 N 张，0=全部 |
| `--top-k / --topk` | 5 | Top-K 精度，1–1000 |
| `--resize-type` | 0 | 直接缩放，1 为 letterbox，与运行示例默认 1 不同 |
| `--priority` | 0 | 调度优先级 |
| `--bpu-cores` | [0] | BPU 核编号 |

<a id="metrics"></a>
## 指标

Top-1 为首位正确数/成功推理数；Top-K 为真值落在 K 个预测内的数量/成功推理数。分母保留源行为，因此必须同时检查缺失和失败图片。仅 K=5 保留 top5_acc，topk_acc 始终为准确命名的指标。FPS 统计循环中的读图及前处理/推理/后处理，不含模型加载、CSV/目录扫描，无预热，不能当作历史表的多线程吞吐。固定图迁移对照使用 ID 一致、分数绝对差 <1e-5，完全平局须提供逐 ID 证据。

<a id="outputs"></a>
## 输出

写入 --json-save-path 并打印报告。字段包含 status（complete/partial/no-results）、扫描/匹配/未匹配/失败/成功数、逐图错误、精度分母、top1_acc/topk_acc（无成功推理时为 null）、可选 top5_acc、elapsed_seconds/fps、asset_id/target/model、数据路径和配置。complete 只表示本次扫描图片全部评测，不证明已覆盖完整 50,000 张数据。复现证据还须保存 stdout/stderr、代码/部署哈希、SDK/板身份及数据集/模型哈希。

<a id="reference-results"></a>
## 参考结果

迁移板端对照、数据集精度及计时均为 **not-run**。以下历史表来自固定源 evaluator：`rdk_x5 @ac115717197920355fc390bb04299b20e6436864`。

源条件：X5 CPU 8×A55@1.8GHz 性能模式、BPU Bayes-e@1GHz。Float Top-1 为量化前 ONNX，Quant Top-1 为部署结果；单线程延迟为单帧单 BPU 核，多线程延迟和 FPS 使用并发提交。源没有固定数据子集、预热或重复次数，可复现条件仍不完整。

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

<a id="boundaries"></a>
## 边界

迁移评测器仅有主机测试，未执行数据集或板端评测。不计算 ONNX 精度、校准质量、单核延迟或多线程吞吐。保留可兼容的源 CLI 名称；外部 --model-path 新增 --asset-id 要求，非法 CSV 改为显式拒绝，图片失败返回非零，K 非 5 时使用 topk_acc 替代误导性的 top5 字段。
