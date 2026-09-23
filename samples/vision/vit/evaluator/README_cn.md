# ViT 评测

<a id="dataset"></a>
## 数据集

随附十张 CIFAR-10 图片每类一张，仅做功能检查，不代表完整数据集精度。未含完整 CIFAR-10 评测集，源没有专用数据集精度 evaluator。历史子集/版本/协议记录不完整，不能把十张图片当历史基准集。

<a id="environment"></a>
## 环境

[Runtime prerequisites](../runtime/python/README_cn.md#environment). 主机测试使用注入运行器及真实保留源前处理辅助实现。板端对照需要 S100 和源/统一运行依赖，不需要 OE。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/vit/tests -v
```

下列板端配方未执行。准备模型后同板同图同制品对照；改为 int16 并遍历十张随附图片分别复测，耗时未测。

```bash
# cwd: repository root
bash samples/vision/vit/model/download.sh s100 int8
```

```bash
# cwd: repository root on S100; download int8 first
PYTHONPATH="$PWD:$PWD/platforms/s:$PWD/platforms/s/samples/vision/vit/runtime/python" python3 - <<'PYTHON'
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
from vit import ViT, ViTConfig
from samples.vision.vit.runtime.python.model_binding import resolve_selection
from samples.vision.vit.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.vit.runtime.python.classification import ClassificationTask
variant = "int8"  # repeat with int16 after preparing its artifact
selection = resolve_selection("s100", variant=variant)
image = cv2.imread("samples/vision/vit/test_data/airplane_0000.png")
if image is None:
    raise FileNotFoundError("airplane_0000.png")
legacy = ViT(ViTConfig(str(selection.model_path), resize_type=0))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_raw = legacy.forward(legacy.pre_process(image))
old_top = legacy.post_process(old_raw, topk=5)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5)
new_raw = task.forward(task.pre_process(image))
result = task.post_process(new_raw)
out = Path("outputs") / ("vit-" + variant + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
out.mkdir(parents=True, exist_ok=False)
np.save(out / "legacy.npy", old_raw[legacy.model_name][legacy.output_names[0]])
np.save(out / "unified.npy", new_raw[binding.output_name])
print("legacy", old_top)
print("unified", result.class_ids.tolist(), result.scores.tolist())
print("raw outputs:", out)
np.testing.assert_array_equal(result.class_ids, [i for i, _ in old_top])
np.testing.assert_allclose(result.scores, [s for _, s in old_top], rtol=0, atol=1e-5)
PYTHON
```

<a id="metrics"></a>
## 指标

主机对两缩放模式和三图像形状的 NV12 双平面逐字节对齐源实现；合成 F32 logits 的 Top-5 ID 一致，分数误差≤1e-7。板端预设标准：ID 一致、分数绝对误差≤1e-5；完全平局需逐 ID 证据，不自动放宽。数据集 Top-1/Top-5 是真值是否在选中类别中，本轮没有新数据集结果。

<a id="outputs"></a>
## 输出

主机测试输出 unittest 结果。板端配方在全新 outputs/vit-<variant>-<UTC> 目录保存完整 legacy.npy/unified.npy，打印双方 Top-K。另保留板镜像/SDK 身份、准确 argv/cwd/UTC/rc/完整 stdout+stderr、代码/部署文件/模型/图片/标签摘要与 metadata。

<a id="reference-results"></a>
## 参考结果

Source: `rdk_s @380e1a2bf42041af54be6f34935e50197cfadff9`, `samples/vision/vit/evaluator/README_cn.md`.

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ONNX | 74.54% | 98.36% |
| HBM | 72.62% | 98.03% |

源说明 PTQ 使用 50 张校准图，无 QAT；未区分 int8/int16，也没有完整基准回执。这些为历史值，本轮未测。统一板端与数据集结果：not-run。

<a id="boundaries"></a>
## 边界

没有专用数据集 evaluator 或延迟基准程序。主机回归不能证明 BPU 行为、当前模型字节、int8/int16 精度或 OE 可复现性。未证实制品专属量化契约前，明确拒绝量化 raw 输出。
