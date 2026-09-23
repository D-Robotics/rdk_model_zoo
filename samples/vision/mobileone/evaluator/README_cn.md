# MobileOne 评测

<a id="dataset"></a>
## 数据集

功能输入为随附 `test_data/tiger_beetle.JPEG`。源没有交付数据集级评测程序，也没有包含 ImageNet 验证集及其准备流程。单图不能证明 ImageNet 精度。

<a id="environment"></a>
## 环境

主机：sample requirements，源对照额外使用 SciPy。板端：runtime README 所述 X5 环境。检查复用同一分类任务；本目录提供操作说明，没有另一个基准可执行程序。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/mobileone/tests -v
```

板端功能检查（耗时未测）：

```bash
# cwd: repository root
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/runtime/python/main.py \
  --target x5 --variant s0 \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

5 个变体与 X5 两种内存板位分别执行。使用下方保留的源任务 API 进行同进程对照。模型字节、图像、resize、Top-K 和调度须一致。原 CLI 仍可使用，但格式化分数不如这里保存的原始数组精确。

```bash
# cwd: repository root on X5, prepare variant s0 first
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/mobileone/runtime/python" python3 - <<'PYTHON'
import cv2
import numpy as np
from mobileone import MobileOne, MobileOneConfig
from samples.vision.mobileone.runtime.python.model_binding import resolve_selection
from samples.vision.mobileone.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.mobileone.runtime.python.classification import ClassificationTask

model_path = "samples/vision/mobileone/model/MobileOne_S0_224x224_nv12.bin"
image = cv2.imread("samples/vision/mobileone/test_data/tiger_beetle.JPEG")
if image is None:
    raise FileNotFoundError("tiger_beetle.JPEG")
legacy = MobileOne(MobileOneConfig(model_path, resize_type=1, topk=5))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
old_outputs = legacy.forward(legacy.pre_process(image))
old_ids, old_scores, _ = legacy.post_process(old_outputs)
selection = resolve_selection("x5", variant="s0")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ClassificationTask(runner, binding, top_k=5, resize_type=1)
new_outputs = task.forward(task.pre_process(image).tensors)
result = task.post_process(new_outputs)
# Preserve full vectors in a new output directory; do not overwrite old evidence.
from pathlib import Path
from datetime import datetime, timezone
out = Path("outputs") / ("mobileone-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
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

<a id="metrics"></a>
## 指标

主机前处理：三个确定性图像尺寸、两种 resize 模式逐字节比较。主机分数对照：相同合成 F32 输出，Top-K ID 一致、分数绝对差 ≤1e-7。拟采用板端判据：ID 一致且分数绝对差 <1e-5；完全平局须提供逐 ID 证据，不能静默放行。未测数据集精度与计时。

<a id="outputs"></a>
## 输出

主机测试打印 unittest 结果；板端 CLI 打印推理结果并可选保存可视化。板端证据须包含代码 SHA/部署文件哈希、板身份/SDK、精确 argv/cwd、UTC 时间、退出码、完整 stdout/stderr、制品/图片/标签哈希、metadata 与原始输出。

<a id="reference-results"></a>
## 参考结果

迁移板端对照、数据集精度及计时均为 **not-run**。以下历史表来自固定源 evaluator：`rdk_x5 @ac115717197920355fc390bb04299b20e6436864`。

源条件：X5 CPU 8×A55@1.8GHz 性能模式、BPU Bayes-e@1GHz。Float Top-1 为量化前 ONNX，Quant Top-1 为部署结果；单线程延迟为单帧单 BPU 核，多线程延迟和 FPS 使用并发提交。源没有固定数据子集、预热或重复次数，可复现条件仍不完整。

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileOne_S4 | 224x224 | 14.8 | 78.75% | 76.50% | 4.58 | 15.44 | 256.52 |
| MobileOne_S3 | 224x224 | 10.1 | 77.27% | 75.75% | 2.93 | 9.04 | 437.85 |
| MobileOne_S2 | 224x224 | 7.8 | 74.75% | 71.25% | 2.11 | 6.04 | 653.68 |
| MobileOne_S1 | 224x224 | 4.8 | 72.31% | 70.45% | 1.31 | 3.69 | 1066.95 |
| MobileOne_S0 | 224x224 | 2.1 | 69.25% | 67.58% | 0.80 | 1.59 | 2453.17 |

<a id="boundaries"></a>
## 边界

主机测试既不执行 BPU，也不认证编译后的制品。OE 导出/量化、数据集精度、延迟和稳定性尚未测试；历史基准不属于本次测量。
