[English](./README.md) | 简体中文

# 模型评估 — DINOv2 ViT-S/14

本文保留源中的性能和精度记录。以下每个数值都是历史源数据；本轮未下载模型或数据集、未运行板卡、未重新执行评估。

<a id="dataset"></a>
## 数据集

源 runtime 冒烟路径使用仓库 fixture：`samples/vision/dinov2/test_data/dog.jpg` 和 `bus.jpg`。源 PTQ 报告使用 50 张多样真实校准图，并说明另一组 50 张校准图配合独立导出脚本复现了相同数值。没有提供评估数据集准备脚本或固定数据集压缩包。校准准备由 `conversion/mapper.py` 实现，见 [`../conversion/README_cn.md`](../conversion/README_cn.md)。

```text
# cwd：仓库根目录
samples/vision/dinov2/test_data/dog.jpg
samples/vision/dinov2/test_data/bus.jpg
# 板端 benchmark 输入：与 runtime 契约相同的预处理 float32 tensor
```

<a id="environment"></a>
## 环境

- 历史板端记录：RDK S100/Nash-E、S100P/Nash-M、S600/Nash-P，使用 `hrt_model_exec` 或 `hbm_runtime`。
- PTQ 报告：OE 3.7.0、hmct 2.6.5 / hbdk 4.7.5，Nash-E。
- runtime 依赖：板端镜像 `hbm_runtime`；主机工具使用 Python 3.14.7、NumPy、OpenCV。
- 板端镜像、固件和当前 runtime 版本未核验。本轮未执行评估。

<a id="command"></a>
## 评估命令

以下源性能命令在匹配板卡和目标制品上运行时，可复现历史的线程/核心设置。本轮只记录命令，没有执行。

```bash
# cwd：目标板上的 samples/vision/dinov2/evaluator；制品已准备在 ../model/
# S100 / Nash-E
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 2

# S100P / Nash-M
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 2

# S600 / Nash-P
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 12 --core_id 1,2,3,4
# 预期：锁定 performance governor 后，输出 200 帧的 BPU 延迟/吞吐
```

精度评估需在主机用 ONNXRuntime 对相同预处理输入运行 float ONNX，在板端用 `hbm_runtime.HB_HBMRuntime(...).run()` 运行 HBM，再分别计算 `cls_feat`、`patch_feat` 的 cosine。源 CLI 双图路径见 [`../runtime/python/README_cn.md`](../runtime/python/README_cn.md)。

| 参数 | 类型 | 示例默认值 | 说明 |
| --- | --- | --- | --- |
| `thread_num` | int | 基线记录为 `1` | `hrt_model_exec perf` 工作线程数。 |
| `core_id` | CSV ints | 未设置；S600 高并发记录除外 | S600 12 线程记录使用 `1,2,3,4`。 |
| `frames` | int | 源记录为 `200` | 性能测量范围。 |
| `input` | tensor | `(1,3,224,224)` F32 | runtime 预处理生成的 RGB 归一化 tensor。 |

### 迁移前后同板双输出对照（待执行）

下面是完整可执行的固定源/统一入口对拍步骤，复用仓库实际保留的源运行时，先比较前处理、再保存两个输出的完整 raw 和反量化结果。修改 `target` 可分别运行三种板卡；每次创建唯一 UTC 目录，不覆盖旧证据。整数 raw 要求逐元素精确相同，浮点 raw/结果使用 `rtol=0, atol=1e-5`；任一比较失败退出非零。它验证迁移一致性，不计算历史 ONNX 精度表。此处尚未在板端执行。

```bash
# cwd: repository root, on the selected board; exact HBM already prepared
PYTHONPATH="$PWD:$PWD/platforms/s:$PWD/platforms/s/samples/vision/dinov2/runtime/python" python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from dinov2 import Dinov2, Dinov2Config
from samples.vision.dinov2.runtime.python.model_binding import resolve_selection
from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
from samples._shared.platforms import require_execution_target

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

repo = Path.cwd()
target = 's100'  # change to s100p or s600 to select that target's own HBM
selection = resolve_selection(target)
require_execution_target(target)
if not selection.model_path.is_file():
    raise FileNotFoundError(selection.model_path)
image_path = repo / 'samples/vision/dinov2/test_data/dog.jpg'
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(image_path)
started = datetime.now(timezone.utc)
output_dir = repo / 'evaluator-output' / ('dinov2-' + started.strftime('%Y%m%dT%H%M%S%fZ'))
output_dir.mkdir(parents=True, exist_ok=False)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
legacy = Dinov2(Dinov2Config(str(selection.model_path)))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
prepared = DINOv2Task(runner, binding).pre_process(image)
old_inputs = legacy.pre_process(image)
np.testing.assert_array_equal(old_inputs[legacy.model_name]['input'], prepared.tensors['input'])
np.save(output_dir / 'input.npy', prepared.tensors['input'], allow_pickle=False)
old_raw = legacy.forward(old_inputs)
new_raw = runner(prepared.tensors)
records = {}
passed = True
for name in ('cls_feat', 'patch_feat'):
    a, b = np.asarray(old_raw[legacy.model_name][name]), np.asarray(new_raw[name])
    np.save(output_dir / ('legacy-raw-' + name + '.npy'), a, allow_pickle=False)
    np.save(output_dir / ('unified-raw-' + name + '.npy'), b, allow_pickle=False)
    same_protocol = a.shape == b.shape and a.dtype == b.dtype
    raw_ok = same_protocol and (np.array_equal(a, b) if np.issubdtype(a.dtype, np.integer)
                               else np.allclose(a, b, rtol=0, atol=1e-5))
    legacy.cfg.output = name
    reference = legacy.post_process(old_raw)
    candidate = DINOv2Task(runner, binding, name).post_process(new_raw)
    np.save(output_dir / ('legacy-result-' + name + '.npy'), reference, allow_pickle=False)
    np.save(output_dir / ('unified-result-' + name + '.npy'), candidate, allow_pickle=False)
    result_ok = reference.shape == candidate.shape and reference.dtype == candidate.dtype and np.allclose(reference, candidate, rtol=0, atol=1e-5)
    records[name] = {'raw_protocol_equal': same_protocol, 'raw_equal': bool(raw_ok),
                     'result_equal': bool(result_ok), 'shape': list(b.shape), 'raw_dtype': str(b.dtype)}
    passed = passed and raw_ok and result_ok
code_paths = [p for base in ('samples/vision/dinov2/runtime/python', 'samples/_shared',
                             'platforms/s/samples/vision/dinov2/runtime/python', 'platforms/s/utils/py_utils')
              for p in (repo / base).glob('*.py')]
report = {'started_utc': started.isoformat(), 'ended_utc': datetime.now(timezone.utc).isoformat(),
          'target': target, 'asset_id': selection.asset.reference,
          'model_sha256': sha256(selection.model_path), 'image_sha256': sha256(image_path),
          'code_sha256': {str(p.relative_to(repo)): sha256(p) for p in code_paths},
          'records': records, 'passed': bool(passed),
          'scope': 'same-board legacy/unified migration parity; not float-ONNX accuracy'}
(output_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(output_dir)
print(json.dumps(report, indent=2))
if not passed:
    raise AssertionError('DINOv2 migration parity failed; full arrays are saved.')
PY
# success: unique output directory, complete input/raw/result arrays and comparison.json; failures exit nonzero
```


历史 ONNX→HBM 的精度复现仍是**人工流程，未提供完整 evaluator 实现（manual / not-implemented）**：必须另准备固定浮点 ONNX、逐图相同的预处理输入及两个输出的引用数组，再逐输出反量化并统计 cosine。不能把上述迁移对拍或两张不同图片之间的 cosine 当作该精度结果；本文不承诺现有命令能独立复现历史精度表。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| BPU 延迟 | 一次模型调用的纯 BPU 前向延迟。 | 200 帧，锁 performance governor；板卡及线程/核心设置见表。 |
| BPU 吞吐 | `hrt_model_exec perf` 报告的每秒帧数。 | 同一 200 帧运行；并发数按行给出。 |
| Calibrated cosine | 校准/工具链输出与 float 参照之间每个 output 的 cosine。 | PTQ 报告，Nash-E，featuremap float32 输入、全 int16、默认 KL 校准。 |
| Quantized cosine | 量化 output 与 float ONNX output 之间每个 output 的 cosine。 | PTQ 报告，仅 Nash-E；分别测量 `cls_feat`、`patch_feat`。 |
| 板端 cosine 范围 | 板端 output 与 float ONNX 参照对拍的最小/最大 cosine。 | 相同预处理；分别记录 S100、S100P、S600。 |

标准预处理为 OpenCV BGR→RGB，bicubic 将短边 resize 到 256，中心 crop 224，`/255`，ImageNet mean/std，contiguous float32 NCHW。比较发生在任何 softmax 或 L2 操作之前。

<a id="outputs"></a>
## 输出

runtime CLI 输出 JSON 统计并可按精确路径保存 NumPy tensor。迁移对照完整保存 input/raw/result 数组及 comparison.json。历史 ONNX 精度尚无完整 evaluator 实现，需另行分别统计 cls_feat/patch_feat 的 cosine。本轮没有新板端输出或指标记录。

<a id="reference-results"></a>
## 参考结果

以下表格保留源评估的全部行和列。本轮状态：`not-run`。来源：`platforms/s/samples/vision/dinov2/evaluator/README.md` 及 `platforms/s/docs/release/benchmarks.yaml` 对应条目。

### 历史性能（本轮未复测）

| Device | Model | Input Size | BPU Task Latency / BPU Throughput |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS (1 thread) <br> 288.26 FPS (2 threads) |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS (1 thread) <br> 357.63 FPS (2 threads) |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS (1 thread) <br> 1898.42 FPS (12 threads, `--core_id 1,2,3,4`) |

模型参数量：22.06 M。延迟为纯 BPU 前向，CPU 预处理另计。

### 历史 PTQ 逐输出 cosine——仅 Nash-E（本轮未复测）

此量化质量表仅属于 Nash-E 工具链报告，不应泛化到 S100P 或 S600。

| Output | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

源记录说明，独立导出脚本和另一组 50 张校准图得到相同数值。

### 历史板端 cosine 对拍 float ONNX（本轮未复测）

| Device | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

<a id="boundaries"></a>
## 边界

- 本目录没有独立评估实现；复现使用 `hrt_model_exec`、ONNXRuntime、`hbm_runtime` 和 runtime CLI。
- 所有参考数值都是历史记录，不是当前板端声明。本轮未做板测、下载 HBM 或转换。
- PTQ 量化质量表明确仅适用于 Nash-E；板端 cosine 范围按 target 分别记录。
- DINOv2 在此作为视觉特征编码器，不覆盖文本编码、分类标签、检索数据集或 C++ 评估。

## 许可

评估文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
