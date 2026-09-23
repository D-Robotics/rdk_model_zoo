English | [简体中文](./README_cn.md)

# Evaluator — DINOv2 ViT-S/14

This document preserves the source performance and accuracy records. Every number below is historical source data; this migration did not download models or datasets, run a board, or rerun the evaluator.

<a id="dataset"></a>
## Dataset

The source runtime smoke path uses the repository fixtures `samples/vision/dinov2/test_data/dog.jpg` and `bus.jpg`. The source PTQ report used 50 diverse real calibration images; it also states that identical values were reproduced with an independent export and a different 50-image calibration set. No evaluator dataset preparation script or fixed dataset archive is supplied. Calibration preparation is implemented by `conversion/mapper.py` and documented in [`../conversion/README.md`](../conversion/README.md).

```text
# cwd: repository root
samples/vision/dinov2/test_data/dog.jpg
samples/vision/dinov2/test_data/bus.jpg
# board benchmark inputs: same preprocessed float32 tensors as the runtime contract
```

<a id="environment"></a>
## Environment

- Historical board records: RDK S100/Nash-E, S100P/Nash-M, and S600/Nash-P with `hrt_model_exec` or `hbm_runtime`.
- PTQ report: OE 3.7.0, hmct 2.6.5 / hbdk 4.7.5 on Nash-E.
- Runtime dependencies: board-image `hbm_runtime`; host utilities use Python 3.14.7, NumPy, and OpenCV.
- Board image, firmware, and current runtime versions were not verified. This migration did not run evaluation.

<a id="command"></a>
## Evaluation Command

The source performance commands below reproduce the historical thread/core settings when run on the matching board with its target artifact. They are documented commands, not commands executed in this migration.

```bash
# cwd: samples/vision/dinov2/evaluator on the target board; artifact prepared under ../model/
# S100 / Nash-E
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 2

# S100P / Nash-M
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 2

# S600 / Nash-P
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 12 --core_id 1,2,3,4
# expect: BPU latency/throughput over 200 frames, after locking the performance governor
```

For accuracy, run the exported float ONNX with ONNXRuntime on the same preprocessed inputs, run the HBM with `hbm_runtime.HB_HBMRuntime(...).run()` on the board, then compute cosine similarity separately for `cls_feat` and `patch_feat`. The source runtime CLI's two-image path is documented in [`../runtime/python/README.md`](../runtime/python/README.md).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `thread_num` | int | `1` in the baseline record | `hrt_model_exec perf` worker count. |
| `core_id` | CSV ints | unset, except S600 12-thread record | BPU cores used for the S600 high-concurrency record: `1,2,3,4`. |
| `frames` | int | `200` in source records | Performance measurement scope. |
| `input` | tensor | `(1,3,224,224)` F32 | RGB normalized tensor produced by the runtime preprocessing contract. |

### Same-board migration comparison for both outputs (not-run)

The complete procedure below uses the preserved legacy runtime and the unified API. It compares preprocessing, then saves both raw outputs and postprocessed results. Change `target` to run each board independently. Every run creates a unique UTC directory. Integer raw arrays require exact equality; floating raw/results use `rtol=0, atol=1e-5`. Any mismatch exits nonzero. This checks migration parity, not historical ONNX accuracy; it has not been executed on a board here.

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


Historical ONNX-to-HBM accuracy reproduction remains a **manual procedure without a complete evaluator implementation (manual / not-implemented)**. It additionally needs the fixed floating ONNX, identical per-image input tensors, and separate reference arrays for both outputs before dequantization and cosine aggregation. Neither the migration comparison above nor the runtime cosine between two different images measures that accuracy. Existing commands do not independently reproduce the historical accuracy table.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| BPU latency | Pure BPU forward latency for one model invocation. | 200 frames, performance governor locked; board and thread/core settings in the tables. CPU preprocessing is additional. |
| BPU throughput | Frames per second reported by `hrt_model_exec perf`. | Same 200-frame run; concurrency is stated per row. |
| Calibrated cosine | Cosine between calibration/toolchain output and float reference for each output. | PTQ report, Nash-E, featuremap float32 input, all-int16, default KL calibration. |
| Quantized cosine | Cosine between quantized output and float ONNX output for each output. | PTQ report, Nash-E; `cls_feat` and `patch_feat` measured separately. |
| Board cosine range | Min/max cosine range over board executions against float ONNX references. | Same input preprocessing; S100, S100P, S600 separately; source board record. |

The canonical preprocessing is OpenCV BGR→RGB, bicubic short-side resize to 256, center crop 224, `/255`, ImageNet mean/std, and contiguous float32 NCHW. Outputs are compared before any softmax or L2 operation.

<a id="outputs"></a>
## Outputs

The runtime CLI emits JSON statistics and optional exact NumPy output files. The migration comparison saves full input/raw/result arrays plus comparison.json. The historical ONNX accuracy comparison has no implemented evaluator; it would need separate cosine values for cls_feat and patch_feat. No current board output or fresh metric record exists.

<a id="reference-results"></a>
## Reference Results

The following tables preserve every row and column from the source evaluator. Status in this migration: `not-run`. Source: `platforms/s/samples/vision/dinov2/evaluator/README.md` and corresponding entries in `platforms/s/docs/release/benchmarks.yaml`.

### Historical performance (not rerun)

| Device | Model | Input Size | BPU Task Latency / BPU Throughput |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS (1 thread) <br> 288.26 FPS (2 threads) |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS (1 thread) <br> 357.63 FPS (2 threads) |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS (1 thread) <br> 1898.42 FPS (12 threads, `--core_id 1,2,3,4`) |

Model parameters: 22.06 M. Latency is pure BPU forward; CPU preprocessing is additional.

### Historical PTQ per-output cosine — Nash-E only (not rerun)

This quantization-quality table is only for the Nash-E toolchain report; it must not be generalized to S100P or S600.

| Output | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

The source reports identical values from an independent export and a different 50-image calibration set.

### Historical board-executed cosine vs float ONNX (not rerun)

| Device | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

<a id="boundaries"></a>
## Boundaries

- This directory has no standalone evaluator implementation; reproduction uses `hrt_model_exec`, ONNXRuntime, `hbm_runtime`, and the runtime CLI.
- All reference values are historical and not a current board claim. No board test, HBM download, or conversion was performed here.
- The PTQ quantization-quality table is explicitly Nash-E only. Board cosine ranges are separately attributed to each target.
- DINOv2 is documented as a vision feature encoder. Text encoding, classification labels, retrieval datasets, and C++ evaluation are outside this sample.

## License

Evaluator documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
