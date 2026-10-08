English | [简体中文](README_cn.md)

# HIMLoco Python policy stages

<a id="overview"></a>
## Python inference

Run the HIMLoco policy offline on X5 from six frames of observation history. `HimLocoTask.from_model` loads the runtime; `predict` returns twelve raw policy actions. `cli.py` reads inputs and writes action files and reports.

<a id="directory"></a>
## Directory structure

```text
python/
├── cli.py  # Arguments, model selection and result presentation
├── input_io.py  # Input files and data records
├── main.py  # Command-line entry: construct the model and call predict
├── model_binding.py  # Model selection and physical tensor contracts
├── policy.py  # Model stages and prediction
└── run.sh  # Locate the Python entry and forward arguments
```

<a id="environment"></a>
## Environment

The core requires Python and NumPy only. It imports no board SDK, Torch, robot
middleware or conversion toolchain. Run the commands below from repository root.
Use the repository's host Python environment with NumPy installed. Actual X5
inference requires the BSP-provided `hbm_runtime` matching the board libraries;
the core does not install or substitute that dependency.

<a id="usage"></a>
## Usage

Construct `HimLocoTask.from_model(selection)` after preparing the model. Its runtime accepts `{"obs_history": float32[1,270]}` and returns `{"actions": float32[1,12]}`. Use `predict(observation)` for one observation history, or call the three stages shown below.

```bash
# Repository root; these commands do not load SDKs or download.
python samples/robotics/himloco/runtime/python/main.py --list-models
python samples/robotics/himloco/runtime/python/main.py --target x5 --dry-run

# Explicit model preparation, followed by offline inference on X5.
bash samples/robotics/himloco/model/download_model.sh --target x5
python samples/robotics/himloco/runtime/python/main.py --target x5 \
  --input-path samples/robotics/himloco/test_data/obs_history \
  --output-dir outputs/himloco
```
`run.sh` forwards the same arguments and accepts `PYTHON` for the interpreter.
A new output directory is required for each run. An alternate model path requires
`--asset-id x5:himloco:himloco_go2_bayese_1x270.bin`; the same published SHA-256
is enforced. Target mismatch or missing/mismatched BIN fails before SDK creation.
Use the BSP runtime, not an unrelated PyPI package named hbm_runtime.

<a id="parameters"></a>
## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | Local detection for execution; list maps auto to x5; dry-run needs explicit x5 |
| `--list-models` | `false` | Print the exact publication without SDK/network/file writes |
| `--dry-run` | `false` | Preview selection only; not actual metadata verification |
| `--asset-id` | `null` | Explicit published identity for external model paths |
| `--model-path` | `null` | Defaults to model/bayes-e/himloco_go2_bayese_1x270.bin under this sample |
| `--input-path` | `samples/robotics/himloco/test_data/obs_history` | One numeric BIN or directory; sample-local absolute default |
| `--output-dir` | `outputs/himloco` | New action directory relative to cwd |
| `--report` | `null` | Defaults to output-dir/report.json; alternate file must be new |
| `--warmup` | `10` | Nonnegative runs of the first input, excluded from reported samples |
| `--priority` | `null` | Optional integer 0–255 passed to SDK |
| `--bpu-cores` | `null` | Optional nonempty list of nonnegative SDK core indexes |

| API input | Contract |
| --- | --- |
| `HimLocoTask.from_model(selection)` | Initialize the runtime from the selected model |
| `observation` | Exactly 270 finite real numeric values, flattened in input order and converted to float32 |
| `preprocess(...).tensors` | Owned, contiguous float32 `obs_history` `[1,270]` |
| `infer(tensors)` | Exact named physical input; returns `RawOutputs` containing raw actions and this call's latency |
| `postprocess(raw)` | Requires `RawOutputs`; no instance-level “last result” dependency |

Flat `[270]`, batch `[1,270]` and history `[6,45]` arrays preserve the source's
flattening convention. Complex, boolean, string/object, non-finite and float32
overflow inputs are rejected. Passing raw sensor data does not prepare the
training observation automatically.

<a id="results"></a>
## Results

`HimLocoResult.actions` is an independently owned float32 `[1,12]` array.
No clipping, activation, reordering, dequantization or scaling is applied.
`latency_ms` measures the synchronous injected runner call, excluding the task's
copies, preprocessing and postprocessing; it is not accelerator-only latency.
Each raw result carries its own time, so processing A after running B cannot
silently report B's time. SDK exceptions propagate; there is no fallback action.

The source deployment applies `default_joint_position + 0.25 * actions` outside
the model boundary. This core returns actions only and does not issue robot commands.

Successful CLI execution returns 0 and writes `000000.bin`-style source-indexed
little-endian float32 files, 48 bytes each, plus a `completed` JSON report. The
report includes model/input/output digests, manifest provenance, runtime metadata,
requested scheduling, completed warmups, UTC times and minimum/mean/p50/p95/maximum
runner latency. After output creation, errors return 2 and preserve a `failed`
report, current source index and completed files; aggregate latency is absent on
partial failure. Preflight failures create no result directory. Existing results
are never reused or overwritten. A killed process can leave a `running` report;
treat it as incomplete.

Inputs are numerically named BIN files, exactly 1080 bytes each, sorted by numeric
source index. Duplicate indexes are rejected. A colocated
`../runtime-input-manifest.json`, when present, must match the fixed input contract
and each selected file's index/digest. Without one, the report records null source
manifest provenance rather than inventing it. Files are hashed from the same bytes
used for inference. No text transcript or controller action is produced.

<a id="integration-example"></a>
## Integration example

After preparing the X5 model, run from the repository root. The example reads the first observation history and returns policy actions with shape `(1,12)`.

```python
from pathlib import Path
import numpy as np
from samples.robotics.himloco.runtime.python.model_binding import resolve_selection
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

selection = resolve_selection("x5")
task = HimLocoTask.from_model(selection)
task.set_scheduling_params(priority=0, bpu_cores=[0])
input_dir = Path("samples/robotics/himloco/test_data/obs_history")
input_path = min(input_dir.glob("*.bin"), key=lambda path: int(path.stem))
observation = np.fromfile(input_path, dtype="<f4").reshape(6, 45)
result = task.predict(observation)
print(result.actions.shape, result.actions.tolist())

# Optional access to intermediate stages.
prepared = task.preprocess(observation)
raw = task.infer(prepared.tensors)
staged_result = task.postprocess(raw)
```

<a id="stage-io"></a>
## Stage semantics and source fidelity

Each 45-value observation contains velocity commands (3), angular velocity (3,
source scale 0.25), projected gravity (3), relative joint positions (12), relative
joint velocities (12, source scale 0.05), and previous actions (12). The current
observation comes first, followed by five earlier observations. This task does not
apply those scales again, accumulate history or choose a joint ordering.

`preprocess` owns the packed features; `infer` executes one model call; `postprocess` validates and owns the action array; `predict` composes the three stages. `pre_process`, `forward` and `post_process` delegate to the same stage implementations. Results and timing records belong to each call.

<a id="troubleshooting"></a>
## Troubleshooting

Wrong observation count requires reconstructing the training-policy history, not
padding or truncating blindly. Wrong output dtype/name/shape requires checking the
bound model interface; the core will not silently cast an incompatible model's
output. Latency attached to a hand-built `RawOutputs` must be finite and nonnegative.
See the [C++ guide](../cpp/README.md) for native execution and the
[evaluator guide](../../evaluator/README.md) for action-dump comparisons.
