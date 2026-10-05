# HIMLoco Python policy stages

[中文](README_cn.md)

The unified Python entry now provides exact X5 model selection, lazy SDK transport,
source-indexed input checks, warmup, owned action dumps and failure reports. Host
integration tests use an explicit SDK double; real board execution remains not-run.
The entry (`main.py`) visibly runs the offline loop: it obtains the bound
`HimLocoTask` through `application.load_task`, executes the explicitly requested
warmup predictions, calls `task.predict(observation)` once per input and records
each action dump via `application.record_sample`; the evidence discipline
(target/asset gating, report reservation, digest re-verification, latency summary,
failure records) lives in `application.py` helpers (`prepare`, `load_task`,
`record_sample`, `complete`), whose single-call composition `application.execute`
remains the compatibility API.
The source runtime guide (historical `../../../../../platforms/x5/samples/robotics/himloco/runtime/python/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)
retains historical board evidence, not a new unified-runtime validation claim.

<a id="environment"></a>
## Environment

The core requires Python and NumPy only. It imports no board SDK, Torch, robot
middleware or conversion toolchain. Run the commands below from repository root.
Use the repository's host Python environment with NumPy installed. Actual X5
inference will require the BSP-provided `hbm_runtime` matching the board libraries;
the core does not install or substitute that dependency.

<a id="usage"></a>
## Usage

Construct `HimLocoTask(runner)`, where `runner` accepts the physical mapping
`{"obs_history": float32[1,270]}` and returns `{"actions": float32[1,12]}`.
The SDK adapter is responsible for target, asset identity and actual model metadata.
This core never downloads a model or opens a device. Use `predict(observation)`
for normal composition, or the three explicit methods shown below.

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
| `runner` | Required callable; no default or implicit SDK construction |
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
## Executable integration example

```bash
python - <<'PYCODE'
import numpy as np
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

def fixture_runner(tensors):
    assert tensors["obs_history"].shape == (1, 270)
    return {"actions": np.arange(12, dtype=np.float32).reshape(1, 12)}

task = HimLocoTask(fixture_runner)
observation = np.zeros((6, 45), dtype=np.float32)
prepared = task.preprocess(observation)
raw = task.infer(prepared.tensors)
explicit = task.postprocess(raw)
result = task.predict(observation)
np.testing.assert_array_equal(explicit.actions, result.actions)
print(result.actions.shape, result.actions.tolist())
PYCODE
```

Expected output: shape `(1, 12)` and actions 0 through 11. These values come from
the fixture, not a learned policy. The core stores no history or per-call context;
callers provide all six observations. Serialize access if the injected SDK runner
uses shared buffers; owned results do not imply a thread-safe device runtime.

<a id="stage-io"></a>
## Stage semantics and source fidelity

Each 45-value observation contains velocity commands (3), angular velocity (3,
source scale 0.25), projected gravity (3), relative joint positions (12), relative
joint velocities (12, source scale 0.05), and previous actions (12). The current
observation comes first, followed by five earlier observations. This task does not
apply those scales again, accumulate history or choose a joint ordering.

`preprocess` packs and owns features; `infer` performs one raw model call;
`postprocess` validates and owns raw actions; `predict` composes these methods.
The established `pre_process`, `forward`, and `post_process` names remain
importable thin aliases of `preprocess`, `infer`, and `postprocess` — one
implementation, two names.
Source preprocessing was compared against all 21 archived observation files with
manifest digest checks. Postprocessing uses synthetic action outputs for numerical
comparison. No board/model accuracy, control stability or robot motion was tested.
The original source's mutable last-latency field and potentially aliased outputs
are replaced by per-call records and independent result storage.

<a id="troubleshooting"></a>
## Troubleshooting

Wrong observation count requires reconstructing the training-policy history, not
padding or truncating blindly. Wrong output dtype/name/shape requires checking the
bound model interface; the core will not silently cast an incompatible model's
output. Latency attached to a hand-built `RawOutputs` must be finite and nonnegative.
See the [C++ guide](../cpp/README.md) for native execution and the
[evaluator guide](../../evaluator/README.md) for action-dump comparisons.
Host tests do not establish SDK/board compatibility.
