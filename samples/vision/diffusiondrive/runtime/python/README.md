[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive Python runtime

<a id="environment"></a>
## Environment

Real inference requires S100P or S600 with its matching `hbm_runtime`, Python, NumPy and OpenCV. The two targets use distinct published models; no S100/X5 asset exists. The [explicit downloader](../../model/README.md) prepares and verifies the selected HBM. Inference never installs dependencies or downloads a model. No real SDK/board run was performed during this migration; host tests inject a runtime fixture.

All commands below run from the repository root. Shell wrappers also change to that root. The original task combined SDK ownership, quantization and rendering; the new task delegates SDK transport to the shared named-array runner and keeps data IO, transforms and visualization in separate modules.

<a id="usage"></a>
## Single-case and batch usage

Inspect an explicit target without importing the board SDK or opening an HBM:

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --dry-run
```

After preparing the correct board environment and model, run a supplied case with a new output directory:

```bash
bash samples/vision/diffusiondrive/runtime/python/run.sh --target s600 --input-npz samples/vision/diffusiondrive/test_data/case_017/inputs.npz --output outputs/diffusiondrive_case017
```

To inspect or execute the five source cases:

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s100p --output outputs/diffusiondrive_cases --dry-run
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s100p --output outputs/diffusiondrive_cases
```

Batch dry-run validates all five input NPZ files and prints commands without SDK execution or output creation. Execution uses the same single-case CLI, loads the model per case as the source did, stops at the first nonzero return, and writes `batch-report.json` with completed return codes, remaining cases and available report digests. A partially completed batch is not five passes. Batch output uses `--output`; the source wrapper's positional output argument is replaced.

<a id="parameters"></a>
## Parameters

Single-case parser defaults are listed literally; resolved paths are explained separately.

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` / `--platform` | `auto` | Explicit target or recognized local identity; no unknown-host S600 fallback |
| `--asset-id` | `null` | Inferred target asset; can identify target under auto |
| `--model-path` | `null` | Resolves target HBM under sample model directory; external path requires asset ID |
| `--input-npz` | `samples/vision/diffusiondrive/test_data/reference_inputs.npz` | Exact four logical feature arrays |
| `--output` | `outputs/diffusiondrive` | New canonical result directory |
| `--output-npz` | `null` | Optional extra copy of decoded outputs; canonical archive is always retained |
| `--img-save-path` / `--output-image` | `null` | Optional extra PNG/JPEG/BMP, encoded according to extension |
| `--agent-score-thres` | `0.5` | Finite [0,1], source sigmoid probability comparison is >= |
| `--priority` | `0` | Integer SDK priority in 0..255 |
| `--bpu-cores` | `[0]` | One or more nonnegative core IDs; actual support depends on runtime |
| `--list-models` | `false` | List manifest identity/checksum without execution |
| `--dry-run` | `false` | Resolve selection and print without execution |

Inspection modes are mutually exclusive. Batch uses the same target/asset/model/threshold/scheduling/inspection flags, plus `--cases-root` (default `samples/vision/diffusiondrive/test_data`) and `--output` (default `outputs/diffusiondrive_cases`). It has no per-case `--input-npz` or extra output-file options. Case order is 000,017,042,073,099.

Output directories and extra paths must be new. Extra destinations must be distinct and cannot replace canonical arrays, image or report. The image alias and `--platform` are retained from source. Original file-output defaults are replaced by a per-run directory; environment target overrides and implicit downloads are removed. Relative direct-Python paths are relative to the current working directory.

<a id="results"></a>
## Saved results

| File | Contract |
| --- | --- |
| `physical_inputs.npz` | Four actual quantized/cast input arrays, names/shapes/dtypes preserved |
| `raw_outputs.npz` | Four raw physical runtime tensors before dequantization |
| `outputs.npz` | Decoded trajectory, agent states/scores/mask, BEV logits/labels |
| `result.png` | Camera/BEV/LiDAR/trajectory/agent visualization |
| `report.json` | Asset and file digests, actual input/output metadata including quantization, scheduling, threshold, UTC interval and processing limits |

Decoded arrays: float32 trajectory `[1,8,3]`, agent states `[1,30,5]`, scores `[1,30]`, BEV logits `[1,7,128,256]`; bool agent mask `[1,30]`; uint8 BEV labels `[1,128,256]`. Runtime version is recorded when exposed, otherwise `unknown`. UTC fields bracket work but are not a latency benchmark. No warmup, noise regeneration or actuation occurs.

Optional decoded archives are byte copies. Optional images are separately encoded; JPEG is lossy and need not match PNG pixels. A failed write can leave a partial output directory; inspect the return code and report before using results. Return code 0 means completed processing/output, not driving quality. The [offline evaluator](../../evaluator/README.md) consumes decoded `outputs.npz`, not raw tensors.

<a id="integration-example"></a>
## Application integration

From the repository root on a prepared S600, this executes the same task path without writing or rendering:

```python
from samples.vision.diffusiondrive.runtime.python.model_binding import resolve_selection
from samples.vision.diffusiondrive.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.diffusiondrive.runtime.python.data_io import load_features
from samples.vision.diffusiondrive.runtime.python.diffusiondrive import DiffusionDriveTask

selection = resolve_selection("s600")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
features = load_features("samples/vision/diffusiondrive/test_data/reference_inputs.npz")
task = DiffusionDriveTask(runner, binding, agent_score_threshold=0.5)
result = task.predict(features)
assert result["trajectory"].shape == (1, 8, 3)
assert result["bev_labels"].shape == (1, 128, 256)
```

Call the stages individually to retain physical inputs and raw outputs. Results own their arrays. Keep the SDK runner alive while using the task; do not share it across concurrent calls without synchronization. The task no longer owns the SDK or supplies a `__call__` alias; use `predict`. It does not accept raw camera/LiDAR sensors in place of prepared feature tensors.

<a id="stage-io"></a>
## Stage IO and quantization

| Stage | Input | Output |
| --- | --- | --- |
| `pre_process` | Exact four finite float32 logical arrays | Flat name→physical array mapping using bound dtype/quantization |
| `forward` | Physical mapping | Owned raw named outputs; no semantic decoding |
| `post_process` | Exact four raw arrays matching metadata | Owned decoded six-array result |
| `predict` | Logical features | Composition of the three stages |

Logical input shapes: camera `[1,3,256,1024]`, lidar `[1,1,256,256]`, status `[1,8]`, noise `[1,20,8,2]`. Source output names/shapes: trajectory `[1,8,3]`, agent_states `[1,30,5]`, agent_labels `[1,30]`, bev_semantic_map `[1,7,128,256]`. Binding requires exactly one model and exact name sets; name order is irrelevant.

Physical types may be int8/uint8/int16/uint16/int32/uint32/float16/float32, subject to actual metadata and transform validation. Integer tensors require explicit positive finite SCALE descriptors; input scales must be per-tensor. Empty quantization on floating tensors means casting/pass-through; NONE descriptors may contain zero-valued zero-point placeholders. Nonempty floating SCALE descriptors follow the source affine behavior. Output scales may be per-axis with a matching axis length; scalar zero points broadcast across channels, correcting a source reshape error. Integer zero points must be integral and in range. Binding snapshots transform values without copying SDK descriptor objects.

Input quantization preserves source float32 `rint(x/scale + zero)`; integer clipping uses float64 bounds before the final cast to avoid an int32/uint32 upper-bound wrap. Missing integer scales, malformed metadata and nonfinite inputs/results fail. Output dequantization precedes source sigmoid (logits clipped to [-60,60]), agent threshold and channel-axis BEV argmax. No undocumented sensor normalization, randomized noise or new planning algorithm is inserted. Actual HBM metadata is still unobserved in this host migration.

<a id="troubleshooting"></a>
## Troubleshooting

- Unknown host under auto: use explicit target for inspection/preparation; real execution still requires matching board identity.
- Hash mismatch: use the correct published target file; do not bypass verification with a renamed custom model.
- Feature error: retain exact float32 shapes and names, without extra archive members or regenerated noise.
- Quantization rejection: capture actual metadata and resolve the contract discrepancy before changing the validation.
- Almost-gray BEV: gray is road; inspect logits/labels and reference metrics before assuming a palette bug.
- Batch failure: consult per-case reports and `remaining_cases`; rerun into a new directory after correcting the cause.

Host tests compare all six supplied float cases against actual source postprocessing and rendering, test quantization edge cases and run the real CLI with an injected SDK. They do not prove physical HBM compatibility, board equality, NAVSIM accuracy or performance. Historical tables and validation limits are in the evaluator guide.
