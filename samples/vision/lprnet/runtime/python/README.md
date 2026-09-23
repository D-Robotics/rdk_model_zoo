# LPRNet Python runtime

<a id="environment"></a>
## Environment

Use Python 3 and NumPy on an RDK X5 image with `hbm_runtime` available. The runtime imports the board SDK only after selection, model-file, and board checks. `--help`, `--list-models`, and `--dry-run` are SDK-free. The compiled model must expose one float32 input `(1,3,24,94)` and one float32 output `(1,68,18)`.

<a id="usage"></a>
## Usage

From the repository root, with `model/lpr.bin` prepared explicitly:

```bash
python3 -m samples.vision.lprnet.runtime.python.main --target x5
```

Success is exit code `0` and one JSON line with `plate`. The zero-argument input default is the absolute sample path `samples/vision/lprnet/test_data/test_input.dat`. `bash samples/vision/lprnet/runtime/python/run.sh --target x5` is an equivalent wrapper.

<a id="parameters"></a>
## Parameters

| option | default | meaning |
|---|---|---|
| `--target` | `auto` | `auto` resolves the only published target, X5; S targets are rejected |
| `--asset-id` | `null` | exact `x5:lprnet:lpr.bin`, required with external model path |
| `--model-path` | `null` | existing model path; no download |
| `--test-bin` | `samples/vision/lprnet/test_data/test_input.dat` | pre-packed float32 input |
| `--priority` | `5` | runtime scheduling priority |
| `--bpu-cores` | `[0]` | one or more BPU core indexes |
| `--list-models` | `false` | print manifest assets without SDK |
| `--dry-run` | `false` | print binding contract without SDK or model load |

`--list-models` and `--dry-run` are mutually exclusive. User/runtime errors return `2`.

<a id="results"></a>
## Results

The CLI prints `target`, qualified `asset_id`, and `plate`. `LPRNetTask.post_process` returns a Python `str`; it applies argmax over 18 time steps, consecutive duplicate removal, and blank index `67` removal. Raw logits remain float32 and are not softmaxed.

<a id="integration-example"></a>
## Integration example

The following is complete after the model file and bundled input exist; it defines every variable and uses the same explicit stages as `predict`:

```python
from pathlib import Path
from samples.vision.lprnet.runtime.python.model_binding import resolve_selection
from samples.vision.lprnet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lprnet.runtime.python.lprnet import LPRNetTask

target = "x5"
asset_id = "x5:lprnet:lpr.bin"
model_path = Path("samples/vision/lprnet/model/lpr.bin")
test_bin = Path("samples/vision/lprnet/test_data/test_input.dat")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=5, bpu_cores=[0])
task = LPRNetTask(runner, binding)
prepared = task.pre_process(test_bin)
raw_logits = task.forward(prepared.tensors)
plate = task.post_process(raw_logits)
assert plate == task.predict(test_bin)
print(plate)
```

<a id="stage-io"></a>
## Stage I/O

- `pre_process(test_bin)` reads exactly `1*3*24*94` float32 values and returns `PreparedInput(tensors, context)` with one NCHW tensor. No image transform is performed.
- `forward(tensors)` validates the bound name/shape/dtype, calls the selected model, and returns an owned raw float32 `(1,68,18)` array.
- `post_process(raw)` performs only the source CTC-style decode and returns `str`.
- `predict(test_bin)` serially composes all three stages; the context is the input path and is not stored as mutable task state.

<a id="troubleshooting"></a>
## Troubleshooting

- A model path without `--asset-id x5:lprnet:lpr.bin` is rejected to prevent filename-based protocol guessing.
- A missing or wrongly sized `.dat` file returns an error before SDK execution.
- Unknown input/output names, shape, or dtype fail metadata binding; runtime casts are not used to hide a mismatch.
