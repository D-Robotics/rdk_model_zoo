English | [简体中文](README_cn.md)

# LPRNet Python runtime

<a id="overview"></a>
## Python inference

Decode license plate text from the prepared LPRNet float32 input tensor.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── lprnet.py  # Model initialization and inference stages
├── cli.py  # Arguments, model selection and result output
├── main.py  # CLI entry: construct model and call predict
└── run.sh  # Run the sample
```

Start with [main.py](main.py): it constructs `LPRNetRecognizer` and calls `predict`. [lprnet.py](lprnet.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

Use Python 3 and NumPy on an RDK X5 image with `hbm_runtime`. Use `--help`, `--list-models` and `--dry-run` to inspect command options and selections. The published `lpr.bin` takes float32 `(1,3,24,94)` input and returns float32 `(1,68,18,1)` logits. The task API also accepts `(1,68,18)` logits; each output must match its declared metadata.

<a id="usage"></a>
## Usage

From the repository root, with `model/lpr.bin` prepared explicitly:

```bash
python3 -m samples.vision.lprnet.runtime.python.main --target x5
```

Success is exit code `0` and one JSON line with `plate`. The zero-argument input default is the absolute sample path `samples/vision/lprnet/test_data/test_input.dat`. `bash samples/vision/lprnet/runtime/python/run.sh --target x5` is an equivalent helper.

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

The CLI prints `target`, qualified `asset_id`, and `plate`. `LPRNetRecognizer.postprocess` returns a Python `str`; it first drops only the bound layout's singleton axes — `(1,68,18,1)` for the released artifact — to the source `(68,18)` CTC payload, then applies argmax over 18 time steps, consecutive duplicate removal, and blank index `67` removal. Raw logits remain float32 and are not softmaxed.

<a id="integration-example"></a>
## Integration example

The following is complete after the model file and bundled input exist; it defines every variable and uses the same explicit stages as `predict`:

```python
from pathlib import Path
from samples.vision.lprnet.runtime.python.cli import resolve_selection
from samples.vision.lprnet.runtime.python.lprnet import LPRNetRecognizer

target = "x5"
asset_id = "x5:lprnet:lpr.bin"
model_path = Path("samples/vision/lprnet/model/lpr.bin")
test_bin = Path("samples/vision/lprnet/test_data/test_input.dat")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
task = LPRNetRecognizer(selection)
prepared = task.preprocess(test_bin)
raw_logits = task.infer(prepared.tensors)
plate = task.postprocess(raw_logits)
assert plate == task.predict(test_bin)
print(plate)
```

<a id="stage-io"></a>
## Stage I/O

- `preprocess(test_bin)` reads exactly `1*3*24*94` float32 values and returns `PreparedInput(tensors, context)` with one NCHW tensor. No image transform is performed.
- `infer(tensors)` validates the bound name/shape/dtype, calls the selected model, and returns an owned raw float32 array in the bound native shape — `(1,68,18,1)` for the released `lpr.bin`; every call must match the bound shape exactly.
- `postprocess(raw)` removes only singleton axes of the bound native logits (never a reshape or axis reorder) and applies the source CTC-style decode, returning `str`.
- `predict(test_bin)` serially composes all three stages; the context is the input path and is not stored as mutable task state.

<a id="troubleshooting"></a>
## Troubleshooting

- A model path without `--asset-id x5:lprnet:lpr.bin` is rejected to prevent filename-based protocol guessing.
- A missing or wrongly sized `.dat` file returns an error before SDK execution.
- Unknown input/output names, shape, or dtype fail metadata binding; runtime casts are not used to hide a mismatch.
- An output metadata shape outside `(1,68,18,1)`/`(1,68,18)` — for example `(1,18,68,1)` or `(1,68,18,2)` — fails binding, and a runtime call whose output drifts from the bound shape fails before decoding.
