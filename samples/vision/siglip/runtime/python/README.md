English | [简体中文](./README_cn.md)

# Python Runtime — SigLIP vision features

<a id="environment"></a>
## Environment

- Execution target: RDK S100 (Nash-E) or S100P (Nash-M), with a board image that provides `hbm_runtime`. Board image and firmware versions were not verified; host execution uses injected fixtures only.
- Host preparation: Python 3.14.7, `numpy`, `opencv-python`, and `PyYAML` from `../../requirements-host.txt` for contract tests and selection utilities.
- `hbm_runtime` is board-image-only. `--help`, `--list-models`, and explicit-target `--dry-run` intentionally work without importing the SDK or loading a model.

<a id="usage"></a>
## Usage

Prepare an HBM first as described by [`../../model/README.md`](../../model/README.md). Run on the board from the repository root:

```bash
# cwd: repository root; model: prepared default HBM under samples/vision/siglip/model/
python3 samples/vision/siglip/runtime/python/main.py
# success: exit code 0 and JSON statistics for pooler_output
```

A customized run selects the target, variant, patch features, and an output file. The file is written exactly as named; NumPy format does not add an extension.

```bash
# cwd: repository root; model: prepared so400m-patch14-384 HBM; input: bundled dog.jpg
python3 samples/vision/siglip/runtime/python/main.py \
  --target s100p --variant so400m-patch14-384 \
  --submodel last_hidden_state --output-file /tmp/siglip-features.npy
# success: exit code 0, JSON summary printed, and /tmp/siglip-features.npy contains the complete raw array
```

For inspection without SDK or model loading, use `--list-models` or `--dry-run --target s100` (or `s100p`). A host dry run with `--target auto` exits 2 with an explicit-target message.

<a id="parameters"></a>
## Parameters

`--list-models` and `--dry-run` are mutually exclusive. Runtime errors, missing files, invalid selections, and unsupported targets exit 2.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Execution target: `auto`, `x5`, `s100`, `s100p`, or `s600`; SigLIP execution supports only S100/S100P. |
| `--variant` | str | `None` | One of the eight `VARIANTS`; when omitted, selection defaults to `base-patch16-224` unless `--asset-id` identifies another variant. |
| `--asset-id` | str | `None` | Exact qualified manifest reference, for example `s:siglip:s100/bpu-siglip-base-patch16-224.hbm`. |
| `--model-path` | str | `None` | Local HBM path; it is valid only together with the exact `--asset-id`. |
| `--test-img` | str | `samples/vision/siglip/test_data/dog.jpg` | Input BGR image path. The default is an absolute path derived from the sample location; a user-supplied relative path is resolved from the current working directory. |
| `--image-size` | int | `None` | Optional assertion; must equal the selected variant's fixed size. |
| `--submodel` | str | `pooler_output` | Selected packed submodel: `pooler_output` or `last_hidden_state`. |
| `--priority` | int | `0` | Runtime priority, constrained to 0–255. |
| `--bpu-cores` | int list | `[0]` | One or more nonnegative BPU core indexes. |
| `--output-file` | str | `None` | Optional path for the complete raw feature array in NumPy format; parent directories are created and no extension is appended. |
| `--list-models` | flag | `false` | List the eight unique manifest assets without SDK, board detection, or model loading. |
| `--dry-run` | flag | `false` | Resolve target/variant/metadata contract and print JSON without SDK or model loading; explicit S100/S100P target required on host. |

<a id="results"></a>
## Results

The CLI prints JSON fields `submodel`, `shape`, `dtype`, `mean`, `std`, `min`, `max`, and `l2_norm`. These statistics summarize the selected raw feature tensor. With `--output-file`, the complete owned NumPy array is written at exactly the supplied path. Output dtype and shape come from bound runtime metadata: `pooler_output` permits `(1,D)` or `(1,1,D)`; `last_hidden_state` is `(1,N,D)`. No dequantization, softmax, normalization, squeeze, or activation is applied.

`VARIANTS` is the source of fixed geometry and feature dimensions:

| Variant | Input size | D | N (`last_hidden_state`) |
| --- | ---: | ---: | ---: |
| `base-patch16-224` | 224 | 768 | 196 |
| `base-patch16-384` | 384 | 768 | 576 |
| `base-patch16-512` | 512 | 768 | 1024 |
| `large-patch16-256` | 256 | 1024 | 256 |
| `large-patch16-384` | 384 | 1024 | 576 |
| `so400m-patch14-224` | 224 | 1152 | 256 |
| `so400m-patch14-384` | 384 | 1152 | 729 |
| `so400m-patch16-256-i18n` | 256 | 1152 | 256 |

<a id="integration-example"></a>
## Integration Example

Prerequisite: place `bpu-siglip-base-patch16-224.hbm` at the default model path using [`model/README.md`](../../model/README.md), and run this snippet on an S100/S100P board. It defines the repository path, image path, target, variant, submodel, scheduling values, selection, runner, binding, tensors, raw output, and both explicit and composed calls.

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.siglip.runtime.python.model_binding import resolve_selection
from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.siglip.runtime.python.embedding import SigLIPTask

repo = Path.cwd()
image_path = repo / "samples/vision/siglip/test_data/dog.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "s100"
variant = "base-patch16-224"
submodel = "pooler_output"
priority = 0
bpu_cores = [0]
selection = resolve_selection(target, variant=variant, submodel=submodel)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
task = SigLIPTask(runner, binding)

prepared = task.pre_process(image)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(image)
assert np.array_equal(explicit_result, composed_result)
print({"shape": composed_result.shape, "dtype": str(composed_result.dtype)})
```

<a id="stage-io"></a>
## Three-Stage I/O

- `pre_process`: BGR `uint8` `H×W×3` → `PreparedInput`; its `_input_0` is owned contiguous RGB `float32` `(1,3,size,size)` in `[-1,1]`, and `context` stores original/resized shapes plus `(top,bottom,left,right)` padding.
- `forward`: `{"_input_0": tensor}` → raw `{"_output_0": ndarray}` for the selected packed submodel. `model_runner` validates metadata and containers but preserves native numeric dtype and values.
- `post_process`: raw output → owned ndarray with metadata-bound shape/dtype; it rejects wrong shape/dtype and NaN/Inf. This vision feature task consumes no geometry context.
- `predict(image)` composes exactly pre-process → forward → post-process. It does not download, save, activate, normalize, or evaluate results.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Model not found: ...; prepare it explicitly with model/download.sh.` | HBM is absent at the resolved path. | Prepare the exact variant under `model/`, or pass its exact `--asset-id` with `--model-path`. |
| `No published SigLIP support for x5/s600` | The selected target is outside the S100/S100P publication. | Use an S100 or S100P board/target. |
| `Host dry-run requires --target s100 or --target s100p` | Host dry-run cannot infer a board from `auto`. | Pass an explicit `--target`. |
| `image-size must be ...` | Explicit size disagrees with the fixed variant geometry. | Omit `--image-size` or use the variant's size. |
| `SigLIP output shape/dtype differs from bound metadata.` | Artifact metadata does not match the selected submodel contract. | Inspect the exact HBM and select the matching asset; do not reshape or cast the output. |

## License

The runtime code is Apache-2.0 under the repository [LICENSE](../../../../../LICENSE).
