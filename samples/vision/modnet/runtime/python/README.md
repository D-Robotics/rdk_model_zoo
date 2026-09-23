# MODNet Python runtime

<a id="environment"></a>
## Environment

Use Python 3, NumPy, and OpenCV on an RDK X5 image with `hbm_runtime`. The runtime lazily imports the board SDK after model-file and board checks. `--help`, `--list-models`, and `--dry-run` are SDK-free. The bound model must expose float32 input `(1,3,512,512)` and float32 output `(1,1,512,512)`.

<a id="usage"></a>
## Usage

After manually placing the exact external asset at `samples/vision/modnet/model/modnet_512x512_rgb.bin`, run from the repository root:

```bash
python3 -m samples.vision.modnet.runtime.python.main --target x5 \
  --asset-id x5:modnet:modnet_512x512_rgb.bin
```

Success is exit code `0`, `matte_path` in the JSON, and a written uint8 matte. The default background is composited to `test_data/result.png` when readable. `bash samples/vision/modnet/runtime/python/run.sh --target x5 --asset-id x5:modnet:modnet_512x512_rgb.bin` is equivalent.

<a id="parameters"></a>
## Parameters

| option | default | meaning |
|---|---|---|
| `--target` | `auto` | resolves the only published target, X5 |
| `--asset-id` | `null` | exact manual identity, required with external model path |
| `--model-path` | `null` | existing external model path; no download |
| `--test-img` | `samples/vision/modnet/test_data/person.jpg` | BGR input image |
| `--bg-img` | `samples/vision/modnet/test_data/bg.jpg` | optional BGR background; missing file skips composite |
| `--matte-save-path` | `samples/vision/modnet/test_data/matte.png` | uint8 grayscale output |
| `--img-save-path` | `samples/vision/modnet/test_data/result.png` | composite output |
| `--priority` | `0` | runtime scheduling priority |
| `--bpu-cores` | `[0]` | BPU core indexes |
| `--ref-size` | `512` | fixed compiled input size; other values are rejected |
| `--list-models` | `false` | list manifest facts without SDK |
| `--dry-run` | `false` | print contract without SDK or model load |

`--list-models` and `--dry-run` are mutually exclusive. User/runtime errors return `2`.

<a id="results"></a>
## Results

`MODNetTask.post_process` returns an owned uint8 grayscale matte in the original image geometry. Optional compositing uses the source linear alpha formula and writes a BGR image. The raw forward result remains float32 `[0,1]` and is not saved or normalized by the runner.

<a id="integration-example"></a>
## Integration example

After the manual model and local images exist, this complete example defines every variable and checks explicit stages against `predict`:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.modnet.runtime.python.model_binding import resolve_selection
from samples.vision.modnet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.modnet.runtime.python.modnet import MODNetTask
from samples.vision.modnet.runtime.python.visualization import composite

target = "x5"
asset_id = "x5:modnet:modnet_512x512_rgb.bin"
model_path = Path("samples/vision/modnet/model/modnet_512x512_rgb.bin")
image_path = Path("samples/vision/modnet/test_data/person.jpg")
background_path = Path("samples/vision/modnet/test_data/bg.jpg")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = MODNetTask(runner, binding)
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
prepared = task.pre_process(image)
raw_matte = task.forward(prepared.tensors)
explicit_matte = task.post_process(raw_matte, prepared.context)
composed_result = composite(image, explicit_matte, cv2.imread(str(background_path)))
assert np.array_equal(explicit_matte, task.predict(image))
print(explicit_matte.shape, composed_result.shape)
```

<a id="stage-io"></a>
## Stage I/O

- `pre_process(image)` validates BGR HWC input, converts BGR→RGB, normalizes `(pixel-127.5)/127.5`, resizes the long side to 512 with centered zero padding, and returns `PreparedInput(tensors, context)`.
- `forward(tensors)` validates the bound tensor and returns an owned raw float32 `(1,1,512,512)` matte.
- `post_process(raw, context)` scales the source `[0,1]` matte to uint8, removes padding, and resizes to the original geometry.
- `predict(image)` composes all three stages; geometry lives in the current call's frozen context, not a mutable task field.

<a id="troubleshooting"></a>
## Troubleshooting

- A model path without the exact `--asset-id x5:modnet:modnet_512x512_rgb.bin` is rejected.
- A missing manual model or unreadable input returns `2` before board inference.
- Metadata with wrong names, shape, or dtype is rejected; no runtime cast hides a mismatch.
- `--ref-size` must remain 512 because no alternative compiled configuration is published.
