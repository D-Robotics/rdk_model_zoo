English | [简体中文](README_cn.md)

# OSNet Python feature extraction

<a id="overview"></a>
## Overview

Extract a 512-dimensional feature from one decoded RGB person crop. CLI artifact selection is separate from the model API.

<a id="directory"></a>
## Directory

main.py constructs the model and calls predict; cli.py handles options/input/output; matching.py owns all three numerical stages; run.sh forwards arguments.

<a id="environment"></a>
## Environment

Use an X5 image supplying hbm_runtime compatible with libdnn 1.24.5 and HBRT 3.15.55.0, the recorded artifact environment. The original OS image version was not recorded. Python ≥3.10, NumPy and Pillow ≥9.1 are required; conversion also needs PyYAML. Preparation uses about 40 MB for the recovery TAR, BINs and fixed inputs. Use an existing environment; these scripts do not install dependencies. hbm_runtime is supplied only by the matching board image. Requirements are in ../../requirements-host.txt. If preparing an isolated user environment, the explicit command is:

```bash
python3 -m pip install -r samples/vision/osnet/requirements-host.txt
```


S-series board testing used UCP/DNN 3.13.6 and HBRT 4.7.5, with board-supplied hbm_runtime, NumPy and Pillow (S100 Pillow 9.0.1 is supported; host export dependencies are in requirements-host.txt). S100/S100P used OS 4.0.5-Beta; S600 used 5.1.0.

<a id="usage"></a>
## Usage

Run at repository root on X5 after preparing the matching BIN and --fixtures. Zero extra arguments use the first fixed crop; a user image can be selected explicitly. Exit 0 and a saved (512,) float32 feature indicate successful execution.

```bash
python3 samples/vision/osnet/runtime/python/main.py
python3 samples/vision/osnet/runtime/python/main.py --target x5 \
  --variant spatial-rgb-int16 --test-img /absolute/path/person_crop.jpg --output person.npy
python3 samples/vision/osnet/runtime/python/main.py --dry-run
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| --target | str | x5 | x5 / s100 / s100p / s600 / auto |
| --variant | str | spatial-rgb-late8 | Registered model variant |
| --model-path | Path | null | Registered BIN/HBM filename |
| --test-img | Path | null | Defaults to the first privately prepared fixture crop |
| --output | Path | feature.npy | NumPy feature file |
| --raw | bool | false | Disable L2 normalization |
| --dry-run | bool | false | Show selection without the SDK |
| --list-models | bool | false | List model identities without the SDK |

<a id="results"></a>
## Results

The API returns an owned NumPy float32 (512,) array. Default L2 normalization uses max(norm,1e-12); --raw leaves raw values. No softmax, class ID or detection box is returned. np.save adds .npy if the chosen name lacks it.

<a id="integration-example"></a>
## Integration example

Prerequisites: the prepared default BIN and images.dat. Run this complete snippet from repository root on X5:

```python
from pathlib import Path
from samples.vision.osnet.runtime.python.cli import read_image
from samples.vision.osnet.runtime.python.matching import OSNet, OSNetConfig

image = read_image()  # first crop from prepared test_data/fixture/images.dat
model = OSNet(OSNetConfig(
    Path("samples/vision/osnet/model/spatial-rgb-late8.bin"), target="x5"))
feature = model.predict(image)
prepared = model.preprocess(image)
raw = model.infer(prepared)
same_feature = model.postprocess(raw)
print(feature.shape, same_feature.shape)
```

<a id="stage-io"></a>
## Stage I/O

Input is Pillow image or H×W×3 uint8 RGB (convert OpenCV BGR first). preprocess uses Pillow bilinear resize to 128×256, subtracts 128, packs int8 CHW planes and reshapes the bytes to metadata [1,256,128,3]. It returns a nested model/input mapping. Context is absent because whole-crop embeddings need no coordinate reconstruction. infer calls the SDK and returns its nested feature mapping. postprocess requires 512 finite float32 values, makes an owned copy and optionally L2-normalizes. This signed planar RGB contract is specific to the compiled artifacts, despite NHWC metadata.

<a id="troubleshooting"></a>
## Troubleshooting

Missing model/fixture: run model/download.py with --fixtures. Hash mismatch: choose the matching --variant and artifact. Unknown board or wrong target: execute on the matching detected X5/S-series board. Wrong shape/type: use the RGB artifact and compatible SDK. The library accepts a custom model path only when its metadata and physical signed-planar contract match; CLI identities remain pinned. Model stages never download, install, print or save.

S-series preprocess returns F32 NCHW [1,3,256,128]: bilinear RGB resize, /255, mean=[.485,.456,.406], std=[.229,.224,.225]. X5 retains the signed-planar contract above. See [S-series commands](../../evaluator/S_SERIES.md).
