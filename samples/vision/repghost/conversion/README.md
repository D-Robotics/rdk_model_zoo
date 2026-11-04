English | [简体中文](README_cn.md)

# RepGhost conversion

<a id="source-model"></a>
## Source model

Use the matching PyTorch/timm RepGhost checkpoint and record its version, revision and digest before exporting each variant.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── RepGhost_100.yaml  # Configuration
├── RepGhost_111.yaml  # Configuration
├── RepGhost_130.yaml  # Configuration
├── RepGhost_150.yaml  # Configuration
└── RepGhost_200.yaml  # Configuration
```

<a id="toolchain-targets"></a>
## Toolchain and targets

Five YAML configurations target X5 `march: bayes-e`. Record the OE version used for compilation.

| Variant | Config | Required ONNX | Published filename |
| --- | --- | --- | --- |
| `100` | `RepGhost_100.yaml` | `./repghostnet_100.onnx` | `RepGhost_100_224x224_nv12.bin` |
| `111` | `RepGhost_111.yaml` | `./repghostnet_111.onnx` | `RepGhost_111_224x224_nv12.bin` |
| `130` | `RepGhost_130.yaml` | `./repghostnet_130.onnx` | `RepGhost_130_224x224_nv12.bin` |
| `150` | `RepGhost_150.yaml` | `./repghostnet_150.onnx` | `RepGhost_150_224x224_nv12.bin` |
| `200` | `RepGhost_200.yaml` | `./repghostnet_200.onnx` | `RepGhost_200_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use the ONNX export example below. Prepare the matching variant ONNX with RGB NCHW input (nominal 1×3×224×224); verify its real I/O first. YAML `input_shape` and `input_name` are empty, so they derive from the graph rather than constraining it to 224. Place the graph beside its YAML as listed above.

### ONNX export example

Run this example in the x86 export environment with PyTorch installed. The simplification step also requires `onnxsim`. Match the exported filename to `onnx_model` in the selected YAML.

The onnx model is transformed using models from the timm library (PyTorch Image Models). Install the required packages using the following command:

```shell
pip install timm onnx
```

The following command fetches MobileOne for a comparison model. The RepGhost export below uses `timm.models.repghost`:

```shell
git clone https://github.com/apple/ml-mobileone.git
```

Model transformation takes repghostnet_100 as an example, and the other four variants are the same:

```Python
import torch
import torch.onnx
import onnx
from onnxsim import simplify
from timm.models import create_model

from timm.models.repghost import repghostnet_100, repghostnet_111, repghostnet_130, repghostnet_150, repghostnet_200

def count_parameters(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    # Get the initializers (weights in the model)
    initializer = model.graph.initializer

    # Calculate the total number of parameters
    total_params = 0
    for tensor in initializer:
        # Get the dimensions of each weight
        dims = tensor.dims
        # Calculate the number of parameters in this weight (product of all dimensions)
        params = 1
        for dim in dims:
            params *= dim
        total_params += params

    return total_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('repghostnet_100', pretrained=True)
    model.eval()

    # print the model structure

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "repghostnet_100.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # Input name
        output_names=["output"],  # Output name
    )

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "repghostnet_100.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    onnx_model_path = simplified_onnx_file_path  # Replace with your ONNX model path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
```

<a id="calibration"></a>
## Calibration

All configs use `./calibration_data_rgb_f32`, `cal_data_type: float32`, calibration `default`, RGB/NCHW training input and runtime NV12. Mean: 123.675/116.28/103.53; scale: 0.01712475/0.017507/0.01742919. Select calibration images and generate RGB float data that matches the graph and YAML normalization. This directory does not include a calibration preprocessing script or dataset.

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and RGB float32 calibration
data are prepared (cwd is this conversion directory). Example for 100:

```bash
cd samples/vision/repghost/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repghostnet_100.onnx
hb_mapper makertbin --model-type onnx --config RepGhost_100.yaml
```

Every config writes working_dir `RepGhost_224x224_nv12` with output prefix `RepGhost_224x224_nv12` (no variant). Expected bin: `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`. Run variants in separate workspaces or preserve each output before the next build. Source compiler options are latency/O3 with dump_calibration_data; there is no safe concurrent shared output directory.

<a id="validation"></a>
## Post-conversion validation

Only after verifying 224 geometry, packed NV12, one F32 output squeezing to (1000,), and score semantics, run a rebuilt 100 artifact with the exact reference plus its separate path:

```bash
# cwd: repository root on X5
python3 samples/vision/repghost/runtime/python/main.py --target x5 \
  --asset-id x5:repghost:RepGhost_100_224x224_nv12.bin \
  --model-path samples/vision/repghost/conversion/RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin \
  --test-img samples/vision/repghost/test_data/ibex.JPEG
```

Select the artifact's matching target and reference, then compare its outputs and dataset accuracy with the float model.

<a id="artifacts"></a>
## Artifacts

The per-variant published filenames are listed above and downloaded under `samples/vision/repghost/model/`. Compilation produces a common basename in its working directory. Preserve the selected variant identity when moving a verified build; record the compilation configuration and evaluation results with the model.

<a id="known-gaps"></a>
## Additional preparation

For each variant, select the matching PyTorch/timm checkpoint and record its revision and digest. Export a graph beside the corresponding YAML with RGB/NCHW input; inspect its actual input and output metadata because `input_shape` and `input_name` are empty. Prepare `./calibration_data_rgb_f32` with float32 RGB values, `cal_data_type: float32`, `calibration: default`, mean `123.675/116.28/103.53`, and scale `0.01712475/0.017507/0.01742919`. Build variants in separate workspaces because each YAML writes `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`.

<!-- converted-batch-20261102 -->
## Pinned converted recipes

[Shared export/calibration instructions](../../../../utils/tools/classification_conversion/README.md).

```bash
python3 utils/tools/classification_conversion/export.py --model repghostnet_100.in1k --checkpoint /path/to/pinned/model.safetensors --calibration-sample /path/to/run/calibration_rgb_f32/0000.rgb --output /path/to/run/model.onnx
cp samples/vision/repghost/conversion/recipes/timm-100-in1k_x5.yaml /path/to/run/compile.yaml
# In the X5 OE container, cwd /path/to/run:
hb_mapper checker --model model.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
```

Use the model/tag, pinned checkpoint and matching YAML from `converted.json`; use a fresh attempt directory. The calibration producer uses raw RGB float32 binary tensors. Do not apply normalization twice or inherit historical board metrics.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

Each target uses its own PTQ, BC, full NodeInfo and final OE. Internal INT16 settings do not determine the public ABI; these profiles bind the observed float32 outputs. Recipes and hashes are in [converted_targets_20261103.json](converted_targets_20261103.json).

- `s100 / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-100-in1k-s-int16-o1-20261103 / model`: [timm_100_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_100_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-111-in1k-s-int16-o1-20261103 / model`: [timm_111_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_111_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-130-in1k-s-int16-o1-20261103 / model`: [timm_130_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_130_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-150-in1k-s-int16-o1-20261103 / model`: [timm_150_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_150_in1k_s_int16_o1_20261103_s600_model.yaml)
- `s100 / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s100_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s100_model.yaml)
- `s100p / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s100p_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s100p_model.yaml)
- `s600 / timm-200-in1k-s-int16-o1-20261103 / model`: [timm_200_in1k_s_int16_o1_20261103_s600_model.yaml](recipes/timm_200_in1k_s_int16_o1_20261103_s600_model.yaml)

S PTQ consumes normalized ONNX-domain `.npy` tensors. NV12 runtime mean/scale remain in the recipe. The shared producer reads checkpoint normalization through `--target` and `--normalization-json`.

[Shared classification workflow](../../../../utils/tools/classification_conversion/README.md).

The original float tolerances, real inputs and checkpoint identity are retained. Compiler estimates and first-calibration-input NodeInfo diagnostics are separate from dataset accuracy and board performance.
