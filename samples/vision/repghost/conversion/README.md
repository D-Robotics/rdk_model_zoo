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

Five unchanged source YAMLs target X5 `march: bayes-e`. The OE version is unspecified in the source and must be recorded when rebuilding.

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

No verified export command can be supplied from the checked-in material. Prepare the matching variant ONNX with RGB NCHW input (nominal 1×3×224×224); verify its real I/O first. YAML `input_shape` and `input_name` are empty, so they derive from the graph rather than constraining it to 224. Place the graph beside its YAML as listed above.

<a id="calibration"></a>
## Calibration

All configs use `./calibration_data_rgb_f32`, `cal_data_type: float32`, calibration `default`, RGB/NCHW training input and runtime NV12. Mean: 123.675/116.28/103.53; scale: 0.01712475/0.017507/0.01742919. No calibration-image selection, count, preprocessing script or output files are provided. These are missing prerequisites, not a runnable preparation recipe. Ensure the prepared floats match the graph and YAML normalization exactly; do not rename an NV12 directory to satisfy the RGB path.

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

Use the matching reference and target for each artifact. Save the build digest and provenance, compare source and rebuilt outputs, and evaluate quantization accuracy before release.

<a id="artifacts"></a>
## Artifacts

The per-variant published filenames are listed above and downloaded under `samples/vision/repghost/model/`. Compilation produces a common basename in its working directory. Preserve the selected variant identity when moving a verified build; renaming alone establishes neither graph equivalence nor accuracy.

<a id="known-gaps"></a>
## Additional preparation

For each variant, select the matching PyTorch/timm checkpoint and record its revision and digest. Export a graph beside the corresponding YAML with RGB/NCHW input; inspect its actual input and output metadata because `input_shape` and `input_name` are empty. Prepare `./calibration_data_rgb_f32` with float32 RGB values, `cal_data_type: float32`, `calibration: default`, mean `123.675/116.28/103.53`, and scale `0.01712475/0.017507/0.01742919`. Build variants in separate workspaces because each YAML writes `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`.
