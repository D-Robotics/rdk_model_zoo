# RepGhost conversion

<a id="source-model"></a>
## Source model

The source describes PyTorch/timm RepGhost variants but fixes no timm/torch version, weight revision or weights digest. No weights or executable ONNX export script are delivered. Correspondence with the published binaries has not been rebuilt.

<a id="toolchain-targets"></a>
## Toolchain and targets

Five unchanged source YAMLs target X5 `march: bayes-e`. The OE version is unspecified in the source and must be recorded when rebuilding. No S recipe exists.

| Variant | Config | Required ONNX | Published filename |
| --- | --- | --- | --- |
| `100` | `RepGhost_100.yaml` | `./repghostnet_100.onnx` | `RepGhost_100_224x224_nv12.bin` |
| `111` | `RepGhost_111.yaml` | `./repghostnet_111.onnx` | `RepGhost_111_224x224_nv12.bin` |
| `130` | `RepGhost_130.yaml` | `./repghostnet_130.onnx` | `RepGhost_130_224x224_nv12.bin` |
| `150` | `RepGhost_150.yaml` | `./repghostnet_150.onnx` | `RepGhost_150_224x224_nv12.bin` |
| `200` | `RepGhost_200.yaml` | `./repghostnet_200.onnx` | `RepGhost_200_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

No verified export command can be supplied from the checked-in material. Prepare the matching variant ONNX with RGB NCHW input (nominal 1×3×224×224); verify its real I/O first. YAML `input_shape` and `input_name` are empty, so they derive from the graph rather than constraining it to 224. Place the graph beside its YAML as listed above.

<a id="calibration"></a>
## Calibration

All configs use `./calibration_data_rgb_f32`, `cal_data_type: float32`, calibration `default`, RGB/NCHW training input and runtime NV12. Mean: 123.675/116.28/103.53; scale: 0.01712475/0.017507/0.01742919. No calibration-image selection, count, preprocessing script or output files are provided. These are missing prerequisites, not a runnable preparation recipe. Ensure the prepared floats match the graph and YAML normalization exactly; do not rename an NV12 directory to satisfy the RGB path.

<a id="compile"></a>
## Compile

Conditional commands only, inside an OE environment after the missing ONNX and RGB float32 calibration data are prepared. Not executed during migration. cwd is this conversion directory. Example for 100:

```bash
cd samples/vision/repghost/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repghostnet_100.onnx
hb_mapper makertbin --model-type onnx --config RepGhost_100.yaml
```

Every config writes working_dir `RepGhost_224x224_nv12` with output prefix `RepGhost_224x224_nv12` (no variant). Expected bin: `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`. Run variants in separate workspaces or preserve each output before the next build. Source compiler options are latency/O3 with dump_calibration_data; there is no safe concurrent shared output directory.

<a id="validation"></a>
## Post-conversion validation

Board validation is not-run. Only after verifying 224 geometry, packed NV12, one F32 output squeezing to (1000,), and score semantics, run a rebuilt 100 artifact with the exact reference plus its separate path:

```bash
# cwd: repository root on X5
python3 samples/vision/repghost/runtime/python/main.py --target x5 \
  --asset-id x5:repghost:RepGhost_100_224x224_nv12.bin \
  --model-path samples/vision/repghost/conversion/RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin \
  --test-img samples/vision/repghost/test_data/ibex.JPEG
```

The reference selects the contract, not a claim that rebuilt bytes equal published bytes. Save the rebuilt digest and provenance separately, compare against the source and evaluate quantization accuracy before release.

<a id="artifacts"></a>
## Artifacts

The per-variant published filenames are listed above and downloaded under `samples/vision/repghost/model/`. Compilation produces a common basename in its working directory. Preserve the selected variant identity when moving a verified build; renaming alone establishes neither graph equivalence nor accuracy.

<a id="known-gaps"></a>
## Known gaps

Missing: pinned framework/OE versions, weight revision/hash, runnable export, calibration selection/count/preparation, and conversion/board accuracy evidence. Five YAMLs and source notices are preserved byte-for-byte. Today the published model download route is available; a reproducible end-to-end conversion is not claimed.
