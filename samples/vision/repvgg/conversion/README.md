# RepVGG conversion

<a id="source-model"></a>
## Source model

The source describes official RepVGG training checkpoints, `create_RepVGG_B1g2(deploy=False)` and `repvgg_model_convert()` before ONNX export. No executable export script, source revision, PyTorch version or checkpoint digest is included. Reparameterization must be completed before compiling a training-form checkpoint.

<a id="toolchain-targets"></a>
## Toolchain and targets

6 unchanged YAMLs from rdk_x5 @ac11571, X5 march `bayes-e`; no S recipe. OE version is not pinned by the source; record the actual rebuild environment.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepVGG_A0_config.yaml` | `./RepVGG-A0.onnx` | `RepVGG-A0_224x224_nv12` | `RepVGG-A0_224x224_nv12.bin` |
| `RepVGG_A1_config.yaml` | `./RepVGG-A1.onnx` | `RepVGG-A1_224x224_nv12` | `RepVGG-A1_224x224_nv12.bin` |
| `RepVGG_A2_config.yaml` | `./RepVGG-A2.onnx` | `RepVGG-A2_224x224_nv12` | `RepVGG-A2_224x224_nv12.bin` |
| `RepVGG_B0_config.yaml` | `./RepVGG-B0.onnx` | `RepVGG-B0_224x224_nv12` | `RepVGG-B0_224x224_nv12.bin` |
| `RepVGG_B1g2_config.yaml` | `./RepVGG-B1g2.onnx` | `RepVGG-B1g2_224x224_nv12` | `RepVGG-B1g2_224x224_nv12.bin` |
| `RepVGG_B1g4_config.yaml` | `./RepVGG-B1g4.onnx` | `RepVGG-B1g4_224x224_nv12` | `RepVGG-B1g4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

The source describes official RepVGG training checkpoints, `create_RepVGG_B1g2(deploy=False)` and `repvgg_model_convert()` before ONNX export. No executable export script, source revision, PyTorch version or checkpoint digest is included. Reparameterization must be completed before compiling a training-form checkpoint.

There is no executable, verified export command in this sample. Prepare the matching graph at the table path, nominal RGB NCHW 1×3×224×224 input and ImageNet-1k output. YAML input_shape/input_name are empty: dimensions and names come from the graph and must be checked.

<a id="calibration"></a>
## Calibration

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

Conditional commands in OE, after the missing graph and calibration prerequisites are supplied. Not run in this migration.

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvgg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./RepVGG-A0.onnx
hb_mapper makertbin --model-type onnx --config RepVGG_A0_config.yaml
```

Expected output for this config: `RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin`. All configs use latency/O3. Each variant has a distinct directory/prefix, using `RepVGG-A0` etc. Compiled basenames use a hyphen while the published filenames use an underscore. Every YAML also removes Quantize/Dequantize/Transpose/Cast/Reshape nodes and enables dump_calibration_data. Verify graph semantics after these transformations; filename changes do not prove equivalence.

<a id="validation"></a>
## Post-conversion validation

Status: not-run. Verify packed NV12 geometry 224×224 and an F32 score output squeezing to (1000,) before inference. Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/repvgg/runtime/python/main.py --target x5 \
  --asset-id x5:repvgg:RepVGG_A0_224x224_nv12.bin \
  --model-path samples/vision/repvgg/conversion/RepVGG-A0_224x224_nv12/RepVGG-A0_224x224_nv12.bin \
  --test-img samples/vision/repvgg/test_data/gooze.JPEG
```

The qualified reference selects a contract; it does not certify that new bytes equal published bytes. Record the new hash and graph provenance, compare source/unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Preserve variant and provenance when moving a verified build; renaming alone is not a repair.

<a id="known-gaps"></a>
## Known gaps

Missing pinned framework/OE/weights, runnable export, calibration preparation and conversion/board accuracy evidence. YAML files and original notices are preserved byte-for-byte. Published artifact download is available; end-to-end conversion reproducibility is not claimed.
