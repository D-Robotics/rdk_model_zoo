# RepViT conversion

<a id="source-model"></a>
## Source model

The source describes `timm.models.create_model` for repvit_m0_9/m1_0/m1_1, PyTorch ONNX export and onnxsim simplification. No executable export script, timm/PyTorch version, upstream source revision or checkpoint digest is included.

<a id="toolchain-targets"></a>
## Toolchain and targets

3 unchanged YAMLs from rdk_x5 @ac11571, X5 march `bayes-e`; no S recipe. OE version is not pinned by the source; record the actual rebuild environment.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `RepViT_m0_9_config.yaml` | `./repvit_m0_9.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |
| `RepViT_m1_0_config.yaml` | `./repvit_m1_0.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |
| `RepViT_m1_1_config.yaml` | `./repvit_m1_1.onnx` | `RepViT_224x224_nv12` | `RepViT_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

The source describes `timm.models.create_model` for repvit_m0_9/m1_0/m1_1, PyTorch ONNX export and onnxsim simplification. No executable export script, timm/PyTorch version, upstream source revision or checkpoint digest is included.

There is no executable, verified export command in this sample. Prepare the matching graph at the table path, nominal RGB NCHW 1×3×224×224 input and ImageNet-1k output. YAML input_shape/input_name are empty: dimensions and names come from the graph and must be checked.

<a id="calibration"></a>
## Calibration

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

Conditional commands in OE, after the missing graph and calibration prerequisites are supplied. Not run in this migration.

```bash
# cwd: repository root, then conversion directory
cd samples/vision/repvit/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repvit_m0_9.onnx
hb_mapper makertbin --model-type onnx --config RepViT_m0_9_config.yaml
```

Expected output for this config: `RepViT_224x224_nv12/RepViT_224x224_nv12.bin`. All configs use latency/O3. All variants share the same working directory and output basename, without a variant suffix; isolate builds to avoid overwriting. YAML removes Quantize/Dequantize/Transpose/Cast/Reshape nodes. Validate the transformed graph rather than assuming these removals preserve semantics for an arbitrary new export.

<a id="validation"></a>
## Post-conversion validation

Status: not-run. Verify packed NV12 geometry 224×224 and an F32 score output squeezing to (1000,) before inference. Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/repvit/runtime/python/main.py --target x5 \
  --asset-id x5:repvit:RepViT_m0_9_224x224_nv12.bin \
  --model-path samples/vision/repvit/conversion/RepViT_224x224_nv12/RepViT_224x224_nv12.bin \
  --test-img samples/vision/repvit/test_data/yurt.JPEG
```

The qualified reference selects a contract; it does not certify that new bytes equal published bytes. Record the new hash and graph provenance, compare source/unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Preserve variant and provenance when moving a verified build; renaming alone is not a repair.

<a id="known-gaps"></a>
## Known gaps

Missing pinned framework/OE/weights, runnable export, calibration preparation and conversion/board accuracy evidence. YAML files and original notices are preserved byte-for-byte. Published artifact download is available; end-to-end conversion reproducibility is not claimed.
