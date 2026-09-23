# MobileOne conversion

<a id="source-model"></a>
## Source model

The source references apple/ml-mobileone, loading an unfused checkpoint such as mobileone_s0_unfused.pth.tar and applying `reparameterize_model(model)` before ONNX export/simplification. No executable export script, upstream revision, package versions or checkpoint digest is included.

<a id="toolchain-targets"></a>
## Toolchain and targets

5 unchanged YAMLs from rdk_x5 @ac11571, X5 march `bayes-e`; no S recipe. OE version is not pinned by the source; record the actual rebuild environment.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `MobileOne_S0_config.yaml` | `./mobileone_s0.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S1_config.yaml` | `./mobileone_s1.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S2_config.yaml` | `./mobileone_s2.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S3_config.yaml` | `./mobileone_s3.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S4_config.yaml` | `./mobileone_s4.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

The source references apple/ml-mobileone, loading an unfused checkpoint such as mobileone_s0_unfused.pth.tar and applying `reparameterize_model(model)` before ONNX export/simplification. No executable export script, upstream revision, package versions or checkpoint digest is included.

There is no executable, verified export command in this sample. Prepare the matching graph at the table path, nominal RGB NCHW 1×3×224×224 input and ImageNet-1k output. YAML input_shape/input_name are empty: dimensions and names come from the graph and must be checked.

<a id="calibration"></a>
## Calibration

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

Conditional commands in OE, after the missing graph and calibration prerequisites are supplied. Not run in this migration.

```bash
# cwd: repository root, then conversion directory
cd samples/vision/mobileone/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./mobileone_s0.onnx
hb_mapper makertbin --model-type onnx --config MobileOne_S0_config.yaml
```

Expected output for this config: `MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin`. All configs use latency/O3. All variants share `MobileOne_224x224_nv12_int8` and the `MobileOne_224x224_nv12` prefix; isolate builds and preserve variant identity. YAML requests 32 compiler jobs: size the build host accordingly. No node-removal override or debug_mode is set.

<a id="validation"></a>
## Post-conversion validation

Status: not-run. Verify packed NV12 geometry 224×224 and an F32 score output squeezing to (1000,) before inference. Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/mobileone/runtime/python/main.py --target x5 \
  --asset-id x5:mobileone:MobileOne_S0_224x224_nv12.bin \
  --model-path samples/vision/mobileone/conversion/MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG
```

The qualified reference selects a contract; it does not certify that new bytes equal published bytes. Record the new hash and graph provenance, compare source/unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Preserve variant and provenance when moving a verified build; renaming alone is not a repair.

<a id="known-gaps"></a>
## Known gaps

Missing pinned framework/OE/weights, runnable export, calibration preparation and conversion/board accuracy evidence. YAML files and original notices are preserved byte-for-byte. Published artifact download is available; end-to-end conversion reproducibility is not claimed.
