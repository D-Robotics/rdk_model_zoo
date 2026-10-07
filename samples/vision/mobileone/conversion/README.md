# MobileOne conversion

<a id="source-model"></a>
## Source model

Use the upstream apple/ml-mobileone flow: load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, then export and simplify ONNX. Record the selected upstream revision, package versions and checkpoint digest for each build.

<a id="toolchain-targets"></a>
## Toolchain and targets

5 unchanged YAMLs X5 march `bayes-e`;  OE version is not pinned by the source; record the actual rebuild environment.

| Config | ONNX input path | Working directory | Compiled basename |
| --- | --- | --- | --- |
| `MobileOne_S0_config.yaml` | `./mobileone_s0.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S1_config.yaml` | `./mobileone_s1.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S2_config.yaml` | `./mobileone_s2.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S3_config.yaml` | `./mobileone_s3.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |
| `MobileOne_S4_config.yaml` | `./mobileone_s4.onnx` | `MobileOne_224x224_nv12_int8` | `MobileOne_224x224_nv12.bin` |


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

Use the upstream apple/ml-mobileone flow: load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, then export and simplify ONNX. Record the selected upstream revision, package versions and checkpoint digest for each build.

Export the matching graph to the table path with nominal RGB/NCHW 1×3×224×224 input and ImageNet-1k output. YAML `input_shape` and `input_name` are empty, so inspect the graph for its actual dimensions and names.

<a id="calibration"></a>
## Calibration

All YAMLs require `./calibration_data_rgb_f32` (float32, calibration default), RGB/NCHW training input and NV12 runtime input. Mean 123.675/116.28/103.53; scale 0.01712475/0.017507/0.01742919. Dataset selection/count, preparation script and actual calibration files are missing. Check prepared float semantics against the graph and normalization; changing an NV12 directory name does not produce RGB calibration data.

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and calibration data
prerequisites are supplied:

```bash
# cwd: repository root, then conversion directory
cd samples/vision/mobileone/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./mobileone_s0.onnx
hb_mapper makertbin --model-type onnx --config MobileOne_S0_config.yaml
```

Expected output for this config: `MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin`. All configs use latency/O3. All variants share `MobileOne_224x224_nv12_int8` and the `MobileOne_224x224_nv12` prefix; isolate builds and preserve variant identity. YAML requests 32 compiler jobs: size the build host accordingly. No node-removal override or debug_mode is set.

<a id="validation"></a>
## Post-conversion validation

Before inference, verify the packed NV12 geometry 224×224 and an F32
score output squeezing to (1000,). Example for the first variant:

```bash
# cwd: repository root on X5
python3 samples/vision/mobileone/runtime/python/main.py --target x5 \
  --asset-id x5:mobileone:MobileOne_S0_224x224_nv12.bin \
  --model-path samples/vision/mobileone/conversion/MobileOne_224x224_nv12_int8/MobileOne_224x224_nv12.bin \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG
```

Use the qualified reference that matches the artifact and target. Record the build hash and graph provenance, compare source and unified raw outputs, and evaluate accuracy before delivery.

<a id="artifacts"></a>
## Artifacts

Compiled paths are listed above. Published files and per-variant targets are listed in [model preparation](../model/README.md#artifacts); downloads land in the sample model directory. Preserve variant and provenance when moving a verified build; renaming alone is not a repair.

<a id="known-gaps"></a>
## Additional preparation

For a MobileOne rebuild, load the matching unfused checkpoint (for example `mobileone_s0_unfused.pth.tar`), apply `reparameterize_model(model)`, and export/simplify an RGB/NCHW 1×3×224×224 graph at the path in the selected YAML. Prepare float32 RGB calibration data at `./calibration_data_rgb_f32` using the YAML mean `123.675/116.28/103.53`, scale `0.01712475/0.017507/0.01742919`, and `default` calibration. Use the X5 `bayes-e` configurations and record the actual framework and OE versions when building.
