# ResNeXt conversion

<a id="source-model"></a>
## Source model

ResNeXt source references timm resnext50_32x4d and ONNX simplification, without executable export, pinned package/weight versions or checksums.

<a id="toolchain-targets"></a>
## Toolchain and targets

X5 march bayes-e; the source files are preserved byte-for-byte. OE version is unspecified.

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `ResNeXt50_32x4d_config.yaml` | `./ResNeXt50_32x4d.onnx` | `ResNeXt50_32x4d_224x224_nv12/ResNeXt50_32x4d_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

No executable export script is delivered. Prepare `ResNeXt50_32x4d.onnx` next to its YAML, RGB NCHW 1×3×224×224 nominal input; input_shape/name are empty and read from the graph.

<a id="calibration"></a>
## Calibration

YAML expects RGB/NCHW training input, NV12 runtime input; mean 123.675/116.28/103.53, scale 0.01712475/0.017507/0.01742919. ResNeXt uses ./calibration_data_rgb_f32, float32, calibration default; no preparation/count/data are provided.

<a id="compile"></a>
## Compile

Only after graph and calibration prerequisites are verified, run in OE (not run here):

```bash
# cwd: repository root
cd samples/vision/resnext/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./ResNeXt50_32x4d.onnx
hb_mapper makertbin --model-type onnx --config ResNeXt50_32x4d_config.yaml
```

latency/O3 is preserved. Output basenames match the published filenames; that naming match does not prove equal graph, bytes or accuracy.

<a id="validation"></a>
## Post-conversion validation

Board validation is not-run. Verify actual metadata, 224×224 packed NV12 and one F32 output squeezing to 1000 scores. A rebuilt model can be selected with the exact contract reference and an external path; keep its hash/provenance separate from the publisher artifact.

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant 50_32x4d
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts). No ONNX/weights/compiled files are added by this migration.

<a id="known-gaps"></a>
## Known gaps

No conversion/accuracy/board evidence has been produced. Export and calibration availability are stated above per actual source, not inferred from the model name. Pin versions, weights and dataset inputs before rebuilding.
