# VargConvNet conversion

<a id="source-model"></a>
## Source model

The source provides a published X5 deployment binary and its wrapper, but no ONNX graph, pinned weight revision, framework version, export script or PTQ YAML.

<a id="toolchain-targets"></a>
## Toolchain and targets

X5 is the published target. No compilation configuration or OE version is provided; no S target recipe is available.

<a id="export"></a>
## ONNX export

No runnable export command exists in the source. The wrapper expects nominal RGB/NCHW 224×224 before NV12 packing and 1000 classification scores; that runtime assumption is not a training/export recipe.

<a id="calibration"></a>
## Calibration

No calibration preparation, selected data/count, normalization recipe or files were delivered. Obtain the actual model graph and matching quantization recipe before preparing calibration data.

<a id="compile"></a>
## Compile

No verified compile command can be supplied without an ONNX graph and PTQ config. Use the published artifact route in model/README; no placeholder YAML or fabricated compile success is provided.

<a id="validation"></a>
## Post-conversion validation

Board validation is not-run. Verify actual metadata, 224×224 packed NV12 and one F32 output squeezing to 1000 scores. A rebuilt model can be selected with the exact contract reference and an external path; keep its hash/provenance separate from the publisher artifact.

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py --target x5 --variant vargconvnet
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts). No ONNX/weights/compiled files are added by this migration.

<a id="known-gaps"></a>
## Known gaps

No conversion/accuracy/board evidence has been produced. Export and calibration availability are stated above per actual source, not inferred from the model name. Pin versions, weights and dataset inputs before rebuilding.
