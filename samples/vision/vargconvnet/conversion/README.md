# VargConvNet conversion

<a id="source-model"></a>
## Source model

The published X5 binary is the inference artifact. A replacement build starts from an ONNX graph, matching weight revision, framework environment and PTQ YAML.

<a id="toolchain-targets"></a>
## Toolchain and targets

The published target is X5; compile replacement models with an X5 OE toolchain and matching PTQ configuration.


Toolchain resources:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX export

The wrapper expects nominal RGB/NCHW 224×224 input before NV12 packing and returns 1,000 classification scores. Export a matching graph with this I/O contract.

<a id="calibration"></a>
## Calibration

Prepare calibration data from the selected model inputs and normalization recipe, then configure the matching PTQ settings.

<a id="compile"></a>
## Compile

For inference, prepare the published artifact as described in [model/README.md](../model/README.md). A replacement build needs an ONNX graph and PTQ configuration for the X5 OE toolchain.

<a id="validation"></a>
## Post-conversion validation

Verify actual metadata, 224×224 packed NV12 and one F32 output squeezing to 1000 scores. A rebuilt model can be selected with the exact contract reference and an external path; keep its hash/provenance separate from the publisher artifact.

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py --target x5 --variant vargconvnet
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts).

<a id="known-gaps"></a>
## Additional preparation

For inference, prepare the published X5 artifact through [model/README.md](../model/README.md). A replacement build requires an ONNX graph matching the wrapper's nominal RGB/NCHW 224×224 input and 1,000 classification scores, a matching PTQ configuration, and calibration data following that graph's normalization. Use the X5 OE toolchain and validate the compiled model through the sample runtime.
