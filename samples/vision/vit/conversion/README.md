# ViT conversion

<a id="source-model"></a>
## Source model

Source describes PyTorch CIFAR-10 training and `vit_cifar10_batch1.onnx`, referring to [ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git). No fixed weight revision or export script is delivered. The YAML and compiler log describe the S100 build settings and output.

<a id="toolchain-targets"></a>
## Toolchain and targets

The recorded build environment of the original log is hbdk 4.2.11 /
hmct 2.4.1 / hb_compile 3.3.22. YAML march=nash-e targets S100; no
configuration is provided for S100P/S600/X5. Use a matching x86 Linux OE
environment ([OE resources](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview),
[toolchain manual](https://toolchain.d-robotics.cc/)).

<a id="export"></a>
## Export

No export script or fixed weight revision is included. To rebuild, supply
the trained weights, model definition and a compatible export environment
(follow [ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git) for
the training flow) and place the resulting ONNX at
`conversion/vit_cifar10_batch1.onnx`.

<a id="calibration"></a>
## Calibration

The recipe uses 50 CIFAR-10 calibration images as float32 RGB NCHW data
under `calibration_data_rgb/` (prepare the directory yourself). YAML
mean=0.4914/0.4822/0.4465 and
scale=4.943153707865546/5.014042553191489/4.975124378109453; softmax
qtype=int32. Before generating data, confirm the encoding (NumPy header
vs raw buffer), the value range, and the OE loader path: the original log
reads raw calibration_*.npy from a different absolute directory.

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and calibration data
prerequisites are supplied:

```bash
cd samples/vision/vit/conversion
hb_compile --config config_vit_nv12.yaml
```

Expected configured output: `vit_cifar10_batch1/vit_cifar10_batch1.hbm`; jobs=8, latency, O2. Inspect the full log and successful exit.

<a id="validation"></a>
## Validation

The original log records Y [1,224,224,1] U8, UV [1,112,112,2] U8, output
[1,10] F32. Check the actual regenerated metadata, then compare numerically
on the same board and evaluate CIFAR-10 accuracy.

<a id="artifacts"></a>
## Artifacts

One YAML produces an unsuffixed HBM. Published int8/int16 artifacts are separate entries in [model](../model/README.md). The provided recipe does not establish how each was built; renaming its output is not proof of an int16 build.

<a id="known-gaps"></a>
## Additional preparation

For a ViT rebuild, start from the linked CIFAR-10 training flow and a pinned weight revision, then export the ONNX graph used by the S100 YAML. Prepare calibration values after checking the encoding and value range expected by the original YAML and log. Build and validate the int8 and int16 artifacts as separate S100 outputs; the YAML command shown above describes its configured output, and each deployment file must match its published variant identity. Use the model downloader for inference with published files.
