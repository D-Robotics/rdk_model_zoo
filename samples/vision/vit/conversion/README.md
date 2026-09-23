# ViT conversion

<a id="source-model"></a>
## Source model

Source describes PyTorch CIFAR-10 training and `vit_cifar10_batch1.onnx`, referring to [ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git). No fixed weight revision or export script is delivered. The original YAML and hb_compile.log are preserved byte-for-byte.

<a id="toolchain-targets"></a>
## Toolchain and targets

The historical log records hbdk 4.2.11 / hmct 2.4.1 / hb_compile 3.3.22. This is a historical environment, not a newly qualified container. YAML march=nash-e targets S100. No configuration is provided for S100P/S600/X5. Use a matching x86 Linux OE environment; [OE resources](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview).

<a id="export"></a>
## Export

Not reproducible from delivered files alone: first supply the exact trained weights, model definition, compatible export environment and resulting ONNX. Place it at `conversion/vit_cifar10_batch1.onnx`. No untested generic export command is presented as a working recipe.

<a id="calibration"></a>
## Calibration

Source reports 50 CIFAR-10 calibration images, float32 RGB NCHW data under `calibration_data_rgb/`. No preparation script/data is included. YAML mean=0.4914/0.4822/0.4465 and scale=4.943153707865546/5.014042553191489/4.975124378109453; softmax qtype=int32. Confirm encoding, value range and OE loader before generating data: the historical log reads raw calibration_*.npy from a different absolute directory. The extension alone does not prove NumPy-header vs raw-buffer encoding.

<a id="compile"></a>
## Compile

Conditional on the missing export/calibration prerequisites and matching OE environment; not executed.

```bash
cd samples/vision/vit/conversion
hb_compile --config config_vit_nv12.yaml
```

Expected configured output: `vit_cifar10_batch1/vit_cifar10_batch1.hbm`; jobs=8, latency, O2. Inspect the full log and successful exit.

<a id="validation"></a>
## Validation

Historical log: Y [1,224,224,1] U8, UV [1,112,112,2] U8, output [1,10] F32. Check actual regenerated metadata, then same-board source/unified numerical comparison and CIFAR-10 accuracy. Current conversion/board validation: not-run.

<a id="artifacts"></a>
## Artifacts

One YAML produces an unsuffixed HBM. Published int8/int16 artifacts are separate entries in [model](../model/README.md). The provided recipe does not establish how each was built; renaming its output is not proof of an int16 build.

<a id="known-gaps"></a>
## Known gaps

Missing fixed weight/export revisions, calibration preparation/data encoding, int8/int16 recipe correspondence, current toolchain/container validation and regenerated accuracy. Preserve historical log path differences instead of rewriting evidence. Prebuilt-artifact inference remains the supported implementation route, with board validation pending.
