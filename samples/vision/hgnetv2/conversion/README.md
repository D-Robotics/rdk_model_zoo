# HGNetV2 conversion

<a id="source-model"></a>
## Source model

Five original export scripts load timm hgnetv2_b0…b4.ssld_stage2_ft_in1k pretrained weights. The source names torch 1.13 and OE Docker v1.2.8; script comments name hb_mapper 1.24.3/opset 11. timm and remote weight revisions/hashes are unpinned. First execution can download weights; no export is run by inference.

<a id="toolchain-targets"></a>
## Toolchain and targets

X5 march bayes-e; the source files are preserved byte-for-byte. The source v1.2.8 environment has not been rerun here.

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `hgnetv2_b0.yaml` | `./onnx_export/hgnetv2_b0.onnx` | `hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin` |
| `hgnetv2_b1.yaml` | `./onnx_export/hgnetv2_b1.onnx` | `hgnetv2_b1_224x224_nv12/hgnetv2_b1_224x224_nv12.bin` |
| `hgnetv2_b2.yaml` | `./onnx_export/hgnetv2_b2.onnx` | `hgnetv2_b2_224x224_nv12/hgnetv2_b2_224x224_nv12.bin` |
| `hgnetv2_b3.yaml` | `./onnx_export/hgnetv2_b3.onnx` | `hgnetv2_b3_224x224_nv12/hgnetv2_b3_224x224_nv12.bin` |
| `hgnetv2_b4.yaml` | `./onnx_export/hgnetv2_b4.onnx` | `hgnetv2_b4_224x224_nv12/hgnetv2_b4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

Conditional export command in the source OE/PyTorch environment. Run from onnx_export so outputs match YAML paths. Replace b0 with b1/b2/b3/b4 for the other scripts. The scripts use input name input, output name output, shape 1×3×224×224 and opset 11. Not executed during migration.

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion/onnx_export
python3 export_hgnetv2_b0_bpu.py
# output: hgnetv2_b0.onnx in this directory
```

<a id="calibration"></a>
## Calibration

YAML sets cal_data_dir ../cal_data, cal_data_type float32 and preprocess_on true; training input RGB/NCHW with mean 123.675/116.28/103.53 and scale 0.01712475/0.017507/0.01742919. The old README suggests copying 20–50 JPEGs there, but no preparation script or conversion receipt proves how these JPEGs satisfy the float32/preprocess_on combination. Treat this as an unresolved prerequisite: verify the exact OE loader and normalization before calibration; do not claim that copying or renaming images completes it. The path resolves to samples/vision/hgnetv2/cal_data from conversion cwd.

<a id="compile"></a>
## Compile

Only after graph and calibration prerequisites are verified, run in OE (not run here):

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx_export/hgnetv2_b0.onnx
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```

latency/O3 is preserved. Output basenames match the published filenames; that naming match does not prove equal graph, bytes or accuracy.

<a id="validation"></a>
## Post-conversion validation

Board validation is not-run. Verify actual metadata, 224×224 packed NV12 and one F32 output squeezing to 1000 scores. A rebuilt model can be selected with the exact contract reference and an external path; keep its hash/provenance separate from the publisher artifact.

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant b0
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts). No ONNX/weights/compiled files are added by this migration.

<a id="known-gaps"></a>
## Known gaps

No conversion/accuracy/board evidence has been produced. Export and calibration availability are stated above per actual source, not inferred from the model name. Pin versions, weights and dataset inputs before rebuilding.
