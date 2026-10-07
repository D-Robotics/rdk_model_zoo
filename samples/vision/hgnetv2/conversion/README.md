# HGNetV2 conversion

<a id="source-model"></a>
## Source model

Five export scripts load `timm` pretrained PP-HGNetV2 weights
(`hgnetv2_b0.ssld_stage2_ft_in1k` … `hgnetv2_b4.ssld_stage2_ft_in1k`).
The delivered scripts name torch 1.13 and OE Docker v1.2.8; script comments
name `hb_mapper` 1.24.3 and opset 11. timm and weight revisions are not
pinned. The first run downloads the weights from Hugging Face; inference
itself never runs an export.

| Variant | timm model id | Output `.bin` |
| --- | --- | --- |
| b0 | `hgnetv2_b0.ssld_stage2_ft_in1k` | `hgnetv2_b0_224x224_nv12.bin` |
| b1 | `hgnetv2_b1.ssld_stage2_ft_in1k` | `hgnetv2_b1_224x224_nv12.bin` |
| b2 | `hgnetv2_b2.ssld_stage2_ft_in1k` | `hgnetv2_b2_224x224_nv12.bin` |
| b3 | `hgnetv2_b3.ssld_stage2_ft_in1k` | `hgnetv2_b3_224x224_nv12.bin` |
| b4 | `hgnetv2_b4.ssld_stage2_ft_in1k` | `hgnetv2_b4_224x224_nv12.bin` |

<a id="toolchain-targets"></a>
## Toolchain and targets

Run X5 model conversion on an x86 Linux host (march `bayes-e`). Install the RDK X5 OpenExplorer toolchain **1.2.8**:

```bash
# download and load the offline Docker image
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

Alternatively, obtain the offline Docker image from the D-Robotics
developer forum
([topic 35229](https://forum.d-robotics.cc/t/topic/35229)). Start the
container with the repository mounted so the workspace is shared:

```bash
# replace /path/to/rdk_model_zoo with your checkout path
docker run -it --rm \
  -v /path/to/rdk_model_zoo:/data \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

Install the export dependency inside the container (or any Python 3
environment with PyTorch ≥ 1.13): `pip install timm`. If the environment
has restricted network access, set a Hugging Face mirror before launching
the exporters: `export HF_ENDPOINT=https://hf-mirror.com`.

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `hgnetv2_b0.yaml` | `./onnx_export/hgnetv2_b0.onnx` | `hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin` |
| `hgnetv2_b1.yaml` | `./onnx_export/hgnetv2_b1.onnx` | `hgnetv2_b1_224x224_nv12/hgnetv2_b1_224x224_nv12.bin` |
| `hgnetv2_b2.yaml` | `./onnx_export/hgnetv2_b2.onnx` | `hgnetv2_b2_224x224_nv12/hgnetv2_b2_224x224_nv12.bin` |
| `hgnetv2_b3.yaml` | `./onnx_export/hgnetv2_b3.onnx` | `hgnetv2_b3_224x224_nv12/hgnetv2_b3_224x224_nv12.bin` |
| `hgnetv2_b4.yaml` | `./onnx_export/hgnetv2_b4.onnx` | `hgnetv2_b4_224x224_nv12/hgnetv2_b4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX export

Run the per-variant exporter inside the OE/PyTorch environment. Run from
`onnx_export/` so the outputs match the YAML paths; replace `b0` with
`b1`/`b2`/`b3`/`b4` for the other variants. The scripts use input name
`input`, output name `output`, shape 1×3×224×224 and opset 11.

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion/onnx_export
python3 export_hgnetv2_b0_bpu.py
# output: hgnetv2_b0.onnx in this directory
```

<a id="calibration"></a>
## Calibration

`hb_mapper` needs 20–50 representative ImageNet-style images for INT8
calibration. The YAML files set `cal_data_dir: '../cal_data'`,
`cal_data_type: float32` and `preprocess_on: true` — the loader applies
the YAML normalization (training input RGB/NCHW, mean
`123.675/116.28/103.53`, scale `0.01712475/0.017507/0.01742919`) to the
JPEGs itself. No preparation script is included; create the folder next to
`conversion/` and fill it yourself:

```bash
# cwd: samples/vision/hgnetv2/conversion
mkdir -p ../cal_data
# copy 20-50 JPEG images sampled from the ImageNet validation set
```

<a id="compile"></a>
## Compile

In the OE environment, after the ONNX graph and calibration data are
prepared:

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx_export/hgnetv2_b0.onnx
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```

The YAMLs keep `compile_mode: latency` / `optimize_level: O3`. The
resulting `hgnetv2_b0_224x224_nv12.bin` is written to
`hgnetv2_b0_224x224_nv12/`; move or symlink it into `../model/` so the
runtime sample can find it:

```bash
# cwd: samples/vision/hgnetv2/conversion
cp hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin ../model/
```

Output basenames match the published filenames.

<a id="validation"></a>
## Post-conversion validation

The runtime protocol is 224×224 packed NV12 input and one F32 output
squeezing to 1000 scores. A rebuilt model can be selected with the exact
contract reference and an external path; keep its hash/provenance separate
from the published artifact. The functional check with the published
artifact:

```bash
# cwd: repository root on X5
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant b0
```

<a id="artifacts"></a>
## Artifacts

Published artifacts and landing paths: [model preparation](../model/README.md#artifacts).

<a id="known-gaps"></a>
## Additional preparation

The export scripts fetch `hgnetv2_b*.ssld_stage2_ft_in1k` at run time. Record the timm version and weight hashes used for a build. Prepare the 20–50 ImageNet-style JPEGs for calibration, then run the per-artifact YAML and compile commands above. Keep each published output name and target paired with its configuration.
