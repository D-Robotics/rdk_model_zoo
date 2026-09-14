English | [简体中文](./README_cn.md)

# YOLOv13 iMoonLab Conversion Guide

This directory documents ONNX export, calibration data preparation, HBM compilation, and output inspection for the YOLOv13 Detect model on RDK S100, and it includes the reference YAML and compile logs.

## Directory Structure

```bash
.
├── config_yolov13_detect_nv12.yaml
├── hb_compile_yolov13.txt
├── hb_model_info_yolov13.txt
├── hrt_model_exec_model_info_yolov13.txt
├── README.md
└── README_cn.md
```

## Build Environment

Run model conversion on an x86 Linux host inside the OpenExplore environment instead of installing the toolchain on the board.

- OE resource entry point (Docker + OE dev package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

### 1. Install Docker

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. Load the offline image

Download the CPU Docker image for the RDK S100 series from the OE resource page, then load it:

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
```

### 3. Start the container

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

## Conversion Flow

### 1. Prepare the training environment and weights

YOLOv13 ONNX export must be performed in the iMoonLab/Ultralytics training environment. The source `.pt` weights should come from the official training flow or the official release weights.

```bash
git clone https://github.com/iMoonLab/yolov13.git
cd yolov13
wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
```

Training instructions:

- <https://docs.ultralytics.com/modes/train/>

No code changes and no `forward` changes are required during training.

### 2. Export ONNX

It is recommended to uninstall the `ultralytics` package installed via `pip` or `conda` first, so the source tree you edit is the one that is actually imported.

```bash
conda list | grep ultralytics
pip list | grep ultralytics
conda uninstall ultralytics
pip uninstall ultralytics
```

To verify which `ultralytics` path is imported:

```python
import ultralytics
print(ultralytics.__path__)
```

Then edit `ultralytics/nn/modules/head.py` and replace `Detect.forward` so that each feature level emits separate classification and box tensors, for a total of 6 outputs:

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

If the exported output order is reversed compared with the reference model, swap the append order of `cv2` and `cv3` and export again:

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

Then export ONNX:

```python
from ultralytics import YOLO
YOLO('yolov13n.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

If you hit `No module named onnxsim`, install the dependency. If the exported ONNX IR version is too high, keeping `simplify=False` is acceptable.

### 3. Prepare calibration data

Prepare 20 to 50 images that cover the target scenes for PTQ calibration. The OE development package also provides related examples for calibration data preparation.

## Conversion Reference

ONNX export
PTQ config generation

### 4. Confirm dequant node removal names

Open the exported ONNX in Netron:

- <https://netron.app/>

Locate the three outputs shaped `[1, 80, 80, 64]`, `[1, 40, 40, 64]`, and `[1, 20, 20, 64]`, then fill the corresponding node names into `remove_node_name` in the YAML. A practical rule is to inspect the Dequantize nodes associated with `64 = 4 * REG`, but the exact names depend on the Ultralytics version and must be checked from your export.

![Netron example](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/test_data/netron_conv_example.jpeg)

Reference YAML snippet:

```yaml
model_parameters:
  onnx_model: 'ultralytcs_YOLO.onnx'
  march: nash-e
  layer_out_dump: False
  working_dir: 'ultralytcs_YOLO_output'
  output_model_file_prefix: 'ultralytcs_YOLO'
  remove_node_name: "/model.32/cv2.0/cv2.2.2/Conv;/model.32/cv2.1/cv2.1.2/Conv;/model.32/cv2.2/cv2.2.2/Conv;"
```

### 5. Compile HBM

```bash
hb_compile --config config_yolov13_detect_nv12.yaml
```

The following reference logs are included for comparison:

- [hb_compile_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hb_compile_yolov13.txt)
- [hb_model_info_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hb_model_info_yolov13.txt)
- [hrt_model_exec_model_info_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hrt_model_exec_model_info_yolov13.txt)

## License

This directory follows the repository top-level `LICENSE`.
