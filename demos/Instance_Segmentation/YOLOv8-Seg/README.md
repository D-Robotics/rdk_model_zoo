English | [简体中文](./README_cn.md)

# YOLOv8 Instance Segmentation
- [YOLOv8 Instance Segmentation](#yolov8-instance-segmentation)
  - [Introduction to YOLO](#introduction-to-yolo)
  - [Model download](#model-download)
  - [Input / Output Data](#input--output-data)
  - [Original Processing Flow](#original-processing-flow)
  - [Optimized Processing Flow](#optimized-processing-flow)
  - [Steps Reference](#steps-reference)
    - [Environment and Project Preparation](#environment-and-project-preparation)
    - [Export to ONNX](#export-to-onnx)
    - [PTQ Quantization Transformation](#ptq-quantization-transformation)
    - [Remove Dequantize Nodes for Bounding Box and Mask Coefficients Output Heads](#remove-dequantize-nodes-for-bounding-box-and-mask-coefficients-output-heads)
    - [Partial Compilation Log Reference](#partial-compilation-log-reference)
  - [Model Training](#model-training)
  - [Performance Data](#performance-data)
  - [FAQ](#faq)
  - [Reference](#reference)


## Introduction to YOLO
YOLO (You Only Look Once), a popular object detection and image segmentation model, was developed by Joseph Redmon and Ali Farhadi at the University of Washington. Launched in 2015, YOLO quickly gained popularity for its high speed and accuracy.

 - YOLOv2, released in 2016, improved the original model by incorporating batch normalization, anchor boxes, and dimension clusters.
 - YOLOv3, launched in 2018, further enhanced the model's performance using a more efficient backbone network, multiple anchors and spatial pyramid pooling.
 - YOLOv4 was released in 2020, introducing innovations like Mosaic data augmentation, a new anchor-free detection head, and a new loss function.
 - YOLOv5 further improved the model's performance and added new features such as hyperparameter optimization, integrated experiment tracking and automatic export to popular export formats.
 - YOLOv6 was open-sourced by Meituan in 2022 and is in use in many of the company's autonomous delivery robots.
 - YOLOv7 added additional tasks such as pose estimation on the COCO keypoints dataset.
 - YOLOv8 is the latest version of YOLO by Ultralytics. As a cutting-edge, state-of-the-art (SOTA) model, YOLOv8 builds on the success of previous versions, introducing new features and improvements for enhanced performance, flexibility, and efficiency. YOLOv8 supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification. This versatility allows users to leverage YOLOv8's capabilities across diverse applications and domains.
 - YOLOv9 introduces innovative methods like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).
 - YOLOv10 is created by researchers from Tsinghua University using the Ultralytics Python package. This version provides real-time object detection advancements by introducing an End-to-End head that eliminates Non-Maximum Suppression (NMS) requirements.

## Model download
Reference to `./model/download.md`

## Input / Output Data
- Input: 1x3x640x640, dtype=UINT8
- Output 0: [1, 80, 80, 64], dtype=INT32
- Output 1: [1, 40, 40, 64], dtype=INT32
- Output 2: [1, 20, 20, 64], dtype=INT32
- Output 3: [1, 80, 80, 80], dtype=FLOAT32
- Output 4: [1, 40, 40, 80], dtype=FLOAT32
- Output 5: [1, 20, 20, 80], dtype=FLOAT32
- Output 6: [1, 80, 80, 32], dtype=INT32
- Output 7: [1, 40, 40, 32], dtype=INT32
- Output 8: [1, 20, 20, 32], dtype=INT32
- Output 9: [1, 160, 160, 32], dtype=FLOAT32

## Original Processing Flow
![](imgs/YOLOv8_Instance_Segmentation_Origin.png)

## Optimized Processing Flow
![](imgs/YOLOv8_Instance_Segmentation_Quantize.png)

- In the **Mask Coefficients** section, two **GatherElements** operations are used to obtain the final valid Grid Cell's Mask Coefficients information, which amounts to 32 coefficients. These 32 coefficients are combined linearly with the Mask Protos part, or considered as a weighted sum, to derive the Mask information corresponding to the object in the Grid Cell.

Refer to the YOLOv8 Detect section documentation for the following:
- **Classify** part, **Dequantize** operation.
- **Classify** part, **ReduceMax** operation.
- **Classify** part, **Threshold (TopK)** operation.
- **Classify** part, **GatherElements** operation and **ArgMax** operation.
- **Bounding Box** part, **GatherElements** operation and **Dequantize** operation.
- **Bounding Box** part, **DFL**: **SoftMax + Conv** operation.
- **Bounding Box** part, **Decode**: **dist2bbox(ltrb2xyxy)** operation.
- **nms** operation.


## Steps Reference

Note: For any errors such as **No such file or directory**, **No module named "xxx"**, **command not found**, please carefully check and do not blindly copy and run the commands. If you do not understand the modification process, visit the developer community starting from YOLOv5 for more information.
### Environment and Project Preparation
- Download the ultralytics/ultralytics repository and follow the YOLOv8 official documentation to set up the environment.
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
- Navigate into the local repository and download the official pre-trained weights. Here, we use the 3.4 million parameter YOLOv8n-Seg model as an example.
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
```

### Export to ONNX
- Uninstall the YOLO related command-line tool so that modifying the `./ultralytics/ultralytics` directory will take effect directly.
```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If it exists, uninstall it
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```
- Modify the output head of Detect to separately output the Bounding Box information and Classify information for each feature layer, resulting in six output heads.

File path: `./ultralytics/ultralytics/nn/modules/head.py`, around line 51, replace the `Detect` class's `forward` method with the following content.
Note: It is recommended to keep a backup of the original `forward` method, for example, rename it to `forward_`, to easily revert for training purposes.
```python
def forward(self, x):
    bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    clses = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    return (bboxes, clses)
```

File path: `./ultralytics/ultralytics/nn/modules/head.py`, around line 180, replace the `Segment` class's `forward` function with the following content. In addition to the detection part's six heads, there are three `32*(80*80+40*40+20*20)` mask coefficient tensor output heads, and one `32*160*160` base tensor used to synthesize the result.
```python
def forward(self, x):
    p = self.proto(x[0]).permute(0, 2, 3, 1).contiguous()
    mc = [self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    bboxes, clses = Detect.forward(self, x)
    return (mc, bboxes, clses, p) 
```

- Run the following Python script. If there is a **No module named onnxsim** error, install it.
```python
from ultralytics import YOLO
YOLO('yolov8n-seg.pt').export(imgsz=640, format='onnx', simplify=True, opset=11)
```

### PTQ Quantization Transformation
- Refer to the Tian Gong Kai Wu toolchain manual and OE package to check the model. All operators are on the BPU, so it can be compiled directly. The corresponding YAML file is located in the `./ptq_yamls` directory.
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolov8n-seg.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov8_instance_seg_bayese_640x640_nchw.yaml
```

### Remove Dequantize Nodes for Bounding Box and Mask Coefficients Output Heads
- Identify the names of the dequantize nodes for the three Bounding Box output heads
Through the logs generated when running `hb_mapper makerbin`, find that the three outputs with sizes `[1, 80, 80, 64]`, `[1, 40, 40, 64]`, and `[1, 20, 20, 64]` have the names 379, 387, and 395 respectively.
- Identify the names of the dequantize nodes for the three Mask Coefficients output heads
Through the logs generated when running `hb_mapper makerbin`, find that the three outputs with sizes `[1, 80, 80, 32]`, `[1, 40, 40, 32]`, and `[1, 20, 20, 32]` have the names `output0`, `output1`, and 371 respectively.
```bash
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 32], dtype=FLOAT32
    output1:              shape=[1, 40, 40, 32], dtype=FLOAT32
    371:                  shape=[1, 20, 20, 32], dtype=FLOAT32
    379:                  shape=[1, 80, 80, 64], dtype=FLOAT32
    387:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    395:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    403:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    411:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    419:                  shape=[1, 20, 20, 80], dtype=FLOAT32
    347:                  shape=[1, 160, 160, 32], dtype=FLOAT32
```

- Enter the compiled product directory.
```bash
$ cd yolov8n_instance_seg_bayese_640x640_nchw
```
- Identify the dequantize nodes that can be removed.
```bash
$ hb_model_modifier yolov8n_instance_seg_bayese_640x640_nchw.bin
```
- In the generated `hb_model_modifier.log` file, find the following information. Primarily locate the names of the three output heads with sizes `[1, 64, 80, 80]`, `[1, 64, 40, 40]`, and `[1, 64, 20, 20]` and the three output heads with sizes `[1, 80, 80, 32]`, `[1, 40, 40, 32]`, and `[1, 20, 20, 32]`. You can also use tools like Netron to inspect the ONNX model and obtain the output head names.
The names are:
> "/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize"
> "/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize"
> "/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize"
> "/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize"
> "/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize"
> "/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"

```bash
2024-08-16 14:32:17,593 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv4.0/cv4.0.2/Conv_output_0_quantized"
input: "/model.22/cv4.0/cv4.0.2/Conv_x_scale"
output: "output0"
name: "/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 14:32:17,594 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv4.1/cv4.1.2/Conv_output_0_quantized"
input: "/model.22/cv4.1/cv4.1.2/Conv_x_scale"
output: "output1"
name: "/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 14:32:17,594 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv4.2/cv4.2.2/Conv_output_0_quantized"
input: "/model.22/cv4.2/cv4.2.2/Conv_x_scale"
output: "371"
name: "/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 14:32:17,594 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv2.0/cv2.0.2/Conv_output_0_quantized"
input: "/model.22/cv2.0/cv2.0.2/Conv_x_scale"
output: "379"
name: "/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 14:32:17,594 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv2.1/cv2.1.2/Conv_output_0_quantized"
input: "/model.22/cv2.1/cv2.1.2/Conv_x_scale"
output: "387"
name: "/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 14:32:17,594 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv2.2/cv2.2.2/Conv_output_0_quantized"
input: "/model.22/cv2.2/cv2.2.2/Conv_x_scale"
output: "395"
name: "/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"
```
- Use the following command to remove the aforementioned six dequantize nodes. Note that the names may differ during export, so please confirm them carefully.
```bash
$ hb_model_modifier yolov8n_instance_seg_bayese_640x640_nchw.bin \
-r "/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize" \
-r "/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize" \
-r "/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize" \
-r "/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize" \
-r "/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize" \
-r "/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"
```
- Successful removal will display the following log:
```bash
2024-08-16 14:45:18,923 INFO log will be stored in /open_explorer/yolov8n_instance_seg_bayese_640x640_nchw/hb_model_modifier.log
2024-08-16 14:45:18,929 INFO Nodes that will be removed from this model: ['/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize', '/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize', '/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize', '/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize', '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize', '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize']
2024-08-16 14:45:18,929 INFO Node '/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,929 INFO scale: /model.22/cv4.0/cv4.0.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,930 INFO Node '/model.22/cv4.0/cv4.0.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,930 INFO Node '/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,930 INFO scale: /model.22/cv4.1/cv4.1.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,930 INFO Node '/model.22/cv4.1/cv4.1.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,930 INFO Node '/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,931 INFO scale: /model.22/cv4.2/cv4.2.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,931 INFO Node '/model.22/cv4.2/cv4.2.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,931 INFO Node '/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,931 INFO scale: /model.22/cv2.0/cv2.0.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,931 INFO Node '/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,932 INFO Node '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,932 INFO scale: /model.22/cv2.1/cv2.1.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,932 INFO Node '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,932 INFO Node '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 14:45:18,932 INFO scale: /model.22/cv2.2/cv2.2.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 14:45:18,933 INFO Node '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' is removed
2024-08-16 14:45:18,936 INFO modified model saved as yolov8n_instance_seg_bayese_640x640_nchw_modified.bin
```

- The resulting bin model name is `yolov8n_instance_seg_bayese_640x640_nchw_modified.bin`, which is the final model.
- An NCHW input model can prepare input data using OpenCV and numpy.
- An NV12 input model can prepare input data using hardware devices such as codec, JPU, VPU, GPU, or directly use the corresponding functionality provided by TROS.


### Partial Compilation Log Reference
```bash
2024-08-16 14:14:00,022 file: build.py func: build line No: 36 Start to Horizon NN Model Convert.
2024-08-16 14:14:00,023 file: model_debug.py func: model_debug line No: 61 Loading horizon_nn debug methods:[]
2024-08-16 14:14:00,023 file: cali_dict_parser.py func: cali_dict_parser line No: 40 Parsing the calibration parameter
2024-08-16 14:14:00,023 file: build.py func: build line No: 146 The specified model compilation architecture: bayes-e.
2024-08-16 14:14:00,023 file: build.py func: build line No: 148 The specified model compilation optimization parameters: [].
2024-08-16 14:14:00,046 file: build.py func: build line No: 36 Start to prepare the onnx model.
2024-08-16 14:14:00,047 file: utils.py func: utils line No: 53 Input ONNX Model Information:
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 32], dtype=FLOAT32
    output1:              shape=[1, 40, 40, 32], dtype=FLOAT32
    371:                  shape=[1, 20, 20, 32], dtype=FLOAT32
    379:                  shape=[1, 80, 80, 64], dtype=FLOAT32
    387:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    395:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    403:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    411:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    419:                  shape=[1, 20, 20, 80], dtype=FLOAT32
    347:                  shape=[1, 160, 160, 32], dtype=FLOAT32
2024-08-16 14:14:00,230 file: build.py func: build line No: 39 End to prepare the onnx model.
2024-08-16 14:14:00,471 file: build.py func: build line No: 186 Saving model: yolov8n_instance_seg_bayese_640x640_nchw_original_float_model.onnx.
2024-08-16 14:14:00,472 file: build.py func: build line No: 36 Start to optimize the model.
2024-08-16 14:14:00,742 file: build.py func: build line No: 39 End to optimize the model.
2024-08-16 14:14:00,755 file: build.py func: build line No: 186 Saving model: yolov8n_instance_seg_bayese_640x640_nchw_optimized_float_model.onnx.
2024-08-16 14:14:00,755 file: build.py func: build line No: 36 Start to calibrate the model.
2024-08-16 14:14:01,026 file: calibration_data_set.py func: calibration_data_set line No: 82 input name: images,  number_of_samples: 50
2024-08-16 14:14:01,026 file: calibration_data_set.py func: calibration_data_set line No: 93 There are 50 samples in the calibration data set.
2024-08-16 14:14:01,034 file: default_calibrater.py func: default_calibrater line No: 122 Run calibration model with default calibration method.
2024-08-16 14:14:02,347 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 14:15:05,577 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 14:15:18,218 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 14:15:54,739 file: default_calibrater.py func: default_calibrater line No: 211 Select kl:num_bins=1024 method.
2024-08-16 14:15:58,749 file: build.py func: build line No: 39 End to calibrate the model.
2024-08-16 14:15:58,775 file: build.py func: build line No: 186 Saving model: yolov8n_instance_seg_bayese_640x640_nchw_calibrated_model.onnx.
2024-08-16 14:15:58,775 file: build.py func: build line No: 36 Start to quantize the model.
2024-08-16 14:15:59,721 file: build.py func: build line No: 39 End to quantize the model.
2024-08-16 14:15:59,825 file: build.py func: build line No: 186 Saving model: yolov8n_instance_seg_bayese_640x640_nchw_quantized_model.onnx.
2024-08-16 14:16:00,120 file: build.py func: build line No: 36 Start to compile the model with march bayes-e.
2024-08-16 14:16:00,278 file: hybrid_build.py func: hybrid_build line No: 133 Compile submodel: main_graph_subgraph_0
2024-08-16 14:16:00,477 file: hbdk_cc.py func: hbdk_cc line No: 115 hbdk-cc parameters:['--O3', '--core-num', '1', '--fast', '--input-layout', 'NHWC', '--output-layout', 'NHWC', '--input-source', 'ddr']
2024-08-16 14:16:00,478 file: hbdk_cc.py func: hbdk_cc line No: 116 hbdk-cc command used:hbdk-cc -f hbir -m /tmp/tmpcxr_kxb1/main_graph_subgraph_0.hbir -o /tmp/tmpcxr_kxb1/main_graph_subgraph_0.hbm --march bayes-e --progressbar --O3 --core-num 1 --fast --input-layout NHWC --output-layout NHWC --input-source ddr
2024-08-16 14:16:00,478 file: tool_utils.py func: tool_utils line No: 317 Can not find the scale for node HZ_PREPROCESS_FOR_images_NCHW2NHWC_LayoutConvert_Input0
2024-08-16 14:18:48,605 file: tool_utils.py func: tool_utils line No: 322 consumed time 168.103
2024-08-16 14:18:48,702 file: tool_utils.py func: tool_utils line No: 322 FPS=188.82, latency = 5296.0 us, DDR = 25174368 bytes   (see main_graph_subgraph_0.html)
2024-08-16 14:18:48,782 file: build.py func: build line No: 39 End to compile the model with march bayes-e.
2024-08-16 14:18:48,807 file: print_node_info.py func: print_node_info line No: 57 The converted model node information:
==================================================================================================================================
Node                                                ON   Subgraph  Type           Cosine Similarity  Threshold   In/Out DataType  
----------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess   0.999779           127.000000  int8/int8        
/model.0/conv/Conv                                  BPU  id(0)     Conv           0.999641           0.995605    int8/int8        
/model.0/act/Mul                                    BPU  id(0)     HzSwish        0.989701           22.810610   int8/int8        
/model.1/conv/Conv                                  BPU  id(0)     Conv           0.972192           9.151398    int8/int8        
/model.1/act/Mul                                    BPU  id(0)     HzSwish        0.972105           26.311205   int8/int8        
/model.2/cv1/conv/Conv                              BPU  id(0)     Conv           0.976181           25.371887   int8/int8        
/model.2/cv1/act/Mul                                BPU  id(0)     HzSwish        0.985506           19.969627   int8/int8        
/model.2/Split                                      BPU  id(0)     Split          0.995095           7.959165    int8/int8        
/model.2/m.0/cv1/conv/Conv                          BPU  id(0)     Conv           0.982515           7.959165    int8/int8        
/model.2/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish        0.986101           8.674012    int8/int8        
/model.2/m.0/cv2/conv/Conv                          BPU  id(0)     Conv           0.973077           8.419121    int8/int8        
/model.2/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish        0.975514           12.190964   int8/int8        
UNIT_CONV_FOR_/model.2/m.0/Add                      BPU  id(0)     Conv           0.995095           7.959165    int8/int8        
/model.2/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.2/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.2/Concat                                     BPU  id(0)     Concat         0.987132           7.959165    int8/int8        
/model.2/cv2/conv/Conv                              BPU  id(0)     Conv           0.983192           14.770185   int8/int8        
/model.2/cv2/act/Mul                                BPU  id(0)     HzSwish        0.993170           9.126329    int8/int8        
/model.3/conv/Conv                                  BPU  id(0)     Conv           0.990073           5.923653    int8/int8        
/model.3/act/Mul                                    BPU  id(0)     HzSwish        0.992970           5.024549    int8/int8        
/model.4/cv1/conv/Conv                              BPU  id(0)     Conv           0.991381           3.821646    int8/int8        
/model.4/cv1/act/Mul                                BPU  id(0)     HzSwish        0.993407           5.574259    int8/int8        
/model.4/Split                                      BPU  id(0)     Split          0.991595           4.008720    int8/int8        
/model.4/m.0/cv1/conv/Conv                          BPU  id(0)     Conv           0.986378           4.008720    int8/int8        
/model.4/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish        0.981661           5.402080    int8/int8        
/model.4/m.0/cv2/conv/Conv                          BPU  id(0)     Conv           0.987953           2.452948    int8/int8        
/model.4/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish        0.990003           4.680353    int8/int8        
UNIT_CONV_FOR_/model.4/m.0/Add                      BPU  id(0)     Conv           0.995746           4.008720    int8/int8        
/model.4/m.1/cv1/conv/Conv                          BPU  id(0)     Conv           0.992773           4.663066    int8/int8        
/model.4/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish        0.992980           4.205463    int8/int8        
/model.4/m.1/cv2/conv/Conv                          BPU  id(0)     Conv           0.988589           2.620327    int8/int8        
/model.4/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish        0.993064           6.268336    int8/int8        
UNIT_CONV_FOR_/model.4/m.1/Add                      BPU  id(0)     Conv           0.995406           4.663066    int8/int8        
/model.4/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.4/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.4/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                  int8/int8        
/model.4/Concat                                     BPU  id(0)     Concat         0.995022           4.008720    int8/int8        
/model.4/cv2/conv/Conv                              BPU  id(0)     Conv           0.986058           4.964625    int8/int8        
/model.4/cv2/act/Mul                                BPU  id(0)     HzSwish        0.980343           5.177021    int8/int8        
/model.5/conv/Conv                                  BPU  id(0)     Conv           0.989115           1.648952    int8/int8        
/model.5/act/Mul                                    BPU  id(0)     HzSwish        0.986823           7.388950    int8/int8        
/model.6/cv1/conv/Conv                              BPU  id(0)     Conv           0.984163           3.007172    int8/int8        
/model.6/cv1/act/Mul                                BPU  id(0)     HzSwish        0.969440           6.992102    int8/int8        
/model.6/Split                                      BPU  id(0)     Split          0.983327           1.739103    int8/int8        
/model.6/m.0/cv1/conv/Conv                          BPU  id(0)     Conv           0.988370           1.739103    int8/int8        
/model.6/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish        0.964151           6.619705    int8/int8        
/model.6/m.0/cv2/conv/Conv                          BPU  id(0)     Conv           0.967682           1.325212    int8/int8        
/model.6/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish        0.965522           6.451187    int8/int8        
UNIT_CONV_FOR_/model.6/m.0/Add                      BPU  id(0)     Conv           0.983327           1.739103    int8/int8        
/model.6/m.1/cv1/conv/Conv                          BPU  id(0)     Conv           0.985118           4.035169    int8/int8        
/model.6/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish        0.972546           6.828810    int8/int8        
/model.6/m.1/cv2/conv/Conv                          BPU  id(0)     Conv           0.978115           1.925712    int8/int8        
/model.6/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish        0.979919           10.571894   int8/int8        
UNIT_CONV_FOR_/model.6/m.1/Add                      BPU  id(0)     Conv           0.976397           4.035169    int8/int8        
/model.6/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.6/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                  int8/int8        
/model.6/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                  int8/int8        
/model.6/Concat                                     BPU  id(0)     Concat         0.979384           1.739103    int8/int8        
/model.6/cv2/conv/Conv                              BPU  id(0)     Conv           0.985567           3.777670    int8/int8        
/model.6/cv2/act/Mul                                BPU  id(0)     HzSwish        0.968548           7.449065    int8/int8        
/model.7/conv/Conv                                  BPU  id(0)     Conv           0.981500           1.436547    int8/int8        
/model.7/act/Mul                                    BPU  id(0)     HzSwish        0.954764           7.771261    int8/int8        
/model.8/cv1/conv/Conv                              BPU  id(0)     Conv           0.972024           2.540126    int8/int8        
/model.8/cv1/act/Mul                                BPU  id(0)     HzSwish        0.946683           9.435396    int8/int8        
/model.8/Split                                      BPU  id(0)     Split          0.944037           3.152479    int8/int8        
/model.8/m.0/cv1/conv/Conv                          BPU  id(0)     Conv           0.964745           3.152479    int8/int8        
/model.8/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish        0.935775           8.760098    int8/int8        
/model.8/m.0/cv2/conv/Conv                          BPU  id(0)     Conv           0.945278           3.625072    int8/int8        
/model.8/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish        0.936085           13.827316   int8/int8        
UNIT_CONV_FOR_/model.8/m.0/Add                      BPU  id(0)     Conv           0.953098           3.152479    int8/int8        
/model.8/Concat                                     BPU  id(0)     Concat         0.934154           3.152479    int8/int8        
/model.8/cv2/conv/Conv                              BPU  id(0)     Conv           0.948634           3.152479    int8/int8        
/model.8/cv2/act/Mul                                BPU  id(0)     HzSwish        0.921557           11.313808   int8/int8        
/model.9/cv1/conv/Conv                              BPU  id(0)     Conv           0.973669           3.151532    int8/int8        
/model.9/cv1/act/Mul                                BPU  id(0)     HzSwish        0.976808           6.803682    int8/int8        
/model.9/m/MaxPool                                  BPU  id(0)     MaxPool        0.991139           7.846062    int8/int8        
/model.9/m_1/MaxPool                                BPU  id(0)     MaxPool        0.993349           7.846062    int8/int8        
/model.9/m_2/MaxPool                                BPU  id(0)     MaxPool        0.993786           7.846062    int8/int8        
/model.9/Concat                                     BPU  id(0)     Concat         0.991218           7.846062    int8/int8        
/model.9/cv2/conv/Conv                              BPU  id(0)     Conv           0.979349           7.846062    int8/int8        
/model.9/cv2/act/Mul                                BPU  id(0)     HzSwish        0.897185           9.506282    int8/int8        
/model.10/Resize                                    BPU  id(0)     Resize         0.897175           2.125477    int8/int8        
/model.10/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                  int8/int8        
...el.6/cv2/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                  int8/int8        
/model.11/Concat                                    BPU  id(0)     Concat         0.918529           2.125477    int8/int8        
/model.12/cv1/conv/Conv                             BPU  id(0)     Conv           0.949666           2.015612    int8/int8        
/model.12/cv1/act/Mul                               BPU  id(0)     HzSwish        0.946858           8.101270    int8/int8        
/model.12/Split                                     BPU  id(0)     Split          0.970928           2.633660    int8/int8        
/model.12/m.0/cv1/conv/Conv                         BPU  id(0)     Conv           0.971128           2.633660    int8/int8        
/model.12/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish        0.957243           7.655683    int8/int8        
/model.12/m.0/cv2/conv/Conv                         BPU  id(0)     Conv           0.957100           2.152981    int8/int8        
/model.12/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish        0.969479           7.765214    int8/int8        
/model.12/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.12/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.12/Concat                                    BPU  id(0)     Concat         0.955293           2.633660    int8/int8        
/model.12/cv2/conv/Conv                             BPU  id(0)     Conv           0.945614           2.655962    int8/int8        
/model.12/cv2/act/Mul                               BPU  id(0)     HzSwish        0.953800           8.277517    int8/int8        
/model.13/Resize                                    BPU  id(0)     Resize         0.953802           3.692319    int8/int8        
/model.13/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                  int8/int8        
...el.4/cv2/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                  int8/int8        
/model.14/Concat                                    BPU  id(0)     Concat         0.961886           3.692319    int8/int8        
/model.15/cv1/conv/Conv                             BPU  id(0)     Conv           0.987480           2.488732    int8/int8        
/model.15/cv1/act/Mul                               BPU  id(0)     HzSwish        0.991672           5.461772    int8/int8        
/model.15/Split                                     BPU  id(0)     Split          0.992359           2.657087    int8/int8        
/model.15/m.0/cv1/conv/Conv                         BPU  id(0)     Conv           0.984787           2.657087    int8/int8        
/model.15/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish        0.980956           7.162855    int8/int8        
/model.15/m.0/cv2/conv/Conv                         BPU  id(0)     Conv           0.980173           3.073536    int8/int8        
/model.15/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish        0.986537           6.183841    int8/int8        
/model.15/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.15/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.15/Concat                                    BPU  id(0)     Concat         0.990079           2.657087    int8/int8        
/model.15/cv2/conv/Conv                             BPU  id(0)     Conv           0.984204           3.450227    int8/int8        
/model.15/cv2/act/Mul                               BPU  id(0)     HzSwish        0.987080           6.789716    int8/int8        
/model.16/conv/Conv                                 BPU  id(0)     Conv           0.962443           3.019450    int8/int8        
/model.22/proto/cv1/conv/Conv                       BPU  id(0)     Conv           0.974608           3.019450    int8/int8        
/model.22/cv4.0/cv4.0.0/conv/Conv                   BPU  id(0)     Conv           0.954863           3.019450    int8/int8        
/model.22/cv2.0/cv2.0.0/conv/Conv                   BPU  id(0)     Conv           0.978980           3.019450    int8/int8        
/model.22/cv3.0/cv3.0.0/conv/Conv                   BPU  id(0)     Conv           0.981073           3.019450    int8/int8        
/model.16/act/Mul                                   BPU  id(0)     HzSwish        0.942932           6.815027    int8/int8        
/model.22/proto/cv1/act/Mul                         BPU  id(0)     HzSwish        0.972202           5.596385    int8/int8        
/model.22/cv4.0/cv4.0.0/act/Mul                     BPU  id(0)     HzSwish        0.949757           6.009086    int8/int8        
/model.22/cv2.0/cv2.0.0/act/Mul                     BPU  id(0)     HzSwish        0.963524           7.337032    int8/int8        
/model.22/cv3.0/cv3.0.0/act/Mul                     BPU  id(0)     HzSwish        0.964152           8.370398    int8/int8        
/model.17/Concat                                    BPU  id(0)     Concat         0.951013           3.692319    int8/int8        
/model.22/proto/upsample/ConvTranspose              BPU  id(0)     ConvTranspose  0.986767           3.132451    int8/int8        
/model.22/cv4.0/cv4.0.1/conv/Conv                   BPU  id(0)     Conv           0.942498           1.826732    int8/int8        
/model.22/cv2.0/cv2.0.1/conv/Conv                   BPU  id(0)     Conv           0.937604           1.274294    int8/int8        
/model.22/cv3.0/cv3.0.1/conv/Conv                   BPU  id(0)     Conv           0.959002           1.971141    int8/int8        
/model.18/cv1/conv/Conv                             BPU  id(0)     Conv           0.954544           3.692319    int8/int8        
/model.22/proto/cv2/conv/Conv                       BPU  id(0)     Conv           0.985993           0.401985    int8/int8        
/model.22/cv4.0/cv4.0.1/act/Mul                     BPU  id(0)     HzSwish        0.953290           7.191059    int8/int8        
/model.22/cv2.0/cv2.0.1/act/Mul                     BPU  id(0)     HzSwish        0.942362           28.006041   int8/int8        
/model.22/cv3.0/cv3.0.1/act/Mul                     BPU  id(0)     HzSwish        0.970100           35.614365   int8/int8        
/model.18/cv1/act/Mul                               BPU  id(0)     HzSwish        0.949186           6.821432    int8/int8        
/model.22/proto/cv2/act/Mul                         BPU  id(0)     HzSwish        0.982941           5.515483    int8/int8        
/model.22/cv4.0/cv4.0.2/Conv                        BPU  id(0)     Conv           0.970855           5.927373    int8/int32       
/model.22/cv2.0/cv2.0.2/Conv                        BPU  id(0)     Conv           0.984537           44.928654   int8/int32       
/model.22/cv3.0/cv3.0.2/Conv                        BPU  id(0)     Conv           0.999475           11.418165   int8/int32       
/model.18/Split                                     BPU  id(0)     Split          0.957854           1.823525    int8/int8        
/model.22/proto/cv3/conv/Conv                       BPU  id(0)     Conv           0.955678           0.552066    int8/int16       
/model.18/m.0/cv1/conv/Conv                         BPU  id(0)     Conv           0.968385           1.823525    int8/int8        
/model.22/proto/cv3/act/Mul                         BPU  id(0)     HzSwish        0.961348           7.294638    int16/int16      
/model.18/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish        0.961402           6.771519    int8/int8        
/model.18/m.0/cv2/conv/Conv                         BPU  id(0)     Conv           0.936460           2.046028    int8/int8        
/model.18/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish        0.933734           10.005683   int8/int8        
/model.18/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.18/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.18/Concat                                    BPU  id(0)     Concat         0.941911           1.823525    int8/int8        
/model.18/cv2/conv/Conv                             BPU  id(0)     Conv           0.964758           3.209702    int8/int8        
/model.18/cv2/act/Mul                               BPU  id(0)     HzSwish        0.956053           8.534504    int8/int8        
/model.19/conv/Conv                                 BPU  id(0)     Conv           0.937904           1.909663    int8/int8        
/model.22/cv4.1/cv4.1.0/conv/Conv                   BPU  id(0)     Conv           0.950913           1.909663    int8/int8        
/model.22/cv2.1/cv2.1.0/conv/Conv                   BPU  id(0)     Conv           0.949635           1.909663    int8/int8        
/model.22/cv3.1/cv3.1.0/conv/Conv                   BPU  id(0)     Conv           0.953869           1.909663    int8/int8        
/model.19/act/Mul                                   BPU  id(0)     HzSwish        0.899583           8.396005    int8/int8        
/model.22/cv4.1/cv4.1.0/act/Mul                     BPU  id(0)     HzSwish        0.939736           6.342734    int8/int8        
/model.22/cv2.1/cv2.1.0/act/Mul                     BPU  id(0)     HzSwish        0.909527           11.879086   int8/int8        
/model.22/cv3.1/cv3.1.0/act/Mul                     BPU  id(0)     HzSwish        0.935315           9.987036    int8/int8        
/model.20/Concat                                    BPU  id(0)     Concat         0.897680           2.125477    int8/int8        
/model.22/cv4.1/cv4.1.1/conv/Conv                   BPU  id(0)     Conv           0.933450           2.494075    int8/int8        
/model.22/cv2.1/cv2.1.1/conv/Conv                   BPU  id(0)     Conv           0.892864           4.399797    int8/int8        
/model.22/cv3.1/cv3.1.1/conv/Conv                   BPU  id(0)     Conv           0.928013           3.700009    int8/int8        
/model.21/cv1/conv/Conv                             BPU  id(0)     Conv           0.922094           2.125477    int8/int8        
/model.22/cv4.1/cv4.1.1/act/Mul                     BPU  id(0)     HzSwish        0.923892           7.927379    int8/int8        
/model.22/cv2.1/cv2.1.1/act/Mul                     BPU  id(0)     HzSwish        0.890919           35.873985   int8/int8        
/model.22/cv3.1/cv3.1.1/act/Mul                     BPU  id(0)     HzSwish        0.947798           41.267296   int8/int8        
/model.21/cv1/act/Mul                               BPU  id(0)     HzSwish        0.899576           9.383277    int8/int8        
/model.22/cv4.1/cv4.1.2/Conv                        BPU  id(0)     Conv           0.965445           6.183768    int8/int32       
/model.22/cv2.1/cv2.1.2/Conv                        BPU  id(0)     Conv           0.976339           14.197179   int8/int32       
/model.22/cv3.1/cv3.1.2/Conv                        BPU  id(0)     Conv           0.998825           30.421883   int8/int32       
/model.21/Split                                     BPU  id(0)     Split          0.931002           2.960426    int8/int8        
/model.21/m.0/cv1/conv/Conv                         BPU  id(0)     Conv           0.949687           2.960426    int8/int8        
/model.21/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish        0.924768           8.884334    int8/int8        
/model.21/m.0/cv2/conv/Conv                         BPU  id(0)     Conv           0.910429           2.125426    int8/int8        
/model.21/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish        0.873477           10.461245   int8/int8        
/model.21/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.21/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                  int8/int8        
/model.21/Concat                                    BPU  id(0)     Concat         0.888471           2.960426    int8/int8        
/model.21/cv2/conv/Conv                             BPU  id(0)     Conv           0.941732           3.847535    int8/int8        
/model.21/cv2/act/Mul                               BPU  id(0)     HzSwish        0.877449           16.453327   int8/int8        
/model.22/cv4.2/cv4.2.0/conv/Conv                   BPU  id(0)     Conv           0.902531           2.343095    int8/int8        
/model.22/cv2.2/cv2.2.0/conv/Conv                   BPU  id(0)     Conv           0.893761           2.343095    int8/int8        
/model.22/cv3.2/cv3.2.0/conv/Conv                   BPU  id(0)     Conv           0.896570           2.343095    int8/int8        
/model.22/cv4.2/cv4.2.0/act/Mul                     BPU  id(0)     HzSwish        0.918133           7.241290    int8/int8        
/model.22/cv2.2/cv2.2.0/act/Mul                     BPU  id(0)     HzSwish        0.878701           13.317915   int8/int8        
/model.22/cv3.2/cv3.2.0/act/Mul                     BPU  id(0)     HzSwish        0.881848           12.449150   int8/int8        
/model.22/cv4.2/cv4.2.1/conv/Conv                   BPU  id(0)     Conv           0.915845           3.366743    int8/int8        
/model.22/cv2.2/cv2.2.1/conv/Conv                   BPU  id(0)     Conv           0.867985           4.329203    int8/int8        
/model.22/cv3.2/cv3.2.1/conv/Conv                   BPU  id(0)     Conv           0.858073           7.521426    int8/int8        
/model.22/cv4.2/cv4.2.1/act/Mul                     BPU  id(0)     HzSwish        0.890156           9.154864    int8/int8        
/model.22/cv2.2/cv2.2.1/act/Mul                     BPU  id(0)     HzSwish        0.889582           39.768051   int8/int8        
/model.22/cv3.2/cv3.2.1/act/Mul                     BPU  id(0)     HzSwish        0.912551           37.483601   int8/int8        
/model.22/cv4.2/cv4.2.2/Conv                        BPU  id(0)     Conv           0.949460           5.405931    int8/int32       
/model.22/cv2.2/cv2.2.2/Conv                        BPU  id(0)     Conv           0.960124           24.789207   int8/int32       
/model.22/cv3.2/cv3.2.2/Conv                        BPU  id(0)     Conv           0.998110           13.536564   int8/int32
```

## Model Training

- Model training should refer to the ultralytics official documentation, which is maintained by ultralytics and is of very high quality. There are also numerous reference materials available online, and obtaining a model with pre-trained weights similar to the official ones is not difficult.
- Note that no modifications to any program code or the `forward` method are required during training.

## Performance Data

RDK X5 & RDK X5 Module  
Instance Segmentation (COCO)  
| Model | Size (px) | Num Classes | Params (M) | FP Precision (box/mask) | INT8 Precision (box/mask) | Latency/Throughput (Single-threaded) | Latency/Throughput (Multi-threaded) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv8n-seg | 640×640 | 80 | 3.4  | 36.7/30.5 |  |  |  |
| YOLOv8s-seg | 640×640 | 80 | 11.8 | 44.6/36.8 |  |  |  |
| YOLOv8m-seg | 640×640 | 80 | 27.3 | 49.9/40.8 |  |  |  |
| YOLOv8l-seg | 640×640 | 80 | 46.0 | 52.3/42.6 |  |  |  |
| YOLOv8x-seg | 640×640 | 80 | 71.8 | 53.4/43.4 |  |  |

Notes:  
1. The X5 is in its optimal state: CPU is 8 × A55 @ 1.8G with full-core Performance scheduling, BPU is 1 × Bayes-e @ 1G with a total equivalent int8 computing power of 10 TOPS.
2. Single-threaded latency is for a single frame, single thread, and single BPU core, representing the ideal delay for BPU inference of a single task.
3. Four-thread engineering frame rate is when four threads simultaneously feed tasks to the dual-core BPU. In general engineering scenarios, four threads can minimize single-frame latency while fully utilizing all BPU cores at 100%, achieving a good balance between throughput (FPS) and frame latency. X5 BPU overall is more powerful, generally 2 threads can eat BPU full, frame delay and throughput are very good.
4. Eight-thread extreme frame rate is when eight threads simultaneously feed tasks to the dual-core BPU on the X3, aiming to test the BPU’s extreme performance. Typically, four cores are already saturated; if eight threads perform significantly better than four, it suggests that the model structure needs to improve the "compute/memory access" ratio, or that DDR bandwidth optimization should be selected during compilation.
5. FP/Q mAP: 50-95 precision is calculated using pycocotools and comes from the COCO dataset. This can refer to Microsoft’s paper, and is used here to assess the degree of accuracy degradation for deployment on the board.
6. Run the following command to test the bin model throughput on the board
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
7. Regarding post-processing: At present, the post-processing of Python reconstruction on X5 only requires a single-core single-thread serial about 5ms to complete, that is, it only needs to occupy 2 CPU cores (200% CPU usage, maximum 800% CPU usage), and can complete 400 frames of image post-processing per minute, and post-processing will not constitute a bottleneck.

## FAQ

[D-Robotics Developer Community](developer.d-robotics.cc)

## Reference

[ultralytics](https://docs.ultralytics.com/)
