English | [简体中文](./README_cn.md)

# YOLOv8 Detect
- [YOLOv8 Detect](#yolov8-detect)
  - [Introduction to YOLO](#introduction-to-yolo)
  - [Performance Data (Summary)](#performance-data-summary)
  - [Model download](#model-download)
  - [Input / Output Data](#input--output-data)
  - [Public Version Processing Flow](#public-version-processing-flow)
  - [Optimized Processing Flow](#optimized-processing-flow)
  - [Step Reference](#step-reference)
    - [Environment and Project Preparation](#environment-and-project-preparation)
    - [Exporting to ONNX](#exporting-to-onnx)
    - [PTQ Quantization Transformation](#ptq-quantization-transformation)
    - [Removing Dequantization Nodes for the Three BBox Output Heads](#removing-dequantization-nodes-for-the-three-bbox-output-heads)
    - [Use the hb\_perf command to visualize the bin model and the hrt\_model\_exec command to check the input/output situation of the bin model](#use-the-hb_perf-command-to-visualize-the-bin-model-and-the-hrt_model_exec-command-to-check-the-inputoutput-situation-of-the-bin-model)
    - [Partial Compilation Log Reference](#partial-compilation-log-reference)
  - [Model Training](#model-training)
  - [Performance Data](#performance-data)
  - [FAQ](#faq)
  - [Reference](#reference)


## Introduction to YOLO

![](imgs/demo_rdkx5_yolov8n_detect.jpg)

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


## Performance Data (Summary)
RDK X5 & RDK X5 Module
Object Detection (COCO)
| Model (Official) | Size (px) | Classes | Params (M) | Throughput (FPS) | Post Process Time (Python) |
|---------|---------|-------|-------------------|--------------------|---|
| YOLOv8n | 640×640 | 80 | 3.2 | 263.6 | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 | 94.9 | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 | 35.7 | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 | 17.9 | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 | 11.2 | 5 ms |

Note: Detailed performance data is at the end of the document.

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

## Public Version Processing Flow
![](imgs/YOLOv8_Detect_Origin.png)

## Optimized Processing Flow
![](imgs/YOLOv8_Detect_Quantize.png)

 - Classification Part, Dequantization Operation
In model compilation, if all dequantization operators were removed, manual dequantization of the three output heads of the classification part is required in post-processing. There are several ways to view the dequantization coefficients, including examining the logs produced during `hb_mapper` execution or obtaining them through the API of the BPU inference interface.
Note that each C dimension has different dequantization coefficients, with 80 dequantization coefficients per head, which can be directly multiplied using numpy broadcasting.
This dequantization is implemented in the bin model, so the obtained outputs are float32.

 - Classification Part, ReduceMax Operation
The ReduceMax operation finds the maximum value along a specific dimension of the Tensor, used here to find the maximum score among the 80 scores for each of the 8400 Grid Cells. The operation targets the values of the 80 categories for each Grid Cell, operating on the C dimension. Note that this operation yields the maximum value but not the index of the maximum value among the 80 values.
The activation function Sigmoid is monotonic, meaning the relative order of the 80 scores before and after the Sigmoid function does not change.
$$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
Therefore, the position of the maximum value (after dequantization) output by the bin model corresponds to the position of the maximum score, and the maximum value output by the bin model, after Sigmoid calculation, is the same as the maximum value from the original onnx model.

 - Classification Part: Threshold (TopK) Operation
This operation identifies the Grid Cells meeting certain criteria among the 8400 Grid Cells. The operation targets the 8400 Grid Cells, operating on the H and W dimensions. If you have read my code, you will notice that I flattened the H and W dimensions for convenience in programming and documentation, but there is no fundamental difference.
Let's assume the score of a category within a Grid Cell is denoted as $x$, and the integer data after activation function application is $y$. A threshold, denoted as $C$, is given, and the condition for a score to be valid is:
$$y = Sigmoid(x) = \frac{1}{1 + e^{-x}} > C$$
Thus, the necessary and sufficient condition for a score to be valid is:
$$x > -\ln\left(\frac{1}{C} - 1\right)$$
This operation returns the indices of the qualifying Grid Cells and the corresponding maximum value, which, after Sigmoid calculation, represents the score for the category of that Grid Cell.

 - Classification Part: GatherElements and ArgMax Operations
Using the indices of the qualifying Grid Cells obtained from the Threshold (TopK) operation, the GatherElements operation retrieves the qualifying Grid Cells, and the ArgMax operation determines which of the 80 categories has the highest value, thus identifying the category of the qualifying Grid Cell.

 - Bounding Box Part: GatherElements and Dequantization Operations
Using the indices of the qualifying Grid Cells obtained from the Threshold (TopK) operation, the GatherElements operation retrieves the qualifying Grid Cells. Here, each C dimension has different dequantization coefficients, with 64 dequantization coefficients per head, which can be directly multiplied using numpy broadcasting, resulting in 1×64×k×1 bounding box information.

 - Bounding Box Part: DFL: Softmax + Conv Operation
Each Grid Cell provides four numbers to determine the location of the bounding box. The DFL structure gives 16 estimates for each side of the bounding box based on the anchor position. The Softmax is applied to these 16 estimates, followed by a convolution operation to compute the expectation. This is a core design of Anchor-Free, where each Grid Cell is responsible for predicting only one bounding box. Assuming the 16 numbers for the prediction of an offset for a particular edge are $l_p$ or $(t_p, t_p, b_p)$, where $p = 0,1,...,15$, the offset formula is:
$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

 - Bounding Box Part: Decode: dist2bbox(ltrb2xyxy) Operation
This operation decodes the ltrb description of each bounding box into xyxy description. ltrb represent the distances from the center of the Grid Cell to the left, top, right, and bottom edges, respectively. After converting the relative positions to absolute positions and multiplying by the sampling factor of the corresponding feature layer, the xyxy coordinates can be restored, where xyxy denotes the predicted coordinates of the top-left and bottom-right corners of the bounding box.
![](imgs/ltrb2xyxy.jpg)

For the input image size of $Size = 640$, for the i-th feature map of the bounding box prediction branch $(i = 1, 2, 3)$, the corresponding downsampling factor is denoted as $Stride(i)$, in YOLOv8 - Detect, $Stride(1) = 8, Stride(2) = 16, Stride(3) = 32$, and the size of the corresponding feature map is $n_i = Size/Stride(i)$, i.e., sizes of $n_1 = 80, n_2 = 40, n_3 = 20$ for three feature maps, resulting in a total of $n_1^2 + n_2^2 + n_3^3 = 8400$ Grid Cells responsible for predicting 8400 bounding boxes.
For feature map i, the detection box at row x and column y predicts the corresponding scale bounding box, where $x, y \in [0, n_i) \cap Z$, $Z$ being the set of integers. The bounding box prediction after the DFL structure is described using ltrb, whereas we need the xyxy description. The transformation is as follows:
$$x_1 = (x + 0.5 - l) \times Stride(i)$$
$$y_1 = (y + 0.5 - t) \times Stride(i)$$
$$x_2 = (x + 0.5 + r) \times Stride(i)$$
$$y_2 = (y + 0.5 + b) \times Stride(i)$$

YOLOv8 and v9 include a non-maximum suppression (NMS) operation to remove duplicate detections, while YOLOv10 does not require it. The final detection results include the class (id), score, and location (xyxy).

## Step Reference

Note: For any errors such as No such file or directory, No module named "xxx", command not found, please carefully check the steps and do not blindly run each command. If you do not understand the modification process, please refer to the developer community starting from YOLOv5.

### Environment and Project Preparation
- Download the ultralytics/ultralytics repository and follow the official YOLOv8 documentation to set up your environment.
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
- Navigate to the local repository and download the official pre-trained weights. Here, we will use the 3.2 million parameter YOLOv8n-Detect model as an example.
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### Exporting to ONNX
- Uninstall the yolo-related command-line commands so that modifications to the `./ultralytics/ultralytics` directory take effect directly.
```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If they exist, uninstall them
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```
- Modify the Detect output head to separate the Bounding Box information and Classify information into six output heads.
File location: `./ultralytics/ultralytics/nn/modules/head.py`, around line 51, replace the `Detect` class's `forward` method with the following content.
Note: It is recommended that you keep the original `forward` method, for example by renaming it to `forward_`, to make it easy to switch back when training.
```python
def forward(self, x):
    bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    clses = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    return (bboxes, clses)
```

- Run the following Python script. If there is a **No module named onnxsim** error, simply install it.
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(imgsz=640, format='onnx', simplify=True, opset=11)
```

### PTQ Quantization Transformation
- Refer to the TianGong KaiWu toolchain manual and OE package to check the model. All operators should be on the BPU for compilation. The corresponding YAML files are located in the `./ptq_yamls` directory.
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolov8n.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov8_detect_nchw.yaml
```

### Removing Dequantization Nodes for the Three BBox Output Heads
- Check the names of the dequantization nodes for the three BBox output heads.
Through the log when running `hb_mapper makerbin`, you can see the sizes `[1, 64, 80, 80]`, `[1, 64, 40, 40]`, and `[1, 64, 20, 20]`. Their names are `output0`, `326`, and `334`.
```bash
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 64], dtype=FLOAT32
    326:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    334:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    342:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    350:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    358:                  shape=[1, 20, 20, 80], dtype=FLOAT32
```

- Navigate to the compiled artifact directory.
```bash
$ cd yolov8n_bayese_640x640_nchw
```
- Check which dequantization nodes can be removed.
```bash
$ hb_model_modifier yolov8n_bayese_640x640_nchw.bin
```
- In the generated `hb_model_modifier.log` file, find the following information. Mainly find the names of the three output heads with sizes `[1, 64, 80, 80]`, `[1, 64, 40, 40]`, and `[1, 64, 20, 20]`. Of course, you can also use tools like Netron to inspect the ONNX model to obtain the names of the output heads.
The names here are:
> "/model.22/cv2.0/cv2.0.2/Conv_output_0_quantized"
> "/model.22/cv2.1/cv2.1.2/Conv_output_0_quantized"
> "/model.22/cv2.2/cv2.2.2/Conv_output_0_quantized"
```bash
2024-08-14 15:50:25,193 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv2.0/cv2.0.2/Conv_output_0_quantized"
input: "/model.22/cv2.0/cv2.0.2/Conv_x_scale"
output: "output0"
name: "/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-14 15:50:25,194 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.22/cv2.1/cv2.1.2/Conv_output_0_quantized"
input: "/model.22/cv2.1/cv2.1.2/Conv_x_scale"
output: "326"
name: "/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

input: "/model.22/cv2.2/cv2.2.2/Conv_x_scale"
output: "334"
name: "/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"
```
- Use the following command to remove the above three dequantization nodes. Note that these names may be different during export, so please carefully confirm.
```bash
$ hb_model_modifier yolov8n_bayese_640x640_nchw.bin \
-r /model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize \
-r /model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize \
-r /model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize
```
- Successful removal will display the following log.

```bash
2024-08-14 15:55:01,233 INFO log will be stored in /open_explorer/yolov8n_bayese_640x640_nchw/hb_model_modifier.log
2024-08-14 15:55:01,238 INFO Nodes that will be removed from this model: ['/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize', '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize', '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize']
2024-08-14 15:55:01,238 INFO Node '/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-14 15:55:01,239 INFO scale: /model.22/cv2.0/cv2.0.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-14 15:55:01,239 INFO Node '/model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' is removed
2024-08-14 15:55:01,239 INFO Node '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-14 15:55:01,239 INFO scale: /model.22/cv2.1/cv2.1.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-14 15:55:01,240 INFO Node '/model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' is removed
2024-08-14 15:55:01,240 INFO Node '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-14 15:55:01,240 INFO scale: /model.22/cv2.2/cv2.2.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-14 15:55:01,240 INFO Node '/model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' is removed
2024-08-14 15:55:01,245 INFO modified model saved as yolov8n_bayese_640x640_nchw_modified.bin
```
- The resulting bin model name will be `yolov8n_bayese_640x640_nchw_modified.bin`, which is the final model.
- Models with NCHW input can prepare input data using OpenCV and numpy.
- Models with NV12 input can prepare input data using hardware devices such as codec, JPU, VPU, GPU, or can be directly used with the corresponding TROS functionality packages.

### Use the hb_perf command to visualize the bin model and the hrt_model_exec command to check the input/output situation of the bin model
 - Bin model before removing the dequantization coefficients
```bash
hb_perf yolov8n_detect_bayese_640x640_nv12.bin
```
The following results can be found in the `hb_perf_result` directory:
![](./imgs/yolov8n_detect_bayese_640x640_nv12.png)


```bash
hrt_model_exec model_info --model_file yolov8n_detect_bayese_640x640_nv12.bin
```
The input/output information of this bin model before removing the dequantization coefficients can be seen
```bash
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-08-27,16:54:41.477.582) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 27.689ms.
This model file has 1 model:
[yolov8n_detect_bayese_640x640_nv12]
---------------------------------------------------------------------
[model name]: yolov8n_detect_bayese_640x640_nv12

input[0]: 
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,640,640,)
aligned shape: (1,3,640,640,)
aligned byte size: 614400
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]: 
name: output0
valid shape: (1,80,80,64,)
aligned shape: (1,80,80,64,)
aligned byte size: 1638400
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1638400,20480,256,4,)

output[1]: 
name: 326
valid shape: (1,40,40,64,)
aligned shape: (1,40,40,64,)
aligned byte size: 409600
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (409600,10240,256,4,)

output[2]: 
name: 334
valid shape: (1,20,20,64,)
aligned shape: (1,20,20,64,)
aligned byte size: 102400
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (102400,5120,256,4,)

output[3]: 
name: 342
valid shape: (1,80,80,80,)
aligned shape: (1,80,80,80,)
aligned byte size: 2048000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (2048000,25600,320,4,)

output[4]: 
name: 350
valid shape: (1,40,40,80,)
aligned shape: (1,40,40,80,)
aligned byte size: 512000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (512000,12800,320,4,)

output[5]: 
name: 358
valid shape: (1,20,20,80,)
aligned shape: (1,20,20,80,)
aligned byte size: 128000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (128000,6400,320,4,)
```


- Bin model after removing the target dequantization coefficients
```bash
hb_perf yolov8n_detect_bayese_640x640_nv12_modified.bin
```
The following results can be found in the `hb_perf_result` directory.
![](./imgs/yolov8n_detect_bayese_640x640_nv12_modified.png)

```bash
hrt_model_exec model_info --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
The input/output information of the bin model before removing the dequantization coefficients, as well as all dequantization coefficients after removing the dequantization nodes, can be seen. This also indicates that these pieces of information are stored within the bin model, which can be obtained using the inference library's API, making it convenient for us to perform corresponding pre-processing and post-processing.
```bash
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-08-27,17:03:38.761.972) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 25.791ms.
This model file has 1 model:
[yolov8n_detect_bayese_640x640_nv12]
---------------------------------------------------------------------
[model name]: yolov8n_detect_bayese_640x640_nv12

input[0]: 
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,640,640,)
aligned shape: (1,3,640,640,)
aligned byte size: 614400
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]: 
name: output0
valid shape: (1,80,80,64,)
aligned shape: (1,80,80,64,)
aligned byte size: 1638400
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (1638400,20480,256,4,)
scale data: 0.000318929,0.00032147,0.000303427,0.000281826,0.000237608,0.000196948,0.000192247,0.000155271,0.000198091,0.000214101,0.000143454,0.000127381,0.000122235,9.83469e-05,8.10663e-05,8.29722e-05,0.000302664,0.000309272,0.000280555,0.000250569,0.000223504,0.000290721,0.000195931,0.000159083,0.000185004,0.00017738,0.000118804,0.00012179,0.000118423,0.000104954,9.37726e-05,0.000122235,0.000314354,0.000319945,0.000282843,0.000243453,0.000244724,0.000179667,0.000201903,0.000133289,0.000208129,0.00018475,0.000123251,0.000120583,0.000112514,9.0469e-05,7.85886e-05,0.0001047,0.000297074,0.000273186,0.000267595,0.000225664,0.000214483,0.000301648,0.000234304,0.000173695,0.000156034,0.000156415,8.93254e-05,0.000109592,0.000106542,8.76736e-05,7.52849e-05,8.96431e-05,
quantizeAxis: 3

output[1]: 
name: 326
valid shape: (1,40,40,64,)
aligned shape: (1,40,40,64,)
aligned byte size: 409600
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (409600,10240,256,4,)
scale data: 0.000231898,0.000221052,0.000178184,0.000175,0.000157353,0.000143408,0.000150209,0.000144097,0.000126451,0.000105533,9.24493e-05,8.53047e-05,9.22772e-05,9.12442e-05,8.67681e-05,9.78723e-05,0.000230349,0.00022329,0.00020332,0.000170609,0.000165359,0.000140051,0.000157095,0.000147196,0.000107427,0.000124815,9.03834e-05,7.26511e-05,8.08717e-05,7.97096e-05,7.42866e-05,9.11582e-05,0.000233964,0.000235341,0.000227078,0.000174655,0.000139363,0.000147971,0.000156148,0.00011965,0.000129033,0.000107169,8.014e-05,8.16894e-05,9.69255e-05,0.000106394,0.000108891,0.000122663,0.00022088,0.000211239,0.000213649,0.000175602,0.000164067,0.000149606,0.000143753,0.000119306,0.00010209,0.000156492,0.000102865,7.32967e-05,8.8834e-05,9.81306e-05,0.000100971,0.000115691,
quantizeAxis: 3

output[2]: 
name: 334
valid shape: (1,20,20,64,)
aligned shape: (1,20,20,64,)
aligned byte size: 102400
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (102400,5120,256,4,)
scale data: 0.00026432,0.000255302,0.000239413,0.000278062,0.000228032,0.000182512,0.000184122,0.000209567,0.00022245,0.000162758,0.000213754,0.000158678,0.000157926,0.000162758,0.000150519,0.000126792,0.000250793,0.000232971,0.000260455,0.00025702,0.000219873,0.000179291,0.000224812,0.000158678,0.000193999,0.000245425,0.000159107,0.00015943,0.000128295,0.000118525,9.5443e-05,6.18393e-05,0.000251008,0.000281068,0.000253155,0.000264535,0.000242204,0.000195825,0.000253799,0.000159644,0.00018176,0.000177573,0.000224597,0.000158249,0.00012486,0.000168985,0.000186914,0.000174889,0.000273983,0.000269044,0.000255731,0.000202803,0.000206668,0.000221591,0.000227388,0.000172742,0.000206131,0.000199045,0.00017242,0.000148694,0.000138172,0.000151378,0.000143326,0.000127436,
quantizeAxis: 3

output[3]: 
name: 342
valid shape: (1,80,80,80,)
aligned shape: (1,80,80,80,)
aligned byte size: 2048000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (2048000,25600,320,4,)

output[4]: 
name: 350
valid shape: (1,40,40,80,)
aligned shape: (1,40,40,80,)
aligned byte size: 512000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (512000,12800,320,4,)

output[5]: 
name: 358
valid shape: (1,20,20,80,)
aligned shape: (1,20,20,80,)
aligned byte size: 128000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (128000,6400,320,4,)
```


### Partial Compilation Log Reference
As can be seen:
- On the X5, YOLOv8n can achieve approximately 260 FPS. However, due to preprocessing, quantization nodes, and some dequantization nodes being executed on the CPU, it runs slightly slower. In practice, three threads can achieve a throughput of 252 FPS.
- The tail transpose node satisfies passive quantization logic and supports acceleration by the BPU, without affecting its parent Convolution operator's ability to output in high int32 precision.
- The cosine similarity of all nodes is greater than 0.9, meeting expectations.
- All operators are on the BPU, and the entire bin model contains only one BPU subgraph.
```bash
2024-08-14 15:08:32,181 file: build.py func: build line No: 36 Start to Horizon NN Model Convert.
2024-08-14 15:08:32,181 file: model_debug.py func: model_debug line No: 61 Loading horizon_nn debug methods:[]
2024-08-14 15:08:32,181 file: cali_dict_parser.py func: cali_dict_parser line No: 40 Parsing the calibration parameter
2024-08-14 15:08:32,182 file: build.py func: build line No: 146 The specified model compilation architecture: bayes-e.
2024-08-14 15:08:32,182 file: build.py func: build line No: 148 The specified model compilation optimization parameters: [].
2024-08-14 15:08:32,202 file: build.py func: build line No: 36 Start to prepare the onnx model.
2024-08-14 15:08:32,202 file: utils.py func: utils line No: 53 Input ONNX Model Information:
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 64], dtype=FLOAT32
    326:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    334:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    342:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    350:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    358:                  shape=[1, 20, 20, 80], dtype=FLOAT32
2024-08-14 15:08:32,370 file: build.py func: build line No: 39 End to prepare the onnx model.
2024-08-14 15:08:32,578 file: build.py func: build line No: 186 Saving model: yolov8n_bayese_640x640_nchw_original_float_model.onnx.
2024-08-14 15:08:32,579 file: build.py func: build line No: 36 Start to optimize the model.
2024-08-14 15:08:32,860 file: build.py func: build line No: 39 End to optimize the model.
2024-08-14 15:08:32,872 file: build.py func: build line No: 186 Saving model: yolov8n_bayese_640x640_nchw_optimized_float_model.onnx.
2024-08-14 15:08:32,872 file: build.py func: build line No: 36 Start to calibrate the model.
2024-08-14 15:08:33,097 file: calibration_data_set.py func: calibration_data_set line No: 82 input name: images,  number_of_samples: 50
2024-08-14 15:08:33,098 file: calibration_data_set.py func: calibration_data_set line No: 93 There are 50 samples in the calibration data set.
2024-08-14 15:08:33,105 file: default_calibrater.py func: default_calibrater line No: 122 Run calibration model with default calibration method.
2024-08-14 15:08:34,199 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-14 15:09:26,063 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-14 15:09:36,342 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-14 15:10:06,999 file: default_calibrater.py func: default_calibrater line No: 211 Select max-percentile:percentile=0.99995 method.
2024-08-14 15:10:10,312 file: build.py func: build line No: 39 End to calibrate the model.
2024-08-14 15:10:10,339 file: build.py func: build line No: 186 Saving model: yolov8n_bayese_640x640_nchw_calibrated_model.onnx.
2024-08-14 15:10:10,340 file: build.py func: build line No: 36 Start to quantize the model.
2024-08-14 15:10:11,134 file: build.py func: build line No: 39 End to quantize the model.
2024-08-14 15:10:11,230 file: build.py func: build line No: 186 Saving model: yolov8n_bayese_640x640_nchw_quantized_model.onnx.
2024-08-14 15:10:11,503 file: build.py func: build line No: 36 Start to compile the model with march bayes-e.
2024-08-14 15:10:11,664 file: hybrid_build.py func: hybrid_build line No: 133 Compile submodel: main_graph_subgraph_0
2024-08-14 15:10:11,847 file: hbdk_cc.py func: hbdk_cc line No: 115 hbdk-cc parameters:['--O3', '--core-num', '1', '--fast', '--input-layout', 'NHWC', '--output-layout', 'NHWC', '--input-source', 'ddr']
2024-08-14 15:10:11,847 file: hbdk_cc.py func: hbdk_cc line No: 116 hbdk-cc command used:hbdk-cc -f hbir -m /tmp/tmp6ndgeafr/main_graph_subgraph_0.hbir -o /tmp/tmp6ndgeafr/main_graph_subgraph_0.hbm --march bayes-e --progressbar --O3 --core-num 1 --fast --input-layout NHWC --output-layout NHWC --input-source ddr
2024-08-14 15:10:11,847 file: tool_utils.py func: tool_utils line No: 317 Can not find the scale for node HZ_PREPROCESS_FOR_images_NCHW2NHWC_LayoutConvert_Input0
2024-08-14 15:12:43,142 file: tool_utils.py func: tool_utils line No: 322 consumed time 151.246
2024-08-14 15:12:43,246 file: tool_utils.py func: tool_utils line No: 322 FPS=260.75, latency = 3835.1 us, DDR = 14293568 bytes   (see main_graph_subgraph_0.html)
2024-08-14 15:12:43,338 file: build.py func: build line No: 39 End to compile the model with march bayes-e.
2024-08-14 15:12:43,360 file: print_node_info.py func: print_node_info line No: 57 The converted model node information:
================================================================================================================================
Node                                                ON   Subgraph  Type          Cosine Similarity  Threshold   In/Out DataType  
---------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess  0.999965           127.000000  int8/int8        
/model.0/conv/Conv                                  BPU  id(0)     Conv          0.999962           1.000000    int8/int8        
/model.0/act/Mul                                    BPU  id(0)     HzSwish       0.999422           35.569626   int8/int8        
/model.1/conv/Conv                                  BPU  id(0)     Conv          0.996618           33.714809   int8/int8        
/model.1/act/Mul                                    BPU  id(0)     HzSwish       0.995918           86.623657   int8/int8        
/model.2/cv1/conv/Conv                              BPU  id(0)     Conv          0.994471           76.724045   int8/int8        
/model.2/cv1/act/Mul                                BPU  id(0)     HzSwish       0.992270           54.829262   int8/int8        
/model.2/Split                                      BPU  id(0)     Split         0.992206           17.175674   int8/int8        
/model.2/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.983177           17.175674   int8/int8        
/model.2/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.983973           27.448835   int8/int8        
/model.2/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.971784           19.258526   int8/int8        
/model.2/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.980951           22.123299   int8/int8        
UNIT_CONV_FOR_/model.2/m.0/Add                      BPU  id(0)     Conv          0.992206           17.175674   int8/int8        
/model.2/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.2/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.2/Concat                                     BPU  id(0)     Concat        0.989789           17.175674   int8/int8        
/model.2/cv2/conv/Conv                              BPU  id(0)     Conv          0.987008           18.081743   int8/int8        
/model.2/cv2/act/Mul                                BPU  id(0)     HzSwish       0.987399           19.032717   int8/int8        
/model.3/conv/Conv                                  BPU  id(0)     Conv          0.981678           7.955267    int8/int8        
/model.3/act/Mul                                    BPU  id(0)     HzSwish       0.988917           6.957337    int8/int8        
/model.4/cv1/conv/Conv                              BPU  id(0)     Conv          0.980457           5.671463    int8/int8        
/model.4/cv1/act/Mul                                BPU  id(0)     HzSwish       0.982962           7.908332    int8/int8        
/model.4/Split                                      BPU  id(0)     Split         0.991741           4.750612    int8/int8        
/model.4/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.981493           4.750612    int8/int8        
/model.4/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.976070           5.481678    int8/int8        
/model.4/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.977046           3.511999    int8/int8        
/model.4/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.981303           5.974695    int8/int8        
UNIT_CONV_FOR_/model.4/m.0/Add                      BPU  id(0)     Conv          0.991741           4.750612    int8/int8        
/model.4/m.1/cv1/conv/Conv                          BPU  id(0)     Conv          0.985302           5.298987    int8/int8        
/model.4/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish       0.979799           6.360068    int8/int8        
/model.4/m.1/cv2/conv/Conv                          BPU  id(0)     Conv          0.981934           2.825270    int8/int8        
/model.4/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish       0.985339           7.777061    int8/int8        
UNIT_CONV_FOR_/model.4/m.1/Add                      BPU  id(0)     Conv          0.992063           5.298987    int8/int8        
/model.4/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/Concat                                     BPU  id(0)     Concat        0.991109           4.750612    int8/int8        
/model.4/cv2/conv/Conv                              BPU  id(0)     Conv          0.975370           7.386191    int8/int8        
/model.4/cv2/act/Mul                                BPU  id(0)     HzSwish       0.973180           7.086750    int8/int8        
/model.5/conv/Conv                                  BPU  id(0)     Conv          0.973787           4.055890    int8/int8        
/model.5/act/Mul                                    BPU  id(0)     HzSwish       0.968264           6.858399    int8/int8        
/model.6/cv1/conv/Conv                              BPU  id(0)     Conv          0.949884           4.230505    int8/int8        
/model.6/cv1/act/Mul                                BPU  id(0)     HzSwish       0.958711           8.624203    int8/int8        
/model.6/Split                                      BPU  id(0)     Split         0.946931           5.108355    int8/int8        
/model.6/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.968698           5.108355    int8/int8        
/model.6/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.954347           6.998667    int8/int8        
/model.6/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.964599           4.572927    int8/int8        
/model.6/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.957438           6.198850    int8/int8        
UNIT_CONV_FOR_/model.6/m.0/Add                      BPU  id(0)     Conv          0.974445           5.108355    int8/int8        
/model.6/m.1/cv1/conv/Conv                          BPU  id(0)     Conv          0.981030           5.458949    int8/int8        
/model.6/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish       0.974083           6.525147    int8/int8        
/model.6/m.1/cv2/conv/Conv                          BPU  id(0)     Conv          0.971755           4.220335    int8/int8        
/model.6/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish       0.973278           8.886635    int8/int8        
UNIT_CONV_FOR_/model.6/m.1/Add                      BPU  id(0)     Conv          0.968867           5.458949    int8/int8        
/model.6/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/Concat                                     BPU  id(0)     Concat        0.973461           5.108355    int8/int8        
/model.6/cv2/conv/Conv                              BPU  id(0)     Conv          0.977780           6.570846    int8/int8        
/model.6/cv2/act/Mul                                BPU  id(0)     HzSwish       0.964863           6.025805    int8/int8        
/model.7/conv/Conv                                  BPU  id(0)     Conv          0.967855           3.736946    int8/int8        
/model.7/act/Mul                                    BPU  id(0)     HzSwish       0.926349           6.648402    int8/int8        
/model.8/cv1/conv/Conv                              BPU  id(0)     Conv          0.956976           4.346023    int8/int8        
/model.8/cv1/act/Mul                                BPU  id(0)     HzSwish       0.929664           8.274260    int8/int8        
/model.8/Split                                      BPU  id(0)     Split         0.922219           6.538882    int8/int8        
/model.8/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.969923           6.538882    int8/int8        
/model.8/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.950819           7.725489    int8/int8        
/model.8/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.947912           6.170047    int8/int8        
/model.8/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.942528           11.247433   int8/int8        
UNIT_CONV_FOR_/model.8/m.0/Add                      BPU  id(0)     Conv          0.955826           6.538882    int8/int8        
/model.8/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.8/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.8/Concat                                     BPU  id(0)     Concat        0.932878           6.538882    int8/int8        
/model.8/cv2/conv/Conv                              BPU  id(0)     Conv          0.936081           8.291936    int8/int8        
/model.8/cv2/act/Mul                                BPU  id(0)     HzSwish       0.895886           8.529250    int8/int8        
/model.9/cv1/conv/Conv                              BPU  id(0)     Conv          0.968476           5.430407    int8/int8        
/model.9/cv1/act/Mul                                BPU  id(0)     HzSwish       0.966452           6.071580    int8/int8        
/model.9/m/MaxPool                                  BPU  id(0)     MaxPool       0.987349           8.527267    int8/int8        
/model.9/m_1/MaxPool                                BPU  id(0)     MaxPool       0.995314           8.527267    int8/int8        
/model.9/m_2/MaxPool                                BPU  id(0)     MaxPool       0.997249           8.527267    int8/int8        
/model.9/Concat                                     BPU  id(0)     Concat        0.991306           8.527267    int8/int8        
/model.9/cv2/conv/Conv                              BPU  id(0)     Conv          0.966815           8.527267    int8/int8        
/model.9/cv2/act/Mul                                BPU  id(0)     HzSwish       0.923252           7.715001    int8/int8        
/model.10/Resize                                    BPU  id(0)     Resize        0.923258           4.932501    int8/int8        
/model.10/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
...el.6/cv2/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8        
/model.11/Concat                                    BPU  id(0)     Concat        0.938195           4.932501    int8/int8        
/model.12/cv1/conv/Conv                             BPU  id(0)     Conv          0.959945           4.693886    int8/int8        
/model.12/cv1/act/Mul                               BPU  id(0)     HzSwish       0.958624           6.333393    int8/int8        
/model.12/Split                                     BPU  id(0)     Split         0.970303           4.907437    int8/int8        
/model.12/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.974184           4.907437    int8/int8        
/model.12/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.952445           7.255689    int8/int8        
/model.12/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.958570           3.604169    int8/int8        
/model.12/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.958411           6.795384    int8/int8        
/model.12/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.12/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.12/Concat                                    BPU  id(0)     Concat        0.957776           4.907437    int8/int8        
/model.12/cv2/conv/Conv                             BPU  id(0)     Conv          0.953549           5.049700    int8/int8        
/model.12/cv2/act/Mul                               BPU  id(0)     HzSwish       0.949341           7.228042    int8/int8        
/model.13/Resize                                    BPU  id(0)     Resize        0.949364           4.618764    int8/int8        
/model.13/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
...el.4/cv2/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8        
/model.14/Concat                                    BPU  id(0)     Concat        0.958421           4.618764    int8/int8        
/model.15/cv1/conv/Conv                             BPU  id(0)     Conv          0.980578           4.258184    int8/int8        
/model.15/cv1/act/Mul                               BPU  id(0)     HzSwish       0.988527           6.334450    int8/int8        
/model.15/Split                                     BPU  id(0)     Split         0.991490           2.849286    int8/int8        
/model.15/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.980763           2.849286    int8/int8        
/model.15/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.983391           4.659245    int8/int8        
/model.15/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.983341           2.519424    int8/int8        
/model.15/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.990863           5.531162    int8/int8        
/model.15/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.15/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.15/Concat                                    BPU  id(0)     Concat        0.989467           2.849286    int8/int8        
/model.15/cv2/conv/Conv                             BPU  id(0)     Conv          0.986367           4.106707    int8/int8        
/model.15/cv2/act/Mul                               BPU  id(0)     HzSwish       0.989680           5.877306    int8/int8        
/model.16/conv/Conv                                 BPU  id(0)     Conv          0.973825           3.469230    int8/int8        
/model.22/cv2.0/cv2.0.0/conv/Conv                   BPU  id(0)     Conv          0.977691           3.469230    int8/int8        
/model.22/cv3.0/cv3.0.0/conv/Conv                   BPU  id(0)     Conv          0.982979           3.469230    int8/int8        
/model.16/act/Mul                                   BPU  id(0)     HzSwish       0.970928           6.168325    int8/int8        
/model.22/cv2.0/cv2.0.0/act/Mul                     BPU  id(0)     HzSwish       0.973318           9.967900    int8/int8        
/model.22/cv3.0/cv3.0.0/act/Mul                     BPU  id(0)     HzSwish       0.974259           7.393052    int8/int8        
/model.17/Concat                                    BPU  id(0)     Concat        0.956750           4.618764    int8/int8        
/model.22/cv2.0/cv2.0.1/conv/Conv                   BPU  id(0)     Conv          0.956873           4.073056    int8/int8        
/model.22/cv3.0/cv3.0.1/conv/Conv                   BPU  id(0)     Conv          0.961494           3.280292    int8/int8        
/model.18/cv1/conv/Conv                             BPU  id(0)     Conv          0.951636           4.618764    int8/int8        
/model.22/cv2.0/cv2.0.1/act/Mul                     BPU  id(0)     HzSwish       0.961625           25.584229   int8/int8        
/model.22/cv3.0/cv3.0.1/act/Mul                     BPU  id(0)     HzSwish       0.977156           30.892729   int8/int8        
/model.18/cv1/act/Mul                               BPU  id(0)     HzSwish       0.943619           6.664549    int8/int8        
/model.22/cv2.0/cv2.0.2/Conv                        BPU  id(0)     Conv          0.988656           25.441504   int8/int32       
/model.22/cv3.0/cv3.0.2/Conv                        BPU  id(0)     Conv          0.999567           22.188290   int8/int32       
/model.18/Split                                     BPU  id(0)     Split         0.879665           4.825432    int8/int8        
/model.18/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.979081           4.825432    int8/int8        
/model.18/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.974407           6.217128    int8/int8        
/model.18/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.939689           3.147952    int8/int8        
/model.18/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.947816           9.928068    int8/int8        
/model.18/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.18/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.18/Concat                                    BPU  id(0)     Concat        0.945172           4.825432    int8/int8        
/model.18/cv2/conv/Conv                             BPU  id(0)     Conv          0.960381           6.118400    int8/int8        
/model.18/cv2/act/Mul                               BPU  id(0)     HzSwish       0.960790           9.250177    int8/int8        
/model.19/conv/Conv                                 BPU  id(0)     Conv          0.945044           3.847975    int8/int8        
/model.22/cv2.1/cv2.1.0/conv/Conv                   BPU  id(0)     Conv          0.948062           3.847975    int8/int8        
/model.22/cv3.1/cv3.1.0/conv/Conv                   BPU  id(0)     Conv          0.965341           3.847975    int8/int8        
/model.19/act/Mul                                   BPU  id(0)     HzSwish       0.925460           7.624742    int8/int8        
/model.22/cv2.1/cv2.1.0/act/Mul                     BPU  id(0)     HzSwish       0.934170           12.168712   int8/int8        
/model.22/cv3.1/cv3.1.0/act/Mul                     BPU  id(0)     HzSwish       0.951460           8.731288    int8/int8        
/model.20/Concat                                    BPU  id(0)     Concat        0.924010           4.932501    int8/int8        
/model.22/cv2.1/cv2.1.1/conv/Conv                   BPU  id(0)     Conv          0.942393           7.182148    int8/int8        
/model.22/cv3.1/cv3.1.1/conv/Conv                   BPU  id(0)     Conv          0.953833           5.352224    int8/int8        
/model.21/cv1/conv/Conv                             BPU  id(0)     Conv          0.949579           4.932501    int8/int8        
/model.22/cv2.1/cv2.1.1/act/Mul                     BPU  id(0)     HzSwish       0.944863           34.185658   int8/int8        
/model.22/cv3.1/cv3.1.1/act/Mul                     BPU  id(0)     HzSwish       0.963382           67.102295   int8/int8        
/model.21/cv1/act/Mul                               BPU  id(0)     HzSwish       0.930833           8.453249    int8/int8        
/model.22/cv2.1/cv2.1.2/Conv                        BPU  id(0)     Conv          0.984995           34.145580   int8/int32       
/model.22/cv3.1/cv3.1.2/Conv                        BPU  id(0)     Conv          0.998982           56.054050   int8/int32       
/model.21/Split                                     BPU  id(0)     Split         0.942453           5.449242    int8/int8        
/model.21/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.960289           5.449242    int8/int8        
/model.21/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.937691           7.658525    int8/int8        
/model.21/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.948792           4.649248    int8/int8        
/model.21/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.928308           10.225516   int8/int8        
/model.21/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.21/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.21/Concat                                    BPU  id(0)     Concat        0.929061           5.449242    int8/int8        
/model.21/cv2/conv/Conv                             BPU  id(0)     Conv          0.957589           7.296426    int8/int8        
/model.21/cv2/act/Mul                               BPU  id(0)     HzSwish       0.916133           15.406893   int8/int8        
/model.22/cv2.2/cv2.2.0/conv/Conv                   BPU  id(0)     Conv          0.936696           5.069451    int8/int8        
/model.22/cv3.2/cv3.2.0/conv/Conv                   BPU  id(0)     Conv          0.956778           5.069451    int8/int8        
/model.22/cv2.2/cv2.2.0/act/Mul                     BPU  id(0)     HzSwish       0.926717           12.501591   int8/int8        
/model.22/cv3.2/cv3.2.0/act/Mul                     BPU  id(0)     HzSwish       0.945714           14.597645   int8/int8        
/model.22/cv2.2/cv2.2.1/conv/Conv                   BPU  id(0)     Conv          0.921961           8.993995    int8/int8        
/model.22/cv3.2/cv3.2.1/conv/Conv                   BPU  id(0)     Conv          0.957305           7.419346    int8/int8        
/model.22/cv2.2/cv2.2.1/act/Mul                     BPU  id(0)     HzSwish       0.923008           34.900085   int8/int8        
/model.22/cv3.2/cv3.2.1/act/Mul                     BPU  id(0)     HzSwish       0.964669           52.946068   int8/int8        
/model.22/cv2.2/cv2.2.2/Conv                        BPU  id(0)     Conv          0.978757           34.900085   int8/int32       
/model.22/cv3.2/cv3.2.2/Conv                        BPU  id(0)     Conv          0.998914           50.421188   int8/int32

```


## Model Training

- Model training should refer to the official ultralytics documentation, which is maintained by ultralytics and is of very high quality. There are also numerous reference materials available online, and obtaining a model with pre-trained weights similar to the official ones is not difficult.
- Please note that no program modifications are required for training, nor is there a need to modify the `forward` method.

## Performance Data

RDK X5 & RDK X5 Module  
Object Detection (COCO)  
| Model | Size (px) | Num. Classes | Params (M) | FP Precision | Q Precision | Latency/Throughput (Single-threaded) | Latency/Throughput (Multi-threaded) | Post Process Time (Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|--------------|
| YOLOv8n | 640×640 | 80 | 3.2 | 37.3 |  | 5.6ms/178.0FPS(1 thread) | 7.5ms/263.6FPS(2 threads) | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 | 44.9 |  | 12.4ms/80.2FPS(1 thread) | 21ms/94.9FPS(2 threads) | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 | 50.2 |  | 29.9ms/33.4FPS(1 thread) | 55.9ms/35.7FPS(2 threads) | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 | 52.9 |  | 57.6ms/17.3FPS(1 thread) | 111.1ms/17.9FPS(2 threads) | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 | 53.9 |  | 90.0ms/11.0FPS(1 thread) | 177.5ms/11.2FPS(2 threads) | 5 ms |


Object Detection (Open Image V7)  
| Model | Size (px) | Num. Classes | Params (M) | FP Precision | Q Precision | Average Frame Latency/Throughput (Single-threaded) | Average Frame Latency/Throughput (Multi-threaded) |
|------|------|-------|---------|---------|-------------------|--------------------|--------------------|
| YOLOv8n | 640×640 | 600 | 3.5 | 18.4 | - | - | - |
| YOLOv8s | 640×640 | 600 | 11.4 | 27.7 | - | - | - |
| YOLOv8m | 640×640 | 600 | 26.2 | 33.6 | - | - | - |
| YOLOv8l | 640×640 | 600 | 44.1 | 34.9 | - | - | - |
| YOLOv8x | 640×640 | 600 | 68.7 | 36.3 | - | - | - |

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