English | [简体中文](./README_cn.md)

# YOLOv5 Detect
- [YOLOv5 Detect](#yolov5-detect)
  - [Introduction to YOLO](#introduction-to-yolo)
  - [Performance Data (in brief)](#performance-data-in-brief)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module)
    - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module)
  - [Where to download models](#where-to-download-models)
  - [Input and output data](#input-and-output-data)
  - [Steps to follow](#steps-to-follow)
    - [YOLOv5 tag v2.0](#yolov5-tag-v20)
      - [Environment, project preparation](#environment-project-preparation)
      - [is exported as onnx](#is-exported-as-onnx)
      - [PTQ scheme Quantization transformation](#ptq-scheme-quantization-transformation)
      - [Uses the hb\_perf command to visualize the bin model](#uses-the-hb_perf-command-to-visualize-the-bin-model)
      - [Use the hrt\_model\_exec command to check the input and output of the bin model](#use-the-hrt_model_exec-command-to-check-the-input-and-output-of-the-bin-model)
    - [YOLOv5 tag v7.0](#yolov5-tag-v70)
      - [Environment, project preparation](#environment-project-preparation-1)
      - [is exported as onnx](#is-exported-as-onnx-1)
      - [PTQ scheme Quantization transformation](#ptq-scheme-quantization-transformation-1)
      - [Uses the hb\_perf command to visualize the bin model](#uses-the-hb_perf-command-to-visualize-the-bin-model-1)
      - [Use the hrt\_model\_exec command to check the input and output of the bin model](#use-the-hrt_model_exec-command-to-check-the-input-and-output-of-the-bin-model-1)
    - [Partial compilation log reference](#partial-compilation-log-reference)
  - [Model training](#model-training)
  - [Performance data](#performance-data)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module-1)
    - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module-1)
  - [Feedback](#feedback)
  - [References](#references)



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


## Performance Data (in brief)
### RDK X5 & RDK X5 Module
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 13.1 FPS | 12 ms |


### RDK X3 & RDK X3 Module
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 38.2 FPS | 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 3.9 FPS | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 37.2 FPS | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 20.9 FPS | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 3.6 FPS | 13 ms |

Note: Detailed performance data are at the end of the paper.

## Where to download models
See './model/download.md '

## Input and output data
- Input: 1x3x640x640, dtype=UINT8
- Output 0: [1, 80, 80, 255], dtype=FLOAT32
- Output 1: [1, 40, 40, 255], dtype=FLOAT32
- Output 2: [1, 20, 20, 255], dtype=FLOAT32

## Steps to follow

Note: Any No such file or directory, No module named "xxx", command not found. If you do not understand the modification process, please go to the developer community to start with YOLOv5.


### YOLOv5 tag v2.0

#### Environment, project preparation
- Download the ultralytics/yolov5 repository and configure the environment according to the ultralytics documentation

```bash
$ git clone https://github.com/ultralytics/yolov5.git
```

- Go to your local repository and switch branches

If tag v2.0 is required, the version corresponding to the LeakyRelu activation function.

Warehouse link: https://github.com/ultralytics/yolov5/tree/v2.0

The release link is: https://github.com/ultralytics/yolov5/releases/tag/v2.0

```bash
# Go to the local repository
$ cd yolov5
# Switch to v2.0 branch
$git checkout v2.0
# check
$ git branch
# Display content
* (HEAD detached at v2.0)
```

- Download the official pre-trained weights. Here, the YOLOv5s-v2.0-Detect model with 7.5 million parameters is taken as an example. Note that the corresponding tag version can only use the corresponding tag version of the pt model, and cannot be mixed.

```bash
Wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt - O yolov5s_tag2. 0. Pt
```

#### is exported as onnx
- Modify the output header to ensure 4-dimensional NHWC output
Modify the './models/yolo.py 'file,' Detect 'class,' forward 'method, about 22 lines.
Note: It is recommended that you keep the original 'forward' method, for example, change it to a different name 'forward_', so that it can be changed back during training.

```bash
def forward(self, x):
    return [self.m[i](x[i]).permute(0,2,3,1).contiguous() for i in range(self.nl)]
```

- Copy the './models/export.py 'file to the root directory
```bash
cp ./models/export.py export.py
```
- Make the following change on line 14 of 'export.py' to change the input pt file path and onnx resolution.
```python
parser.add_argument('--weights', type=str, default='./yolov5s_tag2.0.pt', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
```

 - Comment out 32 lines and export 60 lines to a different format, while modifying the following code block: The main opset version, with an onnx simplify program that does some graph optimizations and constant folding.
```python
# ONNX export
try:
    import onnx
    from onnxsim import simplify

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['small', 'medium', 'big'])
    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    # simplify
    onnx_model, check = simplify(
        onnx_model,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(onnx_model, f)
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)
```

 - Run the 'export.py' script to export to onnx.
```bash
$ python3 export.py
```
#### PTQ scheme Quantization transformation
- Refer to the Tiengong Kaiwu toolchain manual and OE package, check the model, all operators are on the BPU, and compile. The yaml file is in the './ptq_yamls' directory.
```bash
(bpu_docker) $hb_mapper checker -- modeltype onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

#### Uses the hb_perf command to visualize the bin model
```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```
The following results can be found in the 'hb_perf_result' directory:
! [] (imgs/yolov5s_tag_v2. 0 _detect_640x640_bayese_nv12. PNG)

#### Use the hrt_model_exec command to check the input and output of the bin model
```bash
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

Display the inputs and outputs of the model

```bash
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-09-11,20:21:38.941.75) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 249.627ms.
This model file has 1 model:
[yolov5s_tag_v2.0_detect_640x640_bayese_nv12]
---------------------------------------------------------------------
[model name]: yolov5s_tag_v2.0_detect_640x640_bayese_nv12

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
name: small
valid shape: (1,80,80,255,)
aligned shape: (1,80,80,255,)
aligned byte size: 6528000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (6528000,81600,1020,4,)

output[1]: 
name: medium
valid shape: (1,40,40,255,)
aligned shape: (1,40,40,255,)
aligned byte size: 1632000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1632000,40800,1020,4,)

output[2]: 
name: big
valid shape: (1,20,20,255,)
aligned shape: (1,20,20,255,)
aligned byte size: 408000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (408000,20400,1020,4,)
```

### YOLOv5 tag v7.0
#### Environment, project preparation
- Download the ultralytics/yolov5 repository and configure the environment according to the ultralytics documentation

```bash
$ git clone https://github.com/ultralytics/yolov5.git
```

- Go to the local repository

```bash
cd yolov5
```

- Toggle the corresponding branch
If tag v7.0 is required, the version corresponding to the Sigmoid activation function.
Warehouse link: https://github.com/ultralytics/yolov5/tree/v7.0
The release link is: https://github.com/ultralytics/yolov5/releases/tag/v7.0
```bash
# toggle
$checkout v7.0
# check
$ git branch
# Display content
* (HEAD detached at v7.0)
```

- Download the official pre-trained weights, here we take the 1.9 million parameter YOLOv5n-Detect model as an example, note that the corresponding tag version can only use the corresponding tag version of the pt model, not mixed.

```bash
# tag v7.0
Wget HTTP: / / https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

#### is exported as onnx
Open the export.py file and make the following changes: It is recommended that you make a copy before making changes.
- ~ 612 lines, modified as follows
```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include',
                    nargs='+',
                    default=['onnx'],
                    help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```


 - About 552 lines. Comment out the following two lines
```python
# shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
# LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")
```

 - On line 121, the following changes are made
```python
torch.onnx.export(
    model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
    im.cpu() if dynamic else im,
    f,
    verbose=False,
    opset_version=opset,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['small', 'medium', 'big'],
    dynamic_axes=dynamic or None)
```

 - 运行`export.py`脚本, 导出为onnx.
```bash
$ python3 export.py
```
#### PTQ scheme Quantization transformation
- Refer to the Tiengong Kaiwu toolchain manual and OE package, check the model, all operators are on the BPU, and compile. The yaml file is in the './ptq_yamls' directory.
```bash
(bpu_docker) $hb_mapper checker -- modeltype onnx --march bayes-e --model yolov5s_tag_v7.0_detect.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```


#### Uses the hb_perf command to visualize the bin model

```bash
hb_perf yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```
The following results can be found in the 'hb_perf_result' directory:

! [] (imgs/yolov5n_tag_v7. 0 _detect_640x640_bayese_nv12. PNG)


#### Use the hrt_model_exec command to check the input and output of the bin model
```bash
hrt_model_exec model_info --model_file yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```

Display the inputs and outputs of the model

```bash
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-09-11,20:09:38.351.997) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 141.097ms.
This model file has 1 model:
[yolov5n_tag_v7.0_detect_640x640_bayese_nv12]
---------------------------------------------------------------------
[model name]: yolov5n_tag_v7.0_detect_640x640_bayese_nv12

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
name: small
valid shape: (1,80,80,255,)
aligned shape: (1,80,80,255,)
aligned byte size: 6528000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (6528000,81600,1020,4,)

output[1]: 
name: medium
valid shape: (1,40,40,255,)
aligned shape: (1,40,40,255,)
aligned byte size: 1632000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1632000,40800,1020,4,)

output[2]: 
name: big
valid shape: (1,20,20,255,)
aligned shape: (1,20,20,255,)
aligned byte size: 408000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (408000,20400,1020,4,)
```

### Partial compilation log reference
Take the compilation log of 'yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin' as an example, you can see.
- The toolchain estimates that YOLOv5n_v7.0 runs about 288.9FPS, which is actually slightly slower due to the pre-processing, quantization nodes and some anti-quantization nodes running on the CPU. The measured throughput on X5 can reach 277.2FPS with 3 threads.
- The transpose node in the tail satisfies passive quantization logic and supports to be accelerated by BPU without affecting its parent node Conv convolution operator to output int32 high precision.
-All nodes have cosine similarity > 0.9, and the vast majority of nodes have cosine similarity > 0.95, as expected.
-All operators are on the BPU and there is only 1 BPU subgraph for the whole bin model.

```bash
ONNX IR version:          6
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    small:                shape=[1, 80, 80, 255], dtype=FLOAT32
    medium:               shape=[1, 40, 40, 255], dtype=FLOAT32
    big:                  shape=[1, 20, 20, 255], dtype=FLOAT32
2024-09-11 15:44:40,195 file: build.py func: build line No: 39 End to prepare the onnx model.
2024-09-11 15:44:40,450 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_original_float_model.onnx.
2024-09-11 15:44:40,450 file: build.py func: build line No: 36 Start to optimize the model.
2024-09-11 15:44:40,800 file: build.py func: build line No: 39 End to optimize the model.
2024-09-11 15:44:40,806 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_optimized_float_model.onnx.
2024-09-11 15:44:40,806 file: build.py func: build line No: 36 Start to calibrate the model.
2024-09-11 15:44:41,009 file: calibration_data_set.py func: calibration_data_set line No: 82 input name: images,  number_of_samples: 50
2024-09-11 15:44:41,009 file: calibration_data_set.py func: calibration_data_set line No: 93 There are 50 samples in the calibration data set.
2024-09-11 15:44:41,012 file: default_calibrater.py func: default_calibrater line No: 122 Run calibration model with default calibration method.
2024-09-11 15:44:42,100 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-09-11 15:44:46,659 file: ort.py func: ort line No: 179 Reset batch_size=1 and execute forward again...
2024-09-11 15:47:07,752 file: default_calibrater.py func: default_calibrater line No: 140 Select max-percentile:percentile=0.99995 method.
2024-09-11 15:47:10,778 file: build.py func: build line No: 39 End to calibrate the model.
2024-09-11 15:47:10,800 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_calibrated_model.onnx.
2024-09-11 15:47:10,801 file: build.py func: build line No: 36 Start to quantize the model.
2024-09-11 15:47:12,347 file: build.py func: build line No: 39 End to quantize the model.
2024-09-11 15:47:12,408 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_quantized_model.onnx.
2024-09-11 15:47:12,732 file: build.py func: build line No: 36 Start to compile the model with march bayes-e.
2024-09-11 15:47:12,867 file: hybrid_build.py func: hybrid_build line No: 133 Compile submodel: main_graph_subgraph_0
2024-09-11 15:47:13,097 file: hbdk_cc.py func: hbdk_cc line No: 115 hbdk-cc parameters:['--O3', '--core-num', '1', '--fast', '--input-layout', 'NHWC', '--output-layout', 'NHWC', '--input-source', 'pyramid']
2024-09-11 15:47:13,097 file: hbdk_cc.py func: hbdk_cc line No: 116 hbdk-cc command used:hbdk-cc -f hbir -m /tmp/tmp_prdnrjp/main_graph_subgraph_0.hbir -o /tmp/tmp_prdnrjp/main_graph_subgraph_0.hbm --march bayes-e --progressbar --O3 --core-num 1 --fast --input-layout NHWC --output-layout NHWC --input-source pyramid
2024-09-11 15:49:27,117 file: tool_utils.py func: tool_utils line No: 326 consumed time 133.975
2024-09-11 15:49:27,211 file: tool_utils.py func: tool_utils line No: 326 FPS=288.9, latency = 3461.4 us, DDR = 16197440 bytes   (see main_graph_subgraph_0.html)
2024-09-11 15:49:27,273 file: build.py func: build line No: 39 End to compile the model with march bayes-e.
2024-09-11 15:49:27,408 file: print_node_info.py func: print_node_info line No: 57 The converted model node information:
================================================================================================================================
Node                                                ON   Subgraph  Type          Cosine Similarity  Threshold   In/Out DataType  
---------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess  0.999967           127.000000  int8/int8        
/model.0/conv/Conv                                  BPU  id(0)     Conv          0.999724           1.127231    int8/int8        
/model.0/act/Mul                                    BPU  id(0)     HzSwish       0.999239           22.935776   int8/int8        
/model.1/conv/Conv                                  BPU  id(0)     Conv          0.996699           20.392370   int8/int8        
/model.1/act/Mul                                    BPU  id(0)     HzSwish       0.995564           69.509407   int8/int8        
/model.2/cv1/conv/Conv                              BPU  id(0)     Conv          0.996266           61.730789   int8/int8        
/model.2/cv1/act/Mul                                BPU  id(0)     HzSwish       0.996379           32.784687   int8/int8        
/model.2/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.987249           14.013276   int8/int8        
/model.2/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.987381           24.406996   int8/int8        
/model.2/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.983127           10.855639   int8/int8        
/model.2/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.987979           15.401920   int8/int8        
UNIT_CONV_FOR_/model.2/m/m.0/Add                    BPU  id(0)     Conv          0.996379           14.013276   int8/int8        
/model.2/cv2/conv/Conv                              BPU  id(0)     Conv          0.992321           61.730789   int8/int8        
/model.2/cv2/act/Mul                                BPU  id(0)     HzSwish       0.993225           62.547588   int8/int8        
/model.2/Concat                                     BPU  id(0)     Concat        0.993569           17.940353   int8/int8        
/model.2/cv3/conv/Conv                              BPU  id(0)     Conv          0.988253           17.940353   int8/int8        
/model.2/cv3/act/Mul                                BPU  id(0)     HzSwish       0.989918           10.508739   int8/int8        
/model.3/conv/Conv                                  BPU  id(0)     Conv          0.982787           7.924888    int8/int8        
/model.3/act/Mul                                    BPU  id(0)     HzSwish       0.988654           8.092022    int8/int8        
/model.4/cv1/conv/Conv                              BPU  id(0)     Conv          0.992336           5.621616    int8/int8        
/model.4/cv1/act/Mul                                BPU  id(0)     HzSwish       0.993258           3.872177    int8/int8        
/model.4/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.987385           2.466782    int8/int8        
/model.4/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.989359           5.410062    int8/int8        
/model.4/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.982610           4.081088    int8/int8        
/model.4/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.989846           5.585560    int8/int8        
UNIT_CONV_FOR_/model.4/m/m.0/Add                    BPU  id(0)     Conv          0.993258           2.466782    int8/int8        
/model.4/m/m.1/cv1/conv/Conv                        BPU  id(0)     Conv          0.980726           4.777254    int8/int8        
/model.4/m/m.1/cv1/act/Mul                          BPU  id(0)     HzSwish       0.983446           4.889826    int8/int8        
/model.4/m/m.1/cv2/conv/Conv                        BPU  id(0)     Conv          0.982318           3.377826    int8/int8        
/model.4/m/m.1/cv2/act/Mul                          BPU  id(0)     HzSwish       0.985909           7.605175    int8/int8        
UNIT_CONV_FOR_/model.4/m/m.1/Add                    BPU  id(0)     Conv          0.992652           4.777254    int8/int8        
/model.4/cv2/conv/Conv                              BPU  id(0)     Conv          0.982260           5.621616    int8/int8        
/model.4/cv2/act/Mul                                BPU  id(0)     HzSwish       0.985327           7.687179    int8/int8        
/model.4/Concat                                     BPU  id(0)     Concat        0.990299           7.244858    int8/int8        
/model.4/cv3/conv/Conv                              BPU  id(0)     Conv          0.982896           7.244858    int8/int8        
/model.4/cv3/act/Mul                                BPU  id(0)     HzSwish       0.981238           5.655891    int8/int8        
/model.5/conv/Conv                                  BPU  id(0)     Conv          0.980117           3.714849    int8/int8        
/model.5/act/Mul                                    BPU  id(0)     HzSwish       0.976186           6.713475    int8/int8        
/model.6/cv1/conv/Conv                              BPU  id(0)     Conv          0.988606           3.879521    int8/int8        
/model.6/cv1/act/Mul                                BPU  id(0)     HzSwish       0.983306           4.371953    int8/int8        
/model.6/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.973573           1.150585    int8/int8        
/model.6/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.964966           5.939588    int8/int8        
/model.6/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.964386           5.097388    int8/int8        
/model.6/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.941962           3.874449    int8/int8        
UNIT_CONV_FOR_/model.6/m/m.0/Add                    BPU  id(0)     Conv          0.983306           1.150585    int8/int8        
/model.6/m/m.1/cv1/conv/Conv                        BPU  id(0)     Conv          0.948590           2.688997    int8/int8        
/model.6/m/m.1/cv1/act/Mul                          BPU  id(0)     HzSwish       0.941272           5.648884    int8/int8        
/model.6/m/m.1/cv2/conv/Conv                        BPU  id(0)     Conv          0.944876           4.614841    int8/int8        
/model.6/m/m.1/cv2/act/Mul                          BPU  id(0)     HzSwish       0.943065           5.309185    int8/int8        
UNIT_CONV_FOR_/model.6/m/m.1/Add                    BPU  id(0)     Conv          0.963641           2.688997    int8/int8        
/model.6/m/m.2/cv1/conv/Conv                        BPU  id(0)     Conv          0.950161           5.153328    int8/int8        
/model.6/m/m.2/cv1/act/Mul                          BPU  id(0)     HzSwish       0.941146           5.703106    int8/int8        
/model.6/m/m.2/cv2/conv/Conv                        BPU  id(0)     Conv          0.933608           4.234711    int8/int8        
/model.6/m/m.2/cv2/act/Mul                          BPU  id(0)     HzSwish       0.939521           6.980769    int8/int8        
UNIT_CONV_FOR_/model.6/m/m.2/Add                    BPU  id(0)     Conv          0.942982           5.153328    int8/int8        
/model.6/cv2/conv/Conv                              BPU  id(0)     Conv          0.964446           3.879521    int8/int8        
/model.6/cv2/act/Mul                                BPU  id(0)     HzSwish       0.967606           6.328825    int8/int8        
/model.6/Concat                                     BPU  id(0)     Concat        0.956309           6.556569    int8/int8        
/model.6/cv3/conv/Conv                              BPU  id(0)     Conv          0.973727           6.556569    int8/int8        
/model.6/cv3/act/Mul                                BPU  id(0)     HzSwish       0.957020           5.989245    int8/int8        
/model.7/conv/Conv                                  BPU  id(0)     Conv          0.948621           4.072040    int8/int8        
/model.7/act/Mul                                    BPU  id(0)     HzSwish       0.904375           6.163434    int8/int8        
/model.8/cv1/conv/Conv                              BPU  id(0)     Conv          0.981177           4.374038    int8/int8        
/model.8/cv1/act/Mul                                BPU  id(0)     HzSwish       0.973906           4.674935    int8/int8        
/model.8/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.896303           1.916546    int8/int8        
/model.8/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.869613           8.486115    int8/int8        
/model.8/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.839805           8.420576    int8/int8        
/model.8/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.865418           8.457147    int8/int8        
UNIT_CONV_FOR_/model.8/m/m.0/Add                    BPU  id(0)     Conv          0.973906           1.916546    int8/int8        
/model.8/cv2/conv/Conv                              BPU  id(0)     Conv          0.911671           4.374038    int8/int8        
/model.8/cv2/act/Mul                                BPU  id(0)     HzSwish       0.893293           7.450050    int8/int8        
/model.8/Concat                                     BPU  id(0)     Concat        0.866763           6.790506    int8/int8        
/model.8/cv3/conv/Conv                              BPU  id(0)     Conv          0.834800           6.790506    int8/int8        
/model.8/cv3/act/Mul                                BPU  id(0)     HzSwish       0.760843           7.942293    int8/int8        
/model.9/cv1/conv/Conv                              BPU  id(0)     Conv          0.927259           4.785241    int8/int8        
/model.9/cv1/act/Mul                                BPU  id(0)     HzSwish       0.932425           5.132613    int8/int8        
/model.9/m/MaxPool                                  BPU  id(0)     MaxPool       0.979879           6.074298    int8/int8        
/model.9/m_1/MaxPool                                BPU  id(0)     MaxPool       0.993464           6.074298    int8/int8        
/model.9/m_2/MaxPool                                BPU  id(0)     MaxPool       0.995697           6.074298    int8/int8        
/model.9/Concat                                     BPU  id(0)     Concat        0.985728           6.074298    int8/int8        
/model.9/cv2/conv/Conv                              BPU  id(0)     Conv          0.931007           6.074298    int8/int8        
/model.9/cv2/act/Mul                                BPU  id(0)     HzSwish       0.844819           6.023237    int8/int8        
/model.10/conv/Conv                                 BPU  id(0)     Conv          0.841115           5.126980    int8/int8        
/model.10/act/Mul                                   BPU  id(0)     HzSwish       0.858535           6.567430    int8/int8        
/model.11/Resize                                    BPU  id(0)     Resize        0.858531           5.970518    int8/int8        
/model.11/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
...el.6/cv3/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8        
/model.12/Concat                                    BPU  id(0)     Concat        0.902319           5.970518    int8/int8        
/model.13/cv1/conv/Conv                             BPU  id(0)     Conv          0.947352           5.094691    int8/int8        
/model.13/cv1/act/Mul                               BPU  id(0)     HzSwish       0.943633           5.366014    int8/int8        
/model.13/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.936760           2.916995    int8/int8        
/model.13/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.949624           5.177988    int8/int8        
/model.13/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.935308           3.382976    int8/int8        
/model.13/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.944433           5.233689    int8/int8        
/model.13/cv2/conv/Conv                             BPU  id(0)     Conv          0.931966           5.094691    int8/int8        
/model.13/cv2/act/Mul                               BPU  id(0)     HzSwish       0.944159           4.878307    int8/int8        
/model.13/Concat                                    BPU  id(0)     Concat        0.944225           3.718071    int8/int8        
/model.13/cv3/conv/Conv                             BPU  id(0)     Conv          0.941572           3.718071    int8/int8        
/model.13/cv3/act/Mul                               BPU  id(0)     HzSwish       0.918719           6.580989    int8/int8        
/model.14/conv/Conv                                 BPU  id(0)     Conv          0.957411           3.749895    int8/int8        
/model.14/act/Mul                                   BPU  id(0)     HzSwish       0.957810           4.743949    int8/int8        
/model.15/Resize                                    BPU  id(0)     Resize        0.957814           4.288212    int8/int8        
/model.15/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
...el.4/cv3/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8        
/model.16/Concat                                    BPU  id(0)     Concat        0.962476           4.288212    int8/int8        
/model.17/cv1/conv/Conv                             BPU  id(0)     Conv          0.974061           4.112339    int8/int8        
/model.17/cv1/act/Mul                               BPU  id(0)     HzSwish       0.983879           4.123950    int8/int8        
/model.17/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.980654           2.366742    int8/int8        
/model.17/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.982723           3.656986    int8/int8        
/model.17/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.964849           2.821121    int8/int8        
/model.17/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.959107           7.418919    int8/int8        
/model.17/cv2/conv/Conv                             BPU  id(0)     Conv          0.961160           4.112339    int8/int8        
/model.17/cv2/act/Mul                               BPU  id(0)     HzSwish       0.947822           8.756002    int8/int8        
/model.17/Concat                                    BPU  id(0)     Concat        0.951943           4.927598    int8/int8        
/model.17/cv3/conv/Conv                             BPU  id(0)     Conv          0.924104           4.927598    int8/int8        
/model.17/cv3/act/Mul                               BPU  id(0)     HzSwish       0.948889           20.271854   int8/int8        
/model.18/conv/Conv                                 BPU  id(0)     Conv          0.932838           20.202873   int8/int8        
/model.18/act/Mul                                   BPU  id(0)     HzSwish       0.929389           5.688267    int8/int8        
/model.19/Concat                                    BPU  id(0)     Concat        0.949843           4.288212    int8/int8        
/model.20/cv1/conv/Conv                             BPU  id(0)     Conv          0.908277           4.288212    int8/int8        
/model.20/cv1/act/Mul                               BPU  id(0)     HzSwish       0.917393           4.524426    int8/int8        
/model.20/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.943680           3.775438    int8/int8        
/model.20/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.932270           4.696353    int8/int8        
/model.20/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.931095           2.750212    int8/int8        
/model.20/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.941526           8.871795    int8/int8        
/model.20/cv2/conv/Conv                             BPU  id(0)     Conv          0.914818           4.288212    int8/int8        
/model.20/cv2/act/Mul                               BPU  id(0)     HzSwish       0.897235           5.465163    int8/int8        
/model.20/Concat                                    BPU  id(0)     Concat        0.923493           4.722447    int8/int8        
/model.20/cv3/conv/Conv                             BPU  id(0)     Conv          0.924307           4.722447    int8/int8        
/model.20/cv3/act/Mul                               BPU  id(0)     HzSwish       0.931528           21.215944   int8/int8        
/model.21/conv/Conv                                 BPU  id(0)     Conv          0.892469           21.054279   int8/int8        
/model.21/act/Mul                                   BPU  id(0)     HzSwish       0.886928           7.410546    int8/int8        
/model.22/Concat                                    BPU  id(0)     Concat        0.871830           5.970518    int8/int8        
/model.23/cv1/conv/Conv                             BPU  id(0)     Conv          0.855442           5.970518    int8/int8        
/model.23/cv1/act/Mul                               BPU  id(0)     HzSwish       0.805934           7.578112    int8/int8        
/model.23/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.879370           5.911558    int8/int8        
/model.23/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.848773           7.282393    int8/int8        
/model.23/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.904308           4.925912    int8/int8        
/model.23/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.890898           11.411656   int8/int8        
/model.23/cv2/conv/Conv                             BPU  id(0)     Conv          0.894634           5.970518    int8/int8        
/model.23/cv2/act/Mul                               BPU  id(0)     HzSwish       0.867974           7.260511    int8/int8        
/model.23/Concat                                    BPU  id(0)     Concat        0.878335           6.109259    int8/int8        
/model.23/cv3/conv/Conv                             BPU  id(0)     Conv          0.907323           6.109259    int8/int8        
/model.23/cv3/act/Mul                               BPU  id(0)     HzSwish       0.920191           19.757231   int8/int8        
/model.24/m.0/Conv                                  BPU  id(0)     Conv          0.996857           20.202873   int8/int32       
/model.24/m.1/Conv                                  BPU  id(0)     Conv          0.997903           21.054279   int8/int32       
/model.24/m.2/Conv                                  BPU  id(0)     Conv          0.998209           19.711018   int8/int32
```

## Model training
TODO: Training flow
Refer to the ultralytics documentation and resources on the web. Getting a model with pre-trained weights like the ones provided by ultralytics is not complicated.

## Performance data
### RDK X5 & RDK X5 Module
Object Detection (COCO)
| Model | size (pixels) | number of classes | number of parameters (M) | float-point precision <br/>(mAP:50-95) | quantization precision <br/>(mAP:50-95) | BPU latency /BPU throughput (threads) | post-processing time <br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5  | - | - | 14.3 ms / 70.0 FPS(1 thread) <br/> 18.7 ms / 106.8 FPS(2 threads) | 12 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 | - | - | 27.0 ms / 37.0 FPS(1 thread) <br/> 44.1 ms / 45.2 FPS(2 threads) | 12 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 | - | - | 50.8 ms / 19.7 FPS(1 thread) <br/> 91.5 ms / 21.8 FPS(2 threads) | 12 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 | - | - | 86.3 ms / 11.6 FPS(1 thread) <br/> 162.1 ms / 12.3 FPS(2 threads) | 12 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 | 28.0 | - | 8.5 ms / 117.4 FPS(1 thread) <br/> 8.9 ms / 223.0 FPS(2 threads) <br/> 10.7 ms / 277.2 FPS(3 threads) | 12 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 | 37.4 | - | 13.0 ms / 76.6 FPS(1 thread) <br/> 16.0 ms / 124.2 FPS(2 threads) | 12 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 | 45.4 | - | 25.7 ms / 38.8 FPS(1 thread) <br/> 41.2 ms / 48.4 FPS(2 threads) | 12 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 | 49.0 | - | 47.9 ms / 20.9 FPS(1 thread) <br/> 85.7 ms / 23.3 FPS(2 threads) | 12 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 | 50.7 | - | 81.1 ms / 12.3 FPS(1 thread) <br/> 151.9 ms / 13.1 FPS(2 threads) | 12 ms |

### RDK X3 & RDK X3 Module
Object Detection (COCO)
| Model | size (pixels) | number of classes | number of parameters (M) | float-point precision <br/>(mAP:50-95) | quantization precision <br/>(mAP:50-95) | BPU latency /BPU throughput (threads) | post-processing time <br/>(Python) |

|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | - | - | 55.7 ms / 17.9 FPS(1 thread) <br/> 61.1 ms / 32.7 FPS(2 threads) <br/> 78.1 ms / 38.2 FPS(3 threads)| 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | - | - | 512.4 ms / 2.0 FPS(1 thread) <br/> 519.7 ms / 3.8 FPS(2 threads) <br/> 762.1 ms / 3.9 FPS(3 threads) | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 28.0 | - | 85.4 ms / 11.7 FPS(1 thread) <br/> 88.9 ms / 22.4 FPS(2 threads) <br/> 121.9 ms / 32.7 FPS(4 threads) <br/> 213.0 ms / 37.2 FPS(8 threads) | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 37.4 | - | 175.4 ms / 5.7 FPS(1 thread) <br/> 182.3 ms / 11.0 FPS(2 threads) <br/> 217.9 ms / 18.2 FPS(4 threads) <br/> 378.0 ms / 20.9 FPS(8 threads) | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 50.7 | - | 1021.5 ms / 1.0 FPS(1 thread) <br/> 1024.3 ms / 2.0 FPS(2 threads) <br/> 1238.0 ms / 3.1 FPS(4 threads)<br/> 2070.0 ms / 3.6 FPS(8 threads) | 13 ms |

Note:

1. BPU latency vs. BPU throughput.
- Single thread latency is the latency of a single frame, single thread, single BPU core,BPU reasoning about a task.
- Multi-threaded frame rate means that multiple threads can simultaneously jam tasks to the BPU, each BPU core can handle the tasks of multiple threads. In general, 4 threads can control the single frame latency to be small, and eat all Bpus to 100% at the same time, getting a good balance between throughput (FPS) and frame latency. X5 BPU as a whole is quite good, generally 2 threads can eat up the BPU, frame latency and throughput are very good.
- The table generally records data where the throughput no longer increases significantly with the number of threads.
-BPU latency and BPU throughput are tested at the board side using the following commands
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. The test board is in the best condition.
The state of -X5 is the best state: 8 × A55@1.8G for CPU, full core Performance scheduling, and 1 × Bayes-e@10TOPS for BPU.
```bash
Sudo bash - c "echo 1 > / sys/devices/system/CPU/cpufreq/boost" # 1.8 Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
-X3 is in the best state: 4 × A53@1.8G for CPU, full core Performance scheduling, and 2 × Bernoulli2@5TOPS for BPU.
```bash
Sudo bash - c "echo 1 > / sys/devices/system/CPU/cpufreq/boost" # 1.8 Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
Floating-point/fixed-point mAP: 50-95 accuracy calculated using pycocotools, from the COCO dataset, refer to the Microsoft paper, here used to evaluate the accuracy degradation of on-board deployments.
4. On post-processing: At present, the post-processing of Python reconstruction on X5 only takes about 12ms from a single core and a single thread in serial, that is to say, it only takes 2 CPU cores (200% CPU occupancy, and the maximum CPU occupancy is 800%), and 166 frames of image post-processing can be completed every minute, and post-processing will not constitute a bottleneck.


## Feedback
If you are not clear about the expression of this article, please go to the Sweet potato developer community to ask questions and communicate.

[Sweet Potato Robot developer Community](developer.d-robotics.cc).


## References
[1] [GitHub: YOLOv5](https://github.com/ultralytics/yolov5)

[2] [GitHub: YOLOv8](https://github.com/ultralytics/ultralytics)

[3] [ultralytics docs](https://docs.ultralytics.com/models/yolov5/)
