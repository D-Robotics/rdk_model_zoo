[English](./README.md) | 简体中文

# YOLOv10 Detect
- [YOLOv10 Detect](#yolov10-detect)
  - [YOLO介绍](#yolo介绍)
  - [性能数据 (简要)](#性能数据-简要)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module-1)
  - [模型下载地址](#模型下载地址)
  - [输入输出数据](#输入输出数据)
  - [公版处理流程](#公版处理流程)
  - [优化处理流程](#优化处理流程)
  - [步骤参考](#步骤参考)
    - [环境、项目准备](#环境项目准备)
    - [导出为onnx](#导出为onnx)
    - [PTQ方案量化转化](#ptq方案量化转化)
    - [移除Bounding Box信息3个输出头的反量化节点](#移除bounding-box信息3个输出头的反量化节点)
    - [部分编译日志参考](#部分编译日志参考)
  - [模型训练](#模型训练)
  - [性能数据](#性能数据)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module-2)
    - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module)
  - [反馈](#反馈)
  - [参考](#参考)


## YOLO介绍

![](imgs/demo_rdkx5_yolov10n_detect.jpg)

YOLO(You Only Look Once)是一种流行的物体检测和图像分割模型，由华盛顿大学的约瑟夫-雷德蒙（Joseph Redmon）和阿里-法哈迪（Ali Farhadi）开发。YOLO 于 2015 年推出，因其高速度和高精确度而迅速受到欢迎。

 - 2016 年发布的YOLOv2 通过纳入批量归一化、锚框和维度集群改进了原始模型。
2018 年推出的YOLOv3 使用更高效的骨干网络、多锚和空间金字塔池进一步增强了模型的性能。
 - YOLOv4于 2020 年发布，引入了 Mosaic 数据增强、新的无锚检测头和新的损失函数等创新技术。
 - YOLOv5进一步提高了模型的性能，并增加了超参数优化、集成实验跟踪和自动导出为常用导出格式等新功能。
 - YOLOv6于 2022 年由美团开源，目前已用于该公司的许多自主配送机器人。
 - YOLOv7增加了额外的任务，如 COCO 关键点数据集的姿势估计。
 - YOLOv8是YOLO 的最新版本，由Ultralytics 提供。YOLOv8 YOLOv8 支持全方位的视觉 AI 任务，包括检测、分割、姿态估计、跟踪和分类。这种多功能性使用户能够在各种应用和领域中利用YOLOv8 的功能。
 - YOLOv9 引入了可编程梯度信息(PGI) 和广义高效层聚合网络(GELAN)等创新方法。
 - YOLOv10是由清华大学的研究人员使用该软件包创建的。 UltralyticsPython 软件包创建的。该版本通过引入端到端头(End-to-End head),消除了非最大抑制(NMS)要求，实现了实时目标检测的进步。
  
## 性能数据 (简要)
### RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | FLOPs (G) | BPU吞吐量 | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv10n | 640×640 | 80 | 6.7 G | 132.7 FPS | 4.5 ms | 
| YOLOv10s | 640×640 | 80 | 21.6 G | 71.0 FPS | 4.5 ms |  
| YOLOv10m | 640×640 | 80 | 59.1 G | 34.5 FPS | 4.5 ms |  
| YOLOv10b | 640×640 | 80 | 92.0 G | 25.4 FPS | 4.5 ms |  
| YOLOv10l | 640×640 | 80 | 120.3 G | 20.0 FPS | 4.5 ms |  
| YOLOv10x | 640×640 | 80 | 160.4 G | 14.5 FPS | 4.5 ms |  

### RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | FLOPs (G) | BPU吞吐量 | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv10n | 640×640 | 80 | 6.7 G | 18.1 FPS | 5 ms | 

## 模型下载地址
请参考`./model/download.md`

## 输入输出数据
- Input: 1x3x640x640, dtype=UINT8
- Output 0: [1, 80, 80, 64], dtype=INT32
- Output 1: [1, 40, 40, 64], dtype=INT32
- Output 2: [1, 20, 20, 64], dtype=INT32
- Output 3: [1, 80, 80, 80], dtype=FLOAT32
- Output 4: [1, 40, 40, 80], dtype=FLOAT32
- Output 5: [1, 20, 20, 80], dtype=FLOAT32


## 公版处理流程
![](imgs/YOLOv10_Detect_Origin.png)

## 优化处理流程
![](imgs/YOLOv10_Detect_Quantize.png)


以下请参考YOLOv8 Detect部分文档
- Classify部分，Dequantize操作。
- Classify部分，ReduceMax操作。
- Classify部分，Threshold（TopK）操作。
- Classify部分，GatherElements操作和ArgMax操作。
- Bounding Box部分，GatherElements操作和Dequantize操作。
- Bounding Box部分，DFL：SoftMax+Conv操作。
- Bounding Box部分，Decode：dist2bbox(ltrb2xyxy)操作。
- nms操作: YOLOv10无需nms。


## 步骤参考

注：任何No such file or directory, No module named "xxx", command not found.等报错请仔细检查，请勿逐条复制运行，如果对修改过程不理解请前往开发者社区从YOLOv5开始了解。
### 环境、项目准备
 - 下载ultralytics/ultralytics仓库，并参考ultralytics官方文档，配置好环境
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
 - 进入本地仓库，下载官方的预训练权重，这里以320万参数的YOLOv8n-Detect模型为例
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt
```

### 导出为onnx
 - 卸载yolo相关的命令行命令，这样直接修改`./ultralytics/ultralytics`目录即可生效。
```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # 或者
# 如果存在，则卸载
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # 或者
```

如果不是很顺利，可以通过以下Python命令确认需要修改的`ultralytics`目录的位置:
```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# 或者
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```

 - 修改Detect的输出头，直接将三个特征层的Bounding Box信息和Classify信息分开输出，一共6个输出头。

文件目录：./ultralytics/ultralytics/nn/modules/head.py，约第51行，`v10Detect`类的forward方法替换成以下内容.
注：建议您保留好原本的`forward`方法，例如改一个其他的名字`forward_`, 方便在训练的时候换回来。
```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
    # bboxes = [self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    # clses = [self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    # return (bboxes, clses)
```

 - 运行以下Python脚本，如果有**No module named onnxsim**报错，安装一个即可
```python
from ultralytics import YOLO
YOLO('yolov10n.pt').export(imgsz=640, format='onnx', simplify=True, opset=11)
```

### PTQ方案量化转化
 - 参考天工开物工具链手册和OE包，对模型进行检查，所有算子均在BPU上，进行编译即可。对应的yaml文件在`./ptq_yamls`目录下。
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolov10n.onnx
```
 - 根据模型检查结果，找到手动量化算子Softmax, 应有这样的内容, Softmax算子将模型拆为了两个BPU子图。这里的Softmax算子名称为"/model.10/attn/Softmax".
```bash
/model.10/attn/MatMul                               BPU  id(0)     HzSQuantizedMatmul         int8/int8        
/model.10/attn/Mul                                  BPU  id(0)     HzSQuantizedConv           int8/int32       
/model.10/attn/Softmax                              CPU  --        Softmax                    float/float      
/model.10/attn/Transpose_1                          BPU  id(1)     Transpose                  int8/int8        
/model.10/attn/MatMul_1                             BPU  id(1)     HzSQuantizedMatmul         int8/int8 
```
在对应的yaml文件中修改以下内容:
```yaml
model_parameters:
  node_info: {"/model.10/attn/Softmax": {'ON': 'BPU','InputType': 'int16','OutputType': 'int16'}}
```
 
 - 模型编译:
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolov10n.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov10_detect_bayese_640x640_nv12.yaml
```

### 移除Bounding Box信息3个输出头的反量化节点
 - 查看Bounding Box信息的3个输出头的反量化节点名称
通过hb_mapper makerbin时的日志，看到大小为[1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64]的三个输出的名称为output0, 479, 487。
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
    479:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    487:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    501:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    515:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    529:                  shape=[1, 20, 20, 80], dtype=FLOAT32
```

 - 进入编译产物的目录
```bash
$ cd yolov10n_detect_bayese_640x640_nv12
```
 - 查看可以被移除的反量化节点
```bash
$ hb_model_modifier yolov10n_detect_bayese_640x640_nv12.bin
```
 - 在生成的hb_model_modifier.log文件中，找到以下信息。主要是找到大小为[1, 64, 80, 80], [1, 64, 40, 40], [1, 64, 20, 20]的三个输出头的名称。当然，也可以通过netron等工具查看onnx模型，获得输出头的名称。
 此处的名称为：
 > "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize"
 > "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize"
 > "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize"

```bash
2024-08-16 18:30:20,014 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_quantized"
input: "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_x_scale"
output: "output0"
name: "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 18:30:20,014 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_quantized"
input: "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_x_scale"
output: "479"
name: "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-08-16 18:30:20,014 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_quantized"
input: "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_x_scale"
output: "487"
name: "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"
```
 - 使用以下命令移除上述三个反量化节点，注意，导出时这些名称可能不同，请仔细确认。
```bash
$ hb_model_modifier yolov10n_detect_bayese_640x640_nchw.bin \
-r "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize" \
-r "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize" \
-r "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize"
```
 - 移除成功会显示以下日志
```bash
2024-08-16 18:36:57,561 INFO log will be stored in /open_explorer/yolov10n_detect_bayese_640x640_nchw/hb_model_modifier.log
2024-08-16 18:36:57,566 INFO Nodes that will be removed from this model: ['/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize', '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize', '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize']
2024-08-16 18:36:57,566 INFO Node '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 18:36:57,566 INFO scale: /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 18:36:57,567 INFO Node '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize' is removed
2024-08-16 18:36:57,567 INFO Node '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 18:36:57,567 INFO scale: /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 18:36:57,567 INFO Node '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize' is removed
2024-08-16 18:36:57,567 INFO Node '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-08-16 18:36:57,568 INFO scale: /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-08-16 18:36:57,568 INFO Node '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize' is removed
2024-08-16 18:36:57,571 INFO modified model saved as yolov10n_detect_bayese_640x640_nchw_modified.bin
```

 - 接下来得到的bin模型名称为yolov8n_instance_seg_bayese_640x640_nchw_modified.bin, 这个是最终的模型。
 - NCHW输入的模型可以使用OpenCV和numpy来准备输入数据。
 - nv12输入的模型可以使用codec, jpu, vpu, gpu等硬件设备来准备输入数据，或者直接给TROS对应的功能包使用。


### 部分编译日志参考

可以观察到, SoftMax算子已经被BPU支持, 余弦相似度保持在0.9以上, 整个bin模型只有一个BPU子图。
```bash
2024-08-16 17:34:04,753 file: build.py func: build line No: 36 Start to Horizon NN Model Convert.
2024-08-16 17:34:04,753 file: model_debug.py func: model_debug line No: 61 Loading horizon_nn debug methods:[]
2024-08-16 17:34:04,753 file: cali_dict_parser.py func: cali_dict_parser line No: 40 Parsing the calibration parameter
2024-08-16 17:34:04,754 file: node_attribute.py func: node_attribute line No: 36 There are 1 nodes designated to run on the bpu: ['/model.10/attn/Softmax'].
2024-08-16 17:34:04,754 file: build.py func: build line No: 146 The specified model compilation architecture: bayes-e.
2024-08-16 17:34:04,754 file: build.py func: build line No: 148 The specified model compilation optimization parameters: [].
2024-08-16 17:34:04,776 file: build.py func: build line No: 36 Start to prepare the onnx model.
2024-08-16 17:34:04,776 file: utils.py func: utils line No: 53 Input ONNX Model Information:
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 64], dtype=FLOAT32
    479:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    487:                  shape=[1, 20, 20, 64], dtype=FLOAT32
    501:                  shape=[1, 80, 80, 80], dtype=FLOAT32
    515:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    529:                  shape=[1, 20, 20, 80], dtype=FLOAT32
2024-08-16 17:34:04,953 file: build.py func: build line No: 39 End to prepare the onnx model.
2024-08-16 17:34:05,187 file: build.py func: build line No: 186 Saving model: yolov10n_detect_bayese_640x640_nchw_original_float_model.onnx.
2024-08-16 17:34:05,188 file: build.py func: build line No: 36 Start to optimize the model.
2024-08-16 17:34:05,446 file: build.py func: build line No: 39 End to optimize the model.
2024-08-16 17:34:05,458 file: build.py func: build line No: 186 Saving model: yolov10n_detect_bayese_640x640_nchw_optimized_float_model.onnx.
2024-08-16 17:34:05,458 file: build.py func: build line No: 36 Start to calibrate the model.
2024-08-16 17:34:05,767 file: calibration_data_set.py func: calibration_data_set line No: 82 input name: images,  number_of_samples: 50
2024-08-16 17:34:05,767 file: calibration_data_set.py func: calibration_data_set line No: 93 There are 50 samples in the calibration data set.
2024-08-16 17:34:05,770 file: default_calibrater.py func: default_calibrater line No: 122 Run calibration model with default calibration method.
2024-08-16 17:34:07,219 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 17:34:11,596 file: ort.py func: ort line No: 179 Reset batch_size=1 and execute forward again...
2024-08-16 17:37:57,981 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 17:37:58,957 file: ort.py func: ort line No: 179 Reset batch_size=1 and execute forward again...
2024-08-16 17:38:10,479 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-08-16 17:38:12,340 file: ort.py func: ort line No: 179 Reset batch_size=1 and execute forward again...
2024-08-16 17:38:48,924 file: default_calibrater.py func: default_calibrater line No: 211 Select kl:num_bins=1024 method.
2024-08-16 17:38:53,280 file: build.py func: build line No: 39 End to calibrate the model.
2024-08-16 17:38:53,317 file: build.py func: build line No: 186 Saving model: yolov10n_detect_bayese_640x640_nchw_calibrated_model.onnx.
2024-08-16 17:38:53,317 file: build.py func: build line No: 36 Start to quantize the model.
2024-08-16 17:38:54,516 file: build.py func: build line No: 39 End to quantize the model.
2024-08-16 17:38:54,597 file: build.py func: build line No: 186 Saving model: yolov10n_detect_bayese_640x640_nchw_quantized_model.onnx.
2024-08-16 17:38:54,824 file: build.py func: build line No: 36 Start to compile the model with march bayes-e.
2024-08-16 17:38:54,964 file: hybrid_build.py func: hybrid_build line No: 133 Compile submodel: main_graph_subgraph_0
2024-08-16 17:38:55,119 file: hbdk_cc.py func: hbdk_cc line No: 115 hbdk-cc parameters:['--O3', '--core-num', '1', '--fast', '--input-layout', 'NHWC', '--output-layout', 'NHWC', '--input-source', 'ddr']
2024-08-16 17:38:55,120 file: hbdk_cc.py func: hbdk_cc line No: 116 hbdk-cc command used:hbdk-cc -f hbir -m /tmp/tmptfp5gbft/main_graph_subgraph_0.hbir -o /tmp/tmptfp5gbft/main_graph_subgraph_0.hbm --march bayes-e --progressbar --O3 --core-num 1 --fast --input-layout NHWC --output-layout NHWC --input-source ddr
2024-08-16 17:38:55,120 file: tool_utils.py func: tool_utils line No: 317 Can not find the scale for node HZ_PREPROCESS_FOR_images_NCHW2NHWC_LayoutConvert_Input0
2024-08-16 17:42:36,250 file: tool_utils.py func: tool_utils line No: 322 consumed time 221.109
2024-08-16 17:42:36,344 file: tool_utils.py func: tool_utils line No: 322 FPS=129.09, latency = 7746.7 us, DDR = 19285696 bytes   (see main_graph_subgraph_0.html)
2024-08-16 17:42:36,412 file: build.py func: build line No: 39 End to compile the model with march bayes-e.
2024-08-16 17:42:36,442 file: print_node_info.py func: print_node_info line No: 57 The converted model node information:
================================================================================================================================
Node                                                ON   Subgraph  Type          Cosine Similarity  Threshold   In/Out DataType  
---------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess  0.999779           127.000000  int8/int8        
/model.0/conv/Conv                                  BPU  id(0)     Conv          0.998270           0.995605    int8/int8        
/model.0/act/Mul                                    BPU  id(0)     HzSwish       0.997878           10.603966   int8/int8        
/model.1/conv/Conv                                  BPU  id(0)     Conv          0.984483           8.673835    int8/int8        
/model.1/act/Mul                                    BPU  id(0)     HzSwish       0.979556           30.013716   int8/int8        
/model.2/cv1/conv/Conv                              BPU  id(0)     Conv          0.982836           18.365402   int8/int8        
/model.2/cv1/act/Mul                                BPU  id(0)     HzSwish       0.995462           13.615964   int8/int8        
/model.2/Split                                      BPU  id(0)     Split         0.998296           6.394294    int8/int8        
/model.2/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.979835           6.394294    int8/int8        
/model.2/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.986137           5.394912    int8/int8        
/model.2/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.983792           5.073148    int8/int8        
/model.2/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.985896           7.789621    int8/int8        
UNIT_CONV_FOR_/model.2/m.0/Add                      BPU  id(0)     Conv          0.998296           6.394294    int8/int8        
/model.2/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.2/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.2/Concat                                     BPU  id(0)     Concat        0.994953           6.394294    int8/int8        
/model.2/cv2/conv/Conv                              BPU  id(0)     Conv          0.983636           8.365657    int8/int8        
/model.2/cv2/act/Mul                                BPU  id(0)     HzSwish       0.990088           8.100873    int8/int8        
/model.3/conv/Conv                                  BPU  id(0)     Conv          0.988301           5.377353    int8/int8        
/model.3/act/Mul                                    BPU  id(0)     HzSwish       0.992076           5.424640    int8/int8        
/model.4/cv1/conv/Conv                              BPU  id(0)     Conv          0.989007           5.688507    int8/int8        
/model.4/cv1/act/Mul                                BPU  id(0)     HzSwish       0.990244           4.164920    int8/int8        
/model.4/Split                                      BPU  id(0)     Split         0.995551           2.224742    int8/int8        
/model.4/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.985632           2.224742    int8/int8        
/model.4/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.986047           4.467528    int8/int8        
/model.4/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.991723           3.448012    int8/int8        
/model.4/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.993127           4.670990    int8/int8        
UNIT_CONV_FOR_/model.4/m.0/Add                      BPU  id(0)     Conv          0.995551           2.224742    int8/int8        
/model.4/m.1/cv1/conv/Conv                          BPU  id(0)     Conv          0.992123           3.380925    int8/int8        
/model.4/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish       0.993438           3.713458    int8/int8        
/model.4/m.1/cv2/conv/Conv                          BPU  id(0)     Conv          0.989381           2.662815    int8/int8        
/model.4/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish       0.991482           6.437428    int8/int8        
UNIT_CONV_FOR_/model.4/m.1/Add                      BPU  id(0)     Conv          0.995955           3.380925    int8/int8        
/model.4/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
/model.4/Concat                                     BPU  id(0)     Concat        0.994967           2.224742    int8/int8        
/model.4/cv2/conv/Conv                              BPU  id(0)     Conv          0.990687           5.684978    int8/int8        
/model.4/cv2/act/Mul                                BPU  id(0)     HzSwish       0.991007           4.640128    int8/int8        
/model.5/cv1/conv/Conv                              BPU  id(0)     Conv          0.990364           2.622323    int8/int8        
/model.5/cv1/act/Mul                                BPU  id(0)     HzSwish       0.971344           7.331495    int8/int8        
/model.5/cv2/conv/Conv                              BPU  id(0)     Conv          0.966769           2.117791    int8/int8        
/model.6/cv1/conv/Conv                              BPU  id(0)     Conv          0.975072           8.549481    int8/int8        
/model.6/cv1/act/Mul                                BPU  id(0)     HzSwish       0.975069           7.051415    int8/int8        
/model.6/Split                                      BPU  id(0)     Split         0.983371           3.504654    int8/int8        
/model.6/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.979234           3.504654    int8/int8        
/model.6/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.938951           7.225185    int8/int8        
/model.6/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.956522           1.429943    int8/int8        
/model.6/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.956711           5.600646    int8/int8        
UNIT_CONV_FOR_/model.6/m.0/Add                      BPU  id(0)     Conv          0.983371           3.504654    int8/int8        
/model.6/m.1/cv1/conv/Conv                          BPU  id(0)     Conv          0.981077           4.064888    int8/int8        
/model.6/m.1/cv1/act/Mul                            BPU  id(0)     HzSwish       0.965567           6.363821    int8/int8        
/model.6/m.1/cv2/conv/Conv                          BPU  id(0)     Conv          0.969873           2.145753    int8/int8        
/model.6/m.1/cv2/act/Mul                            BPU  id(0)     HzSwish       0.968186           8.400377    int8/int8        
UNIT_CONV_FOR_/model.6/m.1/Add                      BPU  id(0)     Conv          0.975839           4.064888    int8/int8        
/model.6/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/m.0/Add_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
/model.6/Concat                                     BPU  id(0)     Concat        0.977647           3.504654    int8/int8        
/model.6/cv2/conv/Conv                              BPU  id(0)     Conv          0.969809           6.438043    int8/int8        
/model.6/cv2/act/Mul                                BPU  id(0)     HzSwish       0.972988           8.041522    int8/int8        
/model.7/cv1/conv/Conv                              BPU  id(0)     Conv          0.984554           3.157327    int8/int8        
/model.7/cv1/act/Mul                                BPU  id(0)     HzSwish       0.967383           7.064945    int8/int8        
/model.7/cv2/conv/Conv                              BPU  id(0)     Conv          0.955257           3.176519    int8/int8        
/model.8/cv1/conv/Conv                              BPU  id(0)     Conv          0.964391           6.597043    int8/int8        
/model.8/cv1/act/Mul                                BPU  id(0)     HzSwish       0.958399           7.813031    int8/int8        
/model.8/Split                                      BPU  id(0)     Split         0.955191           3.325485    int8/int8        
/model.8/m.0/cv1/conv/Conv                          BPU  id(0)     Conv          0.975019           3.325485    int8/int8        
/model.8/m.0/cv1/act/Mul                            BPU  id(0)     HzSwish       0.954413           8.182122    int8/int8        
/model.8/m.0/cv2/conv/Conv                          BPU  id(0)     Conv          0.956983           2.017900    int8/int8        
/model.8/m.0/cv2/act/Mul                            BPU  id(0)     HzSwish       0.956910           10.418705   int8/int8        
UNIT_CONV_FOR_/model.8/m.0/Add                      BPU  id(0)     Conv          0.968653           3.325485    int8/int8        
/model.8/Split_output_0_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.8/Split_output_1_calibrated_Requantize       BPU  id(0)     HzRequantize                                 int8/int8        
/model.8/Concat                                     BPU  id(0)     Concat        0.954300           3.325485    int8/int8        
/model.8/cv2/conv/Conv                              BPU  id(0)     Conv          0.958738           3.372330    int8/int8        
/model.8/cv2/act/Mul                                BPU  id(0)     HzSwish       0.950081           8.054053    int8/int8        
/model.9/cv1/conv/Conv                              BPU  id(0)     Conv          0.977853           6.169669    int8/int8        
/model.9/cv1/act/Mul                                BPU  id(0)     HzSwish       0.981891           6.056477    int8/int8        
/model.9/m/MaxPool                                  BPU  id(0)     MaxPool       0.992955           6.763361    int8/int8        
/model.9/m_1/MaxPool                                BPU  id(0)     MaxPool       0.995619           6.763361    int8/int8        
/model.9/m_2/MaxPool                                BPU  id(0)     MaxPool       0.996569           6.763361    int8/int8        
/model.9/Concat                                     BPU  id(0)     Concat        0.994307           6.763361    int8/int8        
/model.9/cv2/conv/Conv                              BPU  id(0)     Conv          0.959600           6.763361    int8/int8        
/model.9/cv2/act/Mul                                BPU  id(0)     HzSwish       0.927468           8.180250    int8/int8        
/model.10/cv1/conv/Conv                             BPU  id(0)     Conv          0.926328           2.017437    int8/int8        
/model.10/cv1/act/Mul                               BPU  id(0)     HzSwish       0.925031           9.352364    int8/int8        
/model.10/Split                                     BPU  id(0)     Split         0.885757           5.323861    int8/int8        
/model.10/attn/qkv/conv/Conv                        BPU  id(0)     Conv          0.943905           5.323861    int8/int8        
/model.10/attn/Reshape                              BPU  id(0)     Reshape       0.943905           7.924023    int8/int8        
/model.10/attn/Split                                BPU  id(0)     Split         0.964183           7.924023    int8/int8        
/model.10/attn/Transpose                            BPU  id(0)     Transpose     0.964182           --          int8/int8        
/model.10/attn/Reshape_2                            BPU  id(0)     Reshape       0.929960           --          int8/int8        
/model.10/attn/MatMul                               BPU  id(0)     MatMul        0.958139           7.924023    int8/int8        
/model.10/attn/Mul                                  BPU  id(0)     Conv          0.958141           89.904221   int8/int16       
...0/attn/Softmax_reducemax_FROM_QUANTIZED_SOFTMAX  BPU  id(0)     ReduceMax     0.995906           15.892972   int16/int16      
/model.10/attn/Softmax_sub_FROM_QUANTIZED_SOFTMAX   BPU  id(0)     Sub           0.988598           15.892972   int16/int16      
/model.10/attn/Softmax_exp_FROM_QUANTIZED_SOFTMAX   BPU  id(0)     Exp           0.934911           11.090324   int16/int16      
...0/attn/Softmax_reducesum_FROM_QUANTIZED_SOFTMAX  BPU  id(0)     ReduceSum     0.970950           1.000000    int16/int16      
.../attn/Softmax_reciprocal_FROM_QUANTIZED_SOFTMAX  BPU  id(0)     Reciprocal    0.954017           154.920547  int16/int16      
/model.10/attn/Softmax_mul_FROM_QUANTIZED_SOFTMAX   BPU  id(0)     Mul           0.923754           1.000000    int16/int8       
/model.10/attn/Transpose_1                          BPU  id(0)     Transpose     0.923754           0.136998    int8/int8        
/model.10/attn/MatMul_1                             BPU  id(0)     MatMul        0.922402           0.136998    int8/int8        
/model.10/attn/Reshape_1                            BPU  id(0)     Reshape       0.922402           5.962341    int8/int8        
/model.10/attn/pe/conv/Conv                         BPU  id(0)     Conv          0.933046           7.924023    int8/int8        
/model.10/attn/proj/conv/Conv                       BPU  id(0)     Conv          0.883142           5.235514    int8/int8        
/model.10/ffn/ffn.0/conv/Conv                       BPU  id(0)     Conv          0.906512           8.917325    int8/int8        
/model.10/ffn/ffn.0/act/Mul                         BPU  id(0)     HzSwish       0.843966           8.798843    int8/int8        
/model.10/ffn/ffn.1/conv/Conv                       BPU  id(0)     Conv          0.859181           4.622127    int8/int8        
/model.10/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.10/Concat                                    BPU  id(0)     Concat        0.899151           5.323861    int8/int8        
/model.10/cv2/conv/Conv                             BPU  id(0)     Conv          0.917233           3.629226    int8/int8        
/model.10/cv2/act/Mul                               BPU  id(0)     HzSwish       0.864065           8.974898    int8/int8        
/model.11/Resize                                    BPU  id(0)     Resize        0.864060           2.635198    int8/int8        
/model.11/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8        
/model.12/Concat                                    BPU  id(0)     Concat        0.911233           2.635198    int8/int8        
/model.13/cv1/conv/Conv                             BPU  id(0)     Conv          0.957063           3.157327    int8/int8        
/model.13/cv1/act/Mul                               BPU  id(0)     HzSwish       0.946216           7.360564    int8/int8        
/model.13/Split                                     BPU  id(0)     Split         0.968129           1.736766    int8/int8        
/model.13/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.970430           1.736766    int8/int8        
/model.13/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.940039           6.535757    int8/int8        
/model.13/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.949822           3.729291    int8/int8        
/model.13/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.939619           6.101533    int8/int8        
/model.13/Concat                                    BPU  id(0)     Concat        0.944175           1.736766    int8/int8        
/model.13/cv2/conv/Conv                             BPU  id(0)     Conv          0.939836           1.736766    int8/int8        
/model.13/cv2/act/Mul                               BPU  id(0)     HzSwish       0.930296           7.232765    int8/int8        
/model.14/Resize                                    BPU  id(0)     Resize        0.930296           3.190812    int8/int8        
...el.4/cv2/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8        
/model.15/Concat                                    BPU  id(0)     Concat        0.967072           3.190812    int8/int8        
/model.16/cv1/conv/Conv                             BPU  id(0)     Conv          0.985251           3.190812    int8/int8        
/model.16/cv1/act/Mul                               BPU  id(0)     HzSwish       0.988765           6.394931    int8/int8        
/model.16/Split                                     BPU  id(0)     Split         0.993388           2.141448    int8/int8        
/model.16/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.983419           2.141448    int8/int8        
/model.16/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.985318           5.820910    int8/int8        
/model.16/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.982261           2.331356    int8/int8        
/model.16/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.987834           5.858522    int8/int8        
/model.16/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.16/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.16/Concat                                    BPU  id(0)     Concat        0.988411           2.141448    int8/int8        
/model.16/cv2/conv/Conv                             BPU  id(0)     Conv          0.983385           2.395487    int8/int8        
/model.16/cv2/act/Mul                               BPU  id(0)     HzSwish       0.988149           6.229319    int8/int8        
/model.17/conv/Conv                                 BPU  id(0)     Conv          0.952232           2.382120    int8/int8        
/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv   BPU  id(0)     Conv          0.970563           2.382120    int8/int8        
...3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv  BPU  id(0)     Conv          0.998945           2.382120    int8/int8        
/model.17/act/Mul                                   BPU  id(0)     HzSwish       0.937657           6.560148    int8/int8        
/model.23/one2one_cv2.0/one2one_cv2.0.0/act/Mul     BPU  id(0)     HzSwish       0.967019           11.006473   int8/int8        
...cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/act/Mul  BPU  id(0)     HzSwish       0.998681           3.569149    int8/int8        
/model.18/Concat                                    BPU  id(0)     Concat        0.932330           3.190812    int8/int8        
/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv   BPU  id(0)     Conv          0.945215           4.088514    int8/int8        
...3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/conv/Conv  BPU  id(0)     Conv          0.981223           3.510941    int8/int8        
/model.19/cv1/conv/Conv                             BPU  id(0)     Conv          0.927528           3.190812    int8/int8        
/model.23/one2one_cv2.0/one2one_cv2.0.1/act/Mul     BPU  id(0)     HzSwish       0.947358           27.335110   int8/int8        
...cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.1/act/Mul  BPU  id(0)     HzSwish       0.974823           5.289838    int8/int8        
/model.19/cv1/act/Mul                               BPU  id(0)     HzSwish       0.920491           7.167810    int8/int8        
/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv        BPU  id(0)     Conv          0.988992           11.636202   int8/int32       
...3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/conv/Conv  BPU  id(0)     Conv          0.982968           2.674858    int8/int8        
/model.19/Split                                     BPU  id(0)     Split         0.930839           2.987007    int8/int8        
/model.19/m.0/cv1/conv/Conv                         BPU  id(0)     Conv          0.962431           2.987007    int8/int8        
...cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.0/act/Mul  BPU  id(0)     HzSwish       0.981376           5.826365    int8/int8        
...3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/conv/Conv  BPU  id(0)     Conv          0.956472           2.970755    int8/int8        
/model.19/m.0/cv1/act/Mul                           BPU  id(0)     HzSwish       0.926298           7.250287    int8/int8        
/model.19/m.0/cv2/conv/Conv                         BPU  id(0)     Conv          0.915416           2.247598    int8/int8        
...cv3.0/one2one_cv3.0.1/one2one_cv3.0.1.1/act/Mul  BPU  id(0)     HzSwish       0.965275           27.485249   int8/int8        
/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv        BPU  id(0)     Conv          0.999432           44.006672   int8/int32       
/model.19/m.0/cv2/act/Mul                           BPU  id(0)     HzSwish       0.904141           9.066823    int8/int8        
/model.19/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.19/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.19/Concat                                    BPU  id(0)     Concat        0.913236           2.987007    int8/int8        
/model.19/cv2/conv/Conv                             BPU  id(0)     Conv          0.923298           4.169190    int8/int8        
/model.19/cv2/act/Mul                               BPU  id(0)     HzSwish       0.920483           9.177308    int8/int8        
/model.20/cv1/conv/Conv                             BPU  id(0)     Conv          0.911498           2.309886    int8/int8        
/model.23/one2one_cv2.1/one2one_cv2.1.0/conv/Conv   BPU  id(0)     Conv          0.941168           2.309886    int8/int8        
...3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv  BPU  id(0)     Conv          0.969385           2.309886    int8/int8        
/model.20/cv1/act/Mul                               BPU  id(0)     HzSwish       0.871221           7.872035    int8/int8        
/model.23/one2one_cv2.1/one2one_cv2.1.0/act/Mul     BPU  id(0)     HzSwish       0.937789           13.438343   int8/int8        
...cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/act/Mul  BPU  id(0)     HzSwish       0.969118           7.788738    int8/int8        
/model.20/cv2/conv/Conv                             BPU  id(0)     Conv          0.882901           4.240702    int8/int8        
/model.23/one2one_cv2.1/one2one_cv2.1.1/conv/Conv   BPU  id(0)     Conv          0.934215           4.444287    int8/int8        
...3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/conv/Conv  BPU  id(0)     Conv          0.958680           7.788737    int8/int8        
/model.21/Concat                                    BPU  id(0)     Concat        0.869392           2.635198    int8/int8        
/model.22/cv1/conv/Conv                             BPU  id(0)     Conv          0.872874           2.635198    int8/int8        
/model.23/one2one_cv2.1/one2one_cv2.1.1/act/Mul     BPU  id(0)     HzSwish       0.943696           24.260267   int8/int8        
...cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.1/act/Mul  BPU  id(0)     HzSwish       0.946590           13.584941   int8/int8        
/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv        BPU  id(0)     Conv          0.985716           41.404373   int8/int32       
...3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv  BPU  id(0)     Conv          0.903468           3.631684    int8/int8        
/model.22/cv1/act/Mul                               BPU  id(0)     HzSwish       0.854948           7.503721    int8/int8        
/model.22/Split                                     BPU  id(0)     Split         0.906026           2.796731    int8/int8        
...cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/act/Mul  BPU  id(0)     HzSwish       0.904703           16.311018   int8/int8        
/model.22/m.0/cv1/cv1.0/conv/Conv                   BPU  id(0)     Conv          0.920342           2.796731    int8/int8        
...3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv  BPU  id(0)     Conv          0.939913           6.574324    int8/int8        
/model.22/m.0/cv1/cv1.0/act/Mul                     BPU  id(0)     HzSwish       0.911485           6.698623    int8/int8        
...cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Mul  BPU  id(0)     HzSwish       0.950459           50.715145   int8/int8        
/model.22/m.0/cv1/cv1.1/conv/Conv                   BPU  id(0)     Conv          0.883911           5.590483    int8/int8        
/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv        BPU  id(0)     Conv          0.998491           45.058029   int8/int32       
/model.22/m.0/cv1/cv1.1/act/Mul                     BPU  id(0)     HzSwish       0.873500           5.918747    int8/int8        
/model.22/m.0/cv1/cv1.2/conv/Conv                   BPU  id(0)     Conv          0.906973           3.918176    int8/int8        
/model.22/m.0/cv1/cv1.2/act/Mul                     BPU  id(0)     HzSwish       0.900139           11.114021   int8/int8        
/model.22/m.0/cv1/cv1.3/conv/Conv                   BPU  id(0)     Conv          0.878716           4.137039    int8/int8        
/model.22/m.0/cv1/cv1.3/act/Mul                     BPU  id(0)     HzSwish       0.872436           8.790598    int8/int8        
/model.22/m.0/cv1/cv1.4/conv/Conv                   BPU  id(0)     Conv          0.884580           5.088044    int8/int8        
/model.22/m.0/cv1/cv1.4/act/Mul                     BPU  id(0)     HzSwish       0.879342           9.508633    int8/int8        
UNIT_CONV_FOR_/model.22/m.0/Add                     BPU  id(0)     Conv          0.906026           2.796731    int8/int8        
/model.22/Split_output_0_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.22/Split_output_1_calibrated_Requantize      BPU  id(0)     HzRequantize                                 int8/int8        
/model.22/Concat                                    BPU  id(0)     Concat        0.866978           2.796731    int8/int8        
/model.22/cv2/conv/Conv                             BPU  id(0)     Conv          0.853498           13.280812   int8/int8        
/model.22/cv2/act/Mul                               BPU  id(0)     HzSwish       0.836353           8.961614    int8/int8        
/model.23/one2one_cv2.2/one2one_cv2.2.0/conv/Conv   BPU  id(0)     Conv          0.865065           4.120993    int8/int8        
...3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/conv/Conv  BPU  id(0)     Conv          0.857857           4.120993    int8/int8        
/model.23/one2one_cv2.2/one2one_cv2.2.0/act/Mul     BPU  id(0)     HzSwish       0.879805           9.944433    int8/int8        
...cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.0/act/Mul  BPU  id(0)     HzSwish       0.841405           10.132360   int8/int8        
/model.23/one2one_cv2.2/one2one_cv2.2.1/conv/Conv   BPU  id(0)     Conv          0.870142           6.913608    int8/int8        
...3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/conv/Conv  BPU  id(0)     Conv          0.879809           12.896649   int8/int8        
/model.23/one2one_cv2.2/one2one_cv2.2.1/act/Mul     BPU  id(0)     HzSwish       0.882873           31.428568   int8/int8        
...cv3.2/one2one_cv3.2.0/one2one_cv3.2.0.1/act/Mul  BPU  id(0)     HzSwish       0.865068           9.924242    int8/int8        
/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv        BPU  id(0)     Conv          0.969894           13.934072   int8/int32       
...3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/conv/Conv  BPU  id(0)     Conv          0.636159           4.638824    int8/int8        
...cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.0/act/Mul  BPU  id(0)     HzSwish       0.779163           22.766069   int8/int8        
...3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/conv/Conv  BPU  id(0)     Conv          0.866428           5.616152    int8/int8        
...cv3.2/one2one_cv3.2.1/one2one_cv3.2.1.1/act/Mul  BPU  id(0)     HzSwish       0.907194           35.950649   int8/int8        
/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv        BPU  id(0)     Conv          0.997476           17.939869   int8/int32

```


## 模型训练

 - 模型训练请参考ultralytics官方文档，这个文档由ultralytics维护，质量非常的高。网络上也有非常多的参考材料，得到一个像官方一样的预训练权重的模型并不困难。
 - 请注意，训练时无需修改任何程序，无需修改forward方法。

## 性能数据

### RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型 | 尺寸(像素) | 类别数 | FLOPs (G) | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间<br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv10n | 640×640 | 80 | 6.7  | 38.5 G |  | 9.3 ms / 107.0 FPS (1 thread) <br/> 15.0 ms / 132.7 FPS (2 threads) | 4.5 ms |
| YOLOv10s | 640×640 | 80 | 21.6 | 46.3 G |  | 15.8 ms / 63.0 FPS (1 thread) <br/> 28.1 ms / 71.0 FPS (2 threads) | 4.5 ms |
| YOLOv10m | 640×640 | 80 | 59.1 | 51.1 G |  | 30.8 ms / 32.4 FPS (1 thread) <br/> 51.8 ms / 34.5 FPS (2 threads) | 4.5 ms |
| YOLOv10b | 640×640 | 80 | 92.0 | 52.3 G |  | 41.1 ms / 24.3 FPS (1 thread) <br/> 78.4 ms / 25.4 FPS (2 threads) | 4.5 ms |
| YOLOv10l | 640×640 | 80 | 120.3 | 53.2 G |  | 52.0 ms / 19.2 FPS (1 thread) <br/> 100.0 ms / 20.0 FPS (2 threads) | 4.5 ms |
| YOLOv10x | 640×640 | 80 | 160.4 | 54.4 G |  | 70.7 ms / 14.1 FPS (1 thread) <br/> 137.3 ms / 14.5 FPS (2 threads) | 4.5 ms |

### RDK X3 & RDK X3 Module
目标检测 Detection (COCO)
| 模型 | 尺寸(像素) | 类别数 | FLOPs (G) | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间<br/>(Python) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv10n | 640×640 | 80 | 6.7  | 38.5 G |  | 174.7 ms / 5.7 FPS (1 thread) <br/> 181.5 ms / 11.0 FPS (2 threads) <br/> 240.1 ms / 16.2 FPS (4 threads) <br/> 421.0 ms / 18.1 FPS (8 threads) | 5 ms |

说明: 
1. BPU延迟与BPU吞吐量。
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况。
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中4个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡。X5的BPU整体比较厉害, 一般2个线程就可以将BPU吃满, 帧延迟和吞吐量都非常出色。
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据。
 - BPU延迟和BPU吞吐量使用以下命令在板端测试
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin
```
2. 测试板卡均为最佳状态。
 - X5的状态为最佳状态：CPU为8 × A55@1.8G, 全核心Performance调度, BPU为1 × Bayes-e@10TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
 - X3的状态为最佳状态：CPU为4 × A53@1.8G, 全核心Performance调度, BPU为2 × Bernoulli2@5TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```
3. 浮点/定点mAP：50-95精度使用pycocotools计算,来自于COCO数据集,可以参考微软的论文,此处用于评估板端部署的精度下降程度。
4. 关于后处理: 目前在X5上使用Python重构的后处理, 仅需要单核心单线程串行5ms左右即可完成, 也就是说只需要占用2个CPU核心(200%的CPU占用, 最大800%的CPU占用), 每分钟可完成400帧图像的后处理, 后处理不会构成瓶颈.

## 反馈
本文如果有表达不清楚的地方欢迎前往地瓜开发者社区进行提问和交流.

[地瓜机器人开发者社区](developer.d-robotics.cc).

## 参考

[ultralytics](https://docs.ultralytics.com/)