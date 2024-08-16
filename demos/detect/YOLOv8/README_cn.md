[English](./README.md) | 简体中文

# YOLOv8 Detect
## 公版处理流程
![](imgs/YOLOv8_Detect_Origin.png)

## 优化处理流程
![](imgs/YOLOv8_Detect_Quantize.png)

 - Classify部分，Dequantize操作
在模型编译时，如果选择了移除所有的反量化算子，这里需要在后处理中手动对Classify部分的三个输出头进行反量化。查看反量化系数的方式有多种，可以查看hb_mapper时产物的日志，也可通过BPU推理接口的API来获取。
注意，这里每一个C维度的反量化系数都是不同的，每个头都有80个反量化系数，可以使用numpy的广播直接乘。
此处反量化在bin模型中实现，所以拿到的输出是float32的。

 - Classify部分，ReduceMax操作
ReduceMax操作是沿着Tensor的某一个维度找到最大值，此操作用于找到8400个Grid Cell的80个分数的最大值。操作对象是每个Grid Cell的80类别的值，在C维度操作。注意，这步操作给出的是最大值，并不是80个值中最大值的索引。
激活函数Sigmoid具有单调性，所以Sigmoid作用前的80个分数的大小关系和Sigmoid作用后的80个分数的大小关系不会改变。
$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
综上，bin模型直接输出的最大值(反量化完成)的位置就是最终分数最大值的位置，bin模型输出的最大值经过Sigmoid计算后就是原来onnx模型的最大值。

 - Classify部分，Threshold（TopK）操作
此操作用于找到8400个Grid Cell中，符合要求的Grid Cell。操作对象为8400个Grid Cell，在H和W的维度操作。如果您有阅读我的程序，你会发现我将后面H和W维度拉平了，这样只是为了程序设计和书面表达的方便，它们并没有本质上的不同。
我们假设某一个Grid Cell的某一个类别的分数记为$x$，激活函数作用完的整型数据为$y$，阈值筛选的过程会给定一个阈值，记为$C$，那么此分数合格的充分必要条件为：
$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$
由此可以得出此分数合格的充分必要条件为：
$$x > -ln\left(\frac{1}{C}-1\right)$$
此操作会符合条件的Grid Cell的索引（indices）和对应Grid Cell的最大值，这个最大值经过Sigmoid计算后就是这个Grid Cell对应类别的分数了。

 - Classify部分，GatherElements操作和ArgMax操作
使用Threshold（TopK）操作得到的符合条件的Grid Cell的索引（indices），在GatherElements操作中获得符合条件的Grid Cell，使用ArgMax操作得到具体是80个类别中哪一个最大，得到这个符合条件的Grid Cell的类别。

 - Bounding Box部分，GatherElements操作和Dequantize操作
使用Threshold（TopK）操作得到的符合条件的Grid Cell的索引（indices），在GatherElements操作中获得符合条件的Grid Cell，这里每一个C维度的反量化系数都是不同的，每个头都有64个反量化系数，可以使用numpy的广播直接乘，得到1×64×k×1的bbox信息。

 - Bounding Box部分，DFL：SoftMax+Conv操作
每一个Grid Cell会有4个数字来确定这个框框的位置，DFL结构会对每个框的某条边基于anchor的位置给出16个估计，对16个估计求SoftMax，然后通过一个卷积操作来求期望，这也是Anchor Free的核心设计，即每个Grid Cell仅仅负责预测1个Bounding box。假设在对某一条边偏移量的预测中，这16个数字为 $ l_p $ 或者$(t_p, t_p, b_p)$，其中$p = 0,1,...,15$那么偏移量的计算公式为：
$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

 - Bounding Box部分，Decode：dist2bbox(ltrb2xyxy)操作
此操作将每个Bounding Box的ltrb描述解码为xyxy描述，ltrb分别表示左上右下四条边距离相对于Grid Cell中心的距离，相对位置还原成绝对位置后，再乘以对应特征层的采样倍数，即可还原成xyxy坐标，xyxy表示Bounding Box的左上角和右下角两个点坐标的预测值。
![](imgs/ltrb2xyxy.jpg)

图片输入为$Size=640$，对于Bounding box预测分支的第$i$个特征图$(i=1, 2, 3)$，对应的下采样倍数记为$Stride(i)$，在YOLOv8 - Detect中，$Stride(1)=8, Stride(2)=16, Stride(3)=32$，对应特征图的尺寸记为$n_i = {Size}/{Stride(i)}$，即尺寸为$n_1 = 80, n_2 = 40 ,n_3 = 20$三个特征图，一共有$n_1^2+n_2^2+n_3^3=8400$个Grid Cell，负责预测8400个Bounding Box。
对特征图i，第x行y列负责预测对应尺度Bounding Box的检测框，其中$x,y \in [0, n_i)\bigcap{Z}$，$Z$为整数的集合。DFL结构后的Bounding Box检测框描述为$ltrb$描述，而我们需要的是$xyxy$描述，具体的转化关系如下：
$$x_1 = (x+0.5-l)\times{Stride(i)}$$
$$y_1 = (y+0.5-t)\times{Stride(i)}$$
$$x_2 = (x+0.5+r)\times{Stride(i)}$$
$$y_1 = (y+0.5+b)\times{Stride(i)}$$

YOLOv8，v9，会有一个nms操作去去掉重复识别的目标，YOLOv10不需要。最终的检测结果了，包括类别(id)，分数(score)和位置(xyxy)。

## 步骤参考

注：任何No such file or directory, No module named "xxx", command not found.等报错请仔细检查，请勿逐条复制运行，如果对修改过程不理解请前往开发者社区从YOLOv5开始了解。
### 环境、项目准备
 - 下载ultralytics/ultralytics仓库，并参考YOLOv8官方文档，配置好环境
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
 - 进入本地仓库，下载官方的预训练权重，这里以320万参数的YOLOv8n-Detect模型为例
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
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
 - 修改Detect的输出头，直接将三个特征层的Bounding Box信息和Classify信息分开输出，一共6个输出头。
文件目录：./ultralytics/ultralytics/nn/modules/head.py，约第51行，vDetect类的forward方法替换成以下内容.
注：建议您保留好原本的`forward`方法，例如改一个其他的名字`forward_`, 方便在训练的时候换回来。
```python
def forward(self, x):
    bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    clses = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
    return (bboxes, clses)
```

 - 运行以下Python脚本，如果有**No module named onnxsim**报错，安装一个即可
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(imgsz=640, format='onnx', simplify=True, opset=11)
```

### PTQ方案量化转化
 - 参考天工开物工具链手册和OE包，对模型进行检查，所有算子均在BPU上，进行编译即可。对应的yaml文件在`./ptq_yamls`目录下。
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolov8n.onnx
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolov8_detect_nchw.yaml
```

### 移除bbox信息3个输出头的反量化节点
 - 查看bbox信息的3个输出头的反量化节点名称
通过hb_mapper makerbin时的日志，看到大小为[1, 64, 80, 80], [1, 64, 40, 40], [1, 64, 20, 20]的三个输出的名称为output0, 326, 334。
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

 - 进入编译产物的目录
```bash
$ cd yolov8n_bayese_640x640_nchw
```
 - 查看可以被移除的反量化节点
```bash
$ hb_model_modifier yolov8n_bayese_640x640_nchw.bin
```
 - 在生成的hb_model_modifier.log文件中，找到以下信息。主要是找到大小为[1, 64, 80, 80], [1, 64, 40, 40], [1, 64, 20, 20]的三个输出头的名称。当然，也可以通过netron等工具查看onnx模型，获得输出头的名称。
 此处的名称为:
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
 - 使用以下命令移除上述三个反量化节点，注意，导出时这些名称可能不同，请仔细确认。
```bash
$ hb_model_modifier yolov8n_bayese_640x640_nchw.bin \
-r /model.22/cv2.0/cv2.0.2/Conv_output_0_HzDequantize \
-r /model.22/cv2.1/cv2.1.2/Conv_output_0_HzDequantize \
-r /model.22/cv2.2/cv2.2.2/Conv_output_0_HzDequantize
```
 - 移除成功会显示以下日志
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

 - 接下来得到的bin模型名称为yolov8n_bayese_640x640_nchw_modified.bin, 这个是最终的模型。
 - NCHW输入的模型可以使用OpenCV和numpy来准备输入数据。
 - nv12输入的模型可以使用codec, jpu, vpu, gpu等硬件设备来准备输入数据，或者直接给TROS对应的功能包使用。


### 部分编译日志参考
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

可以看到: 
 - 在X5上, YOLOv8n大约能跑260FPS, 实际由于前处理, 量化节点和部分反量化节点在CPU上进行, 会略慢一些. 实测3个线程可达到252FPS的吞吐量. 
 - 尾部的transpose节点满足被动量化逻辑, 支持被BPU加速, 同时不影响其父节点Conv卷积算子以int32高精度输出.
 - 所有节点的余弦相似度均 > 0.9, 符合预期.
 - 所有算子均在BPU上, 整个bin模型只有1个BPU子图.

## 模型训练

 - 模型训练请参考ultralytics官方文档，这个文档由ultralytics维护，质量非常的高。网络上也有非常多的参考材料，得到一个像官方一样的预训练权重的模型并不困难。
 - 请注意，训练时无需修改任何程序，无需修改forward方法。

## 性能数据

RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | 浮点精度 | 量化精度 | 延迟/吞吐量(单线程) | 延迟/吞吐量(多线程) |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv8n | 640×640 | 80 | 3.2 | 37.3 |  |  |
| YOLOv8s | 640×640 | 80 | 11.2 | 44.9 |  |  |
| YOLOv8m | 640×640 | 80 | 25.9 | 50.2 |  |  |
| YOLOv8l | 640×640 | 80 | 43.7 | 52.9 |  |  |
| YOLOv8x | 640×640 | 80 | 68.2 | 53.9 |  |  |

目标检测 Detection (Open Image V7)
| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | 浮点精度 | 量化精度 | 平均帧延迟/吞吐量(单线程) | 平均帧延迟/吞吐量(多线程) |
|------|------|-------|---------|---------|-------------------|--------------------|--------------------|
| YOLOv8n | 640×640 | 600 | 3.5 | 18.4 |  |  |
| YOLOv8s | 640×640 | 600 | 11.4 | 27.7 |  |  |
| YOLOv8m | 640×640 | 600 | 26.2 | 33.6 |  |  |
| YOLOv8l | 640×640 | 600 | 44.1 | 34.9 |  |  |
| YOLOv8x | 640×640 | 600 | 68.7 | 36.3 |  |  |

说明: 
1. X5的状态为最佳状态：CPU为8*A55@1.8G, 全核心Performance调度, BPU为1*Bayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点mAP：50-95精度使用pycocotools计算，来自于COCO数据集，可以参考微软的论文，此处用于评估板端部署的精度下降程度。