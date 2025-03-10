English | [简体中文](./README_cn.md)

# YOLOv11 Detect


## Introduction to YOLO

YOLO (You Only Look Once) is a popular object detection and image segmentation model developed by Joseph Redmon and Ali Farhadi of the University of Washington. YOLO was introduced in 2015 and quickly gained popularity due to its high speed and accuracy.

 - YOLOv2, released in 2016, improved upon the original model by incorporating batch normalization, anchor boxes, and dimension clustering.
 - YOLOv3: The third iteration of the YOLO model family, originally by Joseph Redmon, known for its efficient real-time object detection capabilities.
 - YOLOv4: A darknet-native update to YOLOv3, released by Alexey Bochkovskiy in 2020.
 - YOLOv5: An improved version of the YOLO architecture by Ultralytics, offering better performance and speed trade-offs compared to previous versions.
 - YOLOv6: Released by Meituan in 2022, and in use in many of the company's autonomous delivery robots.
 - YOLOv7: Updated YOLO models released in 2022 by the authors of YOLOv4.
 - YOLOv8: The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
 - YOLOv9: An experimental model trained on the Ultralytics YOLOv5 codebase implementing Programmable Gradient Information (PGI).
 - YOLOv10: By Tsinghua University, featuring NMS-free training and efficiency-accuracy driven architecture, delivering state-of-the-art performance and latency.
 - YOLO11 🚀 NEW: Ultralytics' latest YOLO models delivering state-of-the-art (SOTA) performance across multiple tasks.
- YOLO12 builds a YOLO framework centered around attention mechanisms, employing innovative methods and architectural improvements to break the dominance of CNN models within the YOLO series. This enables real-time object detection with faster inference speeds and higher detection accuracy.


## Standard Processing Flow
![YOLOv11_Detect_Origin.png](imgs/YOLOv11_Detect_Origin.png)

## Optimized Processing Flow
![YOLOv11_Detect_Quantize.png](imgs/YOLOv11_Detect_Quantize.png)

In the standard processing flow, scores, classes, and xyxy coordinates for all 8400 bounding boxes (bbox) are fully calculated to compute the loss function based on ground truth (GT). However, during deployment, only qualified bounding boxes are needed, not requiring full computation of all 8400 bbox. The optimized processing flow mainly utilizes the monotonicity of the Sigmoid function to first screen and then compute. Meanwhile, by leveraging advanced indexing with Python's numpy, it also screens before computing parts like DFL and feature decoding, saving considerable computation, thus allowing post-processing on a CPU using numpy to be completed in just 5 milliseconds per frame with a single core and thread.

- **Classify Section, Dequantize Operation**
During model compilation, if all dequantization operators are removed, dequantization needs to be manually performed on the three output heads of the Classify section within post-processing. There are multiple ways to check the dequantization coefficients, including viewing the logs from `hb_mapper` or using the BPU inference interface API. Note that each C dimension has different dequantization coefficients; each head has 80 dequantization coefficients, which can be directly multiplied using numpy broadcasting. This dequantization is implemented in the bin model, so the output obtained is in float32.

- **Classify Section, ReduceMax Operation**
The ReduceMax operation finds the maximum value along a certain dimension of a tensor, used here to find the maximum score among the 80 scores of the 8400 Grid Cells. This operation targets the values of each Grid Cell's 80 categories, operating on the C dimension. It's important to note that this step provides the maximum value, not the index of the maximum value among the 80.
Given the monotonic nature of the Sigmoid activation function, the relationship between the sizes of the 80 scores before and after applying Sigmoid remains unchanged.
$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
Therefore, the position of the maximum value directly outputted by the bin model (after dequantization) represents the final maximum score position, and the maximum value outputted by the bin model, after being processed by Sigmoid, equals the maximum value of the original ONNX model.

- **Classify Section, Threshold (TopK) Operation**
This operation identifies qualifying Grid Cells among the 8400 Grid Cells, targeting the 8400 Grid Cells across the H and W dimensions. For convenience in programming and description, these dimensions are flattened, but there's no essential difference. Assuming the score of a certain category in a Grid Cell is denoted as \(x\), and the integer data after activation is \(y\), the thresholding process involves setting a threshold \(C\). A necessary and sufficient condition for a score to qualify is:
$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$
From this, we derive that the condition for qualification is:
$$x > -ln\left(\frac{1}{C}-1\right)$$
This operation retrieves the indices and corresponding maximum values of qualifying Grid Cells, where the maximum value, after Sigmoid calculation, represents the score for that category.

- **Classify Section, GatherElements Operation and ArgMax Operation**
Using the indices obtained from the Threshold (TopK) operation, the GatherElements operation retrieves qualifying Grid Cells. The ArgMax operation then determines which of the 80 categories has the highest score, identifying the category of the qualifying Grid Cell.

- **Bounding Box Section, GatherElements Operation and Dequantize Operation**
Similarly, using the indices from the Threshold (TopK) operation, the GatherElements operation retrieves qualifying Grid Cells. Each C dimension has different dequantization coefficients; each head has 64 dequantization coefficients, which can be directly multiplied using numpy broadcasting to obtain bbox information of shape 1×64×k×1.

- **Bounding Box Section, DFL: SoftMax + Conv Operation**
Each Grid Cell uses four numbers to determine the position of its bounding box. The DFL structure provides 16 estimates for one edge of the bounding box based on the anchor's position, applies SoftMax to these estimates, and uses a convolution operation to calculate the expectation. This is central to Anchor-Free design, where each Grid Cell predicts only one bounding box. If the 16 numbers predicting an offset are denoted as \(l_p\) or \((t_p, t_p, b_p)\), where \(p = 0,1,...,15\), the offset calculation formula is:
$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

- **Bounding Box Section, Decode: dist2bbox(ltrb2xyxy) Operation**
This operation decodes each bounding box's ltrb description into an xyxy description. The ltrb represents distances from the top-left and bottom-right corners relative to the Grid Cell center, which are then converted back to absolute positions and scaled according to the feature layer's sampling ratio to yield the predicted xyxy coordinates.

For input images sized at 640, YOLOv8-Detect features three feature maps (\(i=1,2,3\)) with downsample ratios \(Stride(i)\) of 8, 16, and 32 respectively. This corresponds to feature map sizes of \(n_1=80\), \(n_2=40\), and \(n_3=20\), totaling 8400 Grid Cells responsible for predicting 8400 bounding boxes. 

YOLO versions v8, v9, and v11 include an NMS operation to eliminate redundant object detections, whereas YOLOv10 does not require this. The final detection results include the class (id), score, and location (xyxy).




## Step Reference

Note: For any errors such as "No such file or directory", "No module named 'xxx'", "command not found", etc., please check carefully. Do not simply copy and run each command one by one. If you do not understand the modification process, please visit the developer community to start learning from YOLOv5.

### Environment and Project Preparation
- Download the ultralytics/ultralytics repository and configure the environment according to the official Ultralytics documentation.
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
- Enter the local repository and download the official pre-trained weights. Here we use the YOLO11n-Detect model with 2.6 million parameters as an example.
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### Export to ONNX
- Uninstall yolo-related command-line commands so that modifications directly in the `./ultralytics/ultralytics` directory take effect.
```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If exists, uninstall
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```
If it's not straightforward, you can confirm the location of the `ultralytics` directory that needs to be modified using the following Python command:
```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# or
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```
- Modify the optimized Attention module.
File path: `ultralytics/nn/modules/block.py`, around line 868, replace the `forward` method of the `Attntion` class with the following content. The main optimization points are removing some useless data movement operations and changing the Reduce dimension to C for better BPU compatibility, which currently doubles the throughput of the BPU without needing to retrain the model.
Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for easier switching back during training.
```python
class Attention(nn.Module):   # RDK
        print(f"{x.shape = }")
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```

- Modify the Detect output head to separately output Bounding Box information and Classify information for three feature layers, resulting in a total of six output heads.

File path: `./ultralytics/ultralytics/nn/modules/head.py`, around line 58, replace the `forward` method of the `Detect` class with the following content.
Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for easier switching back during training.
```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result

## If the order of output heads is reversed between bbox and cls, adjust the append order of cv2 and cv3 accordingly,
## then re-export the ONNX model and compile it into a bin model.
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

- Run the following Python script. If there is a **No module named onnxsim** error, install it.
Note: If the generated ONNX model shows too high IR version, set `simplify=False`. Both settings have no impact on the final bin model but turning it on improves readability in Netron.
```python
from ultralytics import YOLO
YOLO('yolov11n.pt').export(imgsz=640, format='onnx', simplify=False, opset=11)
```

### Calibration Data Preparation
Refer to the minimalist calibration data preparation script provided by the RDK Model Zoo: `https://github.com/D-Robotics/rdk_model_zoo/blob/main/demos/tools/generate_calibration_data/generate_calibration_data.py` for preparing calibration data.

### PTQ Quantization Conversion

- Refer to the Tian Gong Kai Wu toolchain manual and OE package to check the model. Ensure all operators are on the BPU before compiling.
```bash
(bpu_docker) $ hb_mapper checker --model-type onnx --march bayes-e --model yolo11n.onnx
```
- If you did not rewrite the Attention module equivalently, based on the model check results, find the manually quantized operator Softmax, which should look like this. The Softmax operator splits the model into two BPU subgraphs. The name of the Softmax operator here is "/model.10/m/m.0/attn/Softmax". If you have rewritten the Attention module, this step will not show the Softmax operator, and you can proceed directly to model compilation.
```bash
/model.10/m/m.0/attn/MatMul      BPU  id(0)  HzSQuantizedMatmul   --   1.0  int8      
/model.10/m/m.0/attn/Mul         BPU  id(0)  HzSQuantizedConv     --   1.0  int8      
/model.10/m/m.0/attn/Softmax     CPU  --     Softmax              --   --   float     
/model.10/m/m.0/attn/Transpose_1 BPU  id(1)  Transpose            --   --   int8      
/model.10/m/m.0/attn/MatMul_1    BPU  id(1)  HzSQuantizedMatmul   --   1.0  int8      
```
Modify the following content in the corresponding YAML file:
```yaml
model_parameters:
  node_info: {"/model.10/m/m.0/attn/Softmax": {'ON': 'BPU','InputType': 'int8','OutputType': 'int8'}}
# If accuracy does not meet standards, consider using the following configuration, or simply delete the node_info configuration item and use FP32 for computing the Softmax operator.
model_parameters:
  node_info: {"/model.10/m/m.0/attn/Softmax": {'ON': 'BPU','InputType': 'int16','OutputType': 'int16'}}
```
For YOLO11 l and x models, specify two SoftMax operators to the BPU:
```yaml
model_parameters:
  node_info: {"/model.10/m/m.0/attn/Softmax": {'ON': 'BPU','InputType': 'int8','OutputType': 'int8'},
              "/model.10/m/m.1/attn/Softmax": {'ON': 'BPU','InputType': 'int8','OutputType': 'int8'}}
```
Note: You can choose to use int8 quantization for the Softmax operator. On the COCO2017 validation dataset of 5000 images, the mAP:.50-.95 accuracy remains consistent. If int8 cannot control precision loss, consider using int16 or omitting this configuration item to compute Softmax using FP32. At the end of this document, performance data for these three configurations for the YOLO11n model are provided.

- Model Compilation:
```bash
(bpu_docker) $ hb_mapper makertbin --model-type onnx --config yolo11_detect_bayese_640x640_nv12.yaml
```

### Remove Dequantization Nodes from 3 Bbox Output Heads
- Check the names of the dequantization nodes for the 3 bbox output heads.
Through the logs generated during `hb_mapper makerbin`, observe that the names of the three outputs with sizes [1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64] are 475, 497, 519.
```bash
ONNX IR version:          9
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Version:                  None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    output0:              shape=[1, 80, 80, 80], dtype=FLOAT32
    475:                  shape=[1, 80, 80, 64], dtype=FLOAT32
    489:                  shape=[1, 40, 40, 80], dtype=FLOAT32
    497:                  shape=[1, 40, 40, 64], dtype=FLOAT32
    511:                  shape=[1, 20, 20, 80], dtype=FLOAT32
    519:                  shape=[1, 20, 20, 64], dtype=FLOAT32
```

- Enter the compiled product directory.
```bash
$ cd yolo11n_detect_bayese_640x640_nv12
```
- Check which dequantization nodes can be removed.
```bash
$ hb_model_modifier yolo11n_detect_bayese_640x640_nv12.bin
```
- In the generated `hb_model_modifier.log` file, find the following information. Mainly identify the names of the three output heads with sizes [1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64]. Alternatively, you can view the ONNX model using tools like Netron to obtain the output head names.
The names are:
> "/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize"
> "/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize"
> "/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"




```bash
2024-10-24 14:03:23,588 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/cv2.0/cv2.0.2/Conv_output_0_quantized"
input: "/model.23/cv2.0/cv2.0.2/Conv_x_scale"
output: "475"
name: "/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-10-24 14:03:23,588 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/cv2.1/cv2.1.2/Conv_output_0_quantized"
input: "/model.23/cv2.1/cv2.1.2/Conv_x_scale"
output: "497"
name: "/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"

2024-10-24 14:03:23,588 file: hb_model_modifier.py func: hb_model_modifier line No: 409 input: "/model.23/cv2.2/cv2.2.2/Conv_output_0_quantized"
input: "/model.23/cv2.2/cv2.2.2/Conv_x_scale"
output: "519"
name: "/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize"
op_type: "Dequantize"
```

- Use the following command to remove the aforementioned three dequantization nodes. Note that the names may differ upon export, so please verify carefully.
```bash
$ hb_model_modifier yolo11n_detect_bayese_640x640_nv12.bin \
-r /model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize \
-r /model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize \
-r /model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize
```
- Upon successful removal, the following logs will be displayed:
```bash
2024-10-24 14:19:59,425 INFO log will be stored in /open_explorer/bin_dir/yolo11n_detect_bayese_640x640_nv12/hb_model_modifier.log
2024-10-24 14:19:59,430 INFO Nodes that will be removed from this model: ['/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize', '/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize', '/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize']
2024-10-24 14:19:59,431 INFO Node '/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-10-24 14:19:59,431 INFO scale: /model.23/cv2.0/cv2.0.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-10-24 14:19:59,431 INFO Node '/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize' is removed
2024-10-24 14:19:59,431 INFO Node '/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-10-24 14:19:59,432 INFO scale: /model.23/cv2.1/cv2.1.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-10-24 14:19:59,432 INFO Node '/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize' is removed
2024-10-24 14:19:59,432 INFO Node '/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2024-10-24 14:19:59,433 INFO scale: /model.23/cv2.2/cv2.2.2/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2024-10-24 14:19:59,433 INFO Node '/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize' is removed
2024-10-24 14:19:59,436 INFO modified model saved as yolo11n_detect_bayese_640x640_nv12_modified.bin
```

- The resulting bin model is named `yolo11n_detect_bayese_640x640_nv12_modified.bin`, which is the final model.
- For models with NCHW input format, input data can be prepared using OpenCV and numpy.
- For models with NV12 input format, input data can be prepared using hardware devices such as codec, JPU, VPU, GPU, or directly used by the corresponding TROS functional packages.

### Use the `hb_perf` command to visualize the bin model, and `hrt_model_exec` command to check the input and output conditions of the bin model

- To visualize the bin model before removing the dequantization coefficients:
```bash
hb_perf yolo11n_detect_bayese_640x640_nv12.bin
```
Results can be found in the `hb_perf_result` directory:
![](./imgs/yolo11n_detect_bayese_640x640_nv12.png)

To check the input and output information of the bin model before removing the dequantization coefficients:
```bash
hrt_model_exec model_info --model_file yolo11n_detect_bayese_640x640_nv12.bin
``` 
This will display the input and output details of the bin model prior to the removal of dequantization coefficients.


```bash
[HBRT] set log level as 0. version = 3.15.55.0
[DNN] Runtime version = 1.24.5_(3.15.55 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-10-24,14:27:27.649.970) [HorizonRT] The model builder version = 1.24.3
Load model to DDR cost 32.671ms.
This model file has 1 model:
[yolo11n_detect_bayese_640x640_nv12]
---------------------------------------------------------------------
[model name]: yolo11n_detect_bayese_640x640_nv12

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
valid shape: (1,80,80,80,)
aligned shape: (1,80,80,80,)
aligned byte size: 2048000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (2048000,25600,320,4,)

output[1]: 
name: 475
valid shape: (1,80,80,64,)
aligned shape: (1,80,80,64,)
aligned byte size: 1638400
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1638400,20480,256,4,)

output[2]: 
name: 489
valid shape: (1,40,40,80,)
aligned shape: (1,40,40,80,)
aligned byte size: 512000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (512000,12800,320,4,)

output[3]: 
name: 497
valid shape: (1,40,40,64,)
aligned shape: (1,40,40,64,)
aligned byte size: 409600
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (409600,10240,256,4,)

output[4]: 
name: 511
valid shape: (1,20,20,80,)
aligned shape: (1,20,20,80,)
aligned byte size: 128000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (128000,6400,320,4,)

output[5]: 
name: 519
valid shape: (1,20,20,64,)
aligned shape: (1,20,20,64,)
aligned byte size: 102400
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (102400,5120,256,4,)
```

- To visualize the bin model after removing the targeted dequantization coefficients:
```bash
hb_perf yolo11n_detect_bayese_640x640_nv12_modified.bin
```
You can find the following results in the `hb_perf_result` directory.
![./imgs/yolo11n_detect_bayese_640x640_nv12_modified.png]

To check the input and output information of the bin model after removing the dequantization coefficients:
```bash
hrt_model_exec model_info --model_file yolo11n_detect_bayese_640x640_nv12_modified.bin
```
This will display the input and output details of the bin model after the removal of dequantization nodes, along with all the dequantization coefficients that have been removed. This indicates that such information is stored within the bin model and can be accessed using the inference library's API, facilitating corresponding pre-processing and post-processing tasks. 

These steps ensure you can effectively analyze the modified model's performance and structure, making it easier to integrate into your application for efficient inference.



```bash
[HBRT] set log level as 0. version = 3.15.55.0
[DNN] Runtime version = 1.24.5_(3.15.55 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-10-24,14:27:47.191.283) [HorizonRT] The model builder version = 1.24.3
Load model to DDR cost 26.723ms.
This model file has 1 model:
[yolo11n_detect_bayese_640x640_nv12]
---------------------------------------------------------------------
[model name]: yolo11n_detect_bayese_640x640_nv12

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
valid shape: (1,80,80,80,)
aligned shape: (1,80,80,80,)
aligned byte size: 2048000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (2048000,25600,320,4,)

output[1]: 
name: 475
valid shape: (1,80,80,64,)
aligned shape: (1,80,80,64,)
aligned byte size: 1638400
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (1638400,20480,256,4,)
scale data: 0.000562654,0.000563576,0.000520224,0.000490708,0.000394319,0.000409077,0.000273487,0.000322834,0.000290781,0.000224716,0.0001839,0.000253425,0.000245584,0.000213301,0.000184822,0.000230596,0.000426833,0.000469723,0.000417609,0.000438362,0.000391782,0.000347508,0.000300697,0.000262418,0.000196583,0.000230596,0.000243048,0.000228751,0.000205115,0.000179403,0.000153577,0.000170871,0.000506388,0.000524836,0.000505927,0.00034059,0.000308768,0.000404465,0.000313841,0.000359499,0.000293548,0.00023613,0.000253886,0.000228174,0.000198312,0.000175137,0.000157958,0.000210995,0.000551124,0.000522069,0.000512845,0.000378869,0.000458885,0.000320067,0.000335747,0.000299313,0.000355348,0.000298852,0.000203155,0.000186437,0.000162109,0.000139395,0.000123138,0.000208574,
quantizeAxis: 3

output[2]: 
name: 489
valid shape: (1,40,40,80,)
aligned shape: (1,40,40,80,)
aligned byte size: 512000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (512000,12800,320,4,)

output[3]: 
name: 497
valid shape: (1,40,40,64,)
aligned shape: (1,40,40,64,)
aligned byte size: 409600
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (409600,10240,256,4,)
scale data: 0.000606957,0.000609319,0.000573893,0.000428649,0.000335125,0.000327568,0.000299228,0.000375038,0.000271359,0.000310091,0.00026144,0.000226369,0.000198737,0.000197792,0.000187637,0.000247742,0.000555944,0.000539413,0.000461949,0.0004662,0.000497374,0.000392515,0.000368662,0.000314342,0.000262621,0.000224007,0.000236288,0.000221528,0.000200627,0.000178308,0.00015481,0.000162485,0.000624434,0.000620655,0.00051863,0.000449668,0.000437623,0.000371023,0.000345281,0.000274902,0.000324498,0.000285057,0.000224598,0.000184685,0.000227078,0.000243491,0.000239358,0.000305368,0.000515323,0.000524298,0.000455808,0.000439749,0.000389445,0.000483204,0.000369134,0.000284585,0.000360159,0.000290017,0.000231801,0.000187637,0.000180906,0.000190235,0.000183977,0.000234517,
quantizeAxis: 3

output[4]: 
name: 511
valid shape: (1,20,20,80,)
aligned shape: (1,20,20,80,)
aligned byte size: 128000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (128000,6400,320,4,)

output[5]: 
name: 519
valid shape: (1,20,20,64,)
aligned shape: (1,20,20,64,)
aligned byte size: 102400
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: SCALE
stride: (102400,5120,256,4,)
scale data: 0.000758878,0.000750577,0.000652753,0.000580126,0.000583387,0.000641489,0.00064801,0.00067469,0.00054159,0.000423608,0.000500385,0.000371731,0.000463627,0.000396632,0.000415901,0.000483784,0.000732791,0.000820536,0.000659868,0.000661054,0.000562933,0.000596134,0.000448212,0.000432205,0.000445544,0.000504831,0.000355131,0.000350092,0.000324005,0.000273759,0.00017801,8.41139e-05,0.000806307,0.000808086,0.000591095,0.00062726,0.000571826,0.00054159,0.000581609,0.000391,0.000415308,0.000553447,0.000406711,0.000471038,0.000344459,0.000296585,0.000320152,0.000345941,0.000716191,0.000649789,0.000591984,0.000567676,0.000583091,0.000597616,0.000638524,0.000523803,0.00056738,0.000534772,0.000559376,0.000401375,0.000401672,0.000345941,0.00037766,0.000407304,
quantizeAxis: 3
```
## Efficient Deployment of YOLO11 with TROS

### Install or Update Packages like tros-humble-hobot-dnn
```bash
sudo apt update # Ensure the DiGua APT source is available
sudo apt install -y tros*-dnn-node* tros*-hobot-usb-cam tros*-hobot-codec
```

### Copy Configuration Files for tros-humble-hobot-dnn
```bash
cp -r /opt/tros/humble/lib/dnn_node_example/config .
```
Configure it as follows:
```json
{
        "model_file": "yourself.bin",
        "dnn_Parser": "yolov8",
        "model_output_count": 6,
        "reg_max": 16,
        "class_num": 80,
        "cls_names_list": "config/coco.list",
        "strides": [8, 16, 32],
        "score_threshold": 0.25,
        "nms_threshold": 0.7,
        "nms_top_k": 300
}
```

### Run the YOLOv8 Inference Node
Note: The post-processing for YOLO12 is identical to YOLOv8, thus you can directly use the YOLOv8 inference node.
```bash
# Configure MIPI camera
export CAM_TYPE=mipi
# Configure USB camera
# export CAM_TYPE=usb
# Launch the file
ros2 launch dnn_node_example dnn_node_example.launch.py dnn_example_config_file:=config/my_workconfig.json
```

For more details, refer to the TROS manual: [TROS Manual](https://developer.d-robotics.cc/rdk_doc/Robot_development/boxs/detection/yolo)

Additionally, TROS simplifies AI inference analysis for multiple video streams from various sources such as MIPI Camera, USB Camera, and IPC Camera. Refer to RDK Video Solutions: [RDK Video Solutions](https://github.com/D-Robotics/rdk_model_zoo/blob/main/demos/solutions/RDK_Video_Solutions/README_cn.md)


## Model Training

- For model training, please refer to the official Ultralytics documentation, which is well-maintained by Ultralytics and provides high-quality guidance. There are numerous resources online that make obtaining a pre-trained weights model similar to the official one straightforward.
- Note: During training, no program modifications or changes to the `forward` method are required.


## Performance Data


### RDK X5 & RDK X5 Module
| Model | Resolution (Pixels) | Number of Classes | Parameters | BPU Task Latency/BPU Throughput (Threads) | Post-processing Time |
|---------|---------|-------|---------|---------|----------|
| YOLO11n_fp32softmax | 640×640 | 80 | 2.6 M  / 6.5 B | 23.3 ms / 42.9 FPS (1 thread  ) <br/> 24.0 ms / 83.3 FPS (2 threads) <br/> 38.8 ms / 201.6 FPS (7 threads) | 3 ms |
| YOLOv11n_int16softmax | 640×640 | 80 | 2.6 M  / 6.5 B  | 8.0 ms / 125.0 FPS (1 thread  ) <br/> 12.2 ms / 163.1 FPS (2 threads) | 3 ms |
| YOLO11n | 640×640 | 80 | 2.6 M  / 6.5 B  | 6.7 ms / 148.5 FPS (1 thread  ) <br/> 9.7 ms / 204.3 FPS (2 threads) | 3 ms |
| YOLO11s | 640×640 | 80 | 9.4 M  / 21.5 B  | 13.0 ms / 77.0 FPS (1 thread  ) <br/> 22.1 ms / 90.3 FPS (2 threads) | 3 ms |
| YOLO11m | 640×640 | 80 | 20.1 M / 68.0 B  | 28.6 ms / 34.9 FPS (1 thread  ) <br/> 53.3 ms / 37.4 FPS (2 threads) | 3 ms |
| YOLO11l | 640×640 | 80 | 25.3 M / 86.9 B  | 37.6 ms / 26.6 FPS (1 thread  ) <br/> 71.2 ms / 28.0 FPS (2 threads) | 3 ms |
| YOLO11x | 640×640 | 80 | 56.9 M / 194.9 B | 80.4 ms / 12.4 FPS (1 thread  ) <br/> 156.4 ms / 12.7 FPS (2 threads) | 3 ms |


Notes:
1. BPU latency vs. throughput.
- Single-thread latency represents the delay for processing a single frame on a single thread using a single BPU core, ideal for task execution.
- Multi-thread frame rate involves feeding tasks to the BPU via multiple threads; typically, four threads balance latency and throughput effectively. On the X5, two threads usually suffice to maximize BPU usage.
- Performance data listed in the table stops increasing significantly with additional threads.
- Use these commands for testing on-device:
```bash
hrt_model_exec perf --thread_num 2 --model_file yolo12n_detect_bayese_640x640_nv12_modified.bin

python3 ../../../tools/batch_perf/batch_perf.py --max 3 --file ptq_models
```
1. Testing boards were in their optimal state.
   - Optimal X5 configuration: CPU at 8 × A55@1.8G, full-core performance scheduling, BPU at 1 × Bayes-e@10TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # CPU: 1.8GHz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
echo 1200000000 > /sys/kernel/debug/clk/bpu_mclk_2x_clk/clk_rate # BPU: 1.2GHz
```
   - Optimal X3 configuration: CPU at 4 × A53@1.8G, full-core performance scheduling, BPU at 2 × Bernoulli2@5TOPS.
```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8GHz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```


## Accuracy Data

### RDK X5 & RDK X5 Module
Object Detection (COCO2017)
| Model | Pytorch | YUV420SP<br/>Python | YUV420SP<br/>C/C++ | NCHWRGB<br/>C/C++ |
|---------|---------|-------|---------|---------|
| YOLO11n | 0.323 | 0.308（95.36%） | 0.310（95.98%） | 0.311（96.28%） |
| YOLO11s | 0.394 | 0.375（95.18%） | 0.379（96.19%） | 0.381（96.70%） |
| YOLO11m | 0.436 | 0.418（95.87%） | 0.422（96.79%） | 0.428（98.17%） |
| YOLO11l | 0.452 | 0.429（94.91%） | 0.434（96.02%） | 0.444（98.23%） |
| YOLO11x | 0.466 | 0.445（95.49%） | 0.449（96.35%） | 0.456（97.85%） |



### Testing Methodology
1. All accuracy data was calculated using Microsoft's unmodified `pycocotools` library, focusing on `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
2. All test data used the COCO2017 dataset's validation set of 5000 images, inferred directly on-device, saved as JSON files, and processed through third-party testing tools (`pycocotools`) with score thresholds set at 0.25 and NMS thresholds at 0.7.
3. Lower accuracy from `pycocotools` compared to Ultralytics' calculations is normal due to differences in area calculation methods. Our focus is on evaluating quantization-induced precision loss using consistent calculation methods.
4. Some accuracy loss occurs when converting NCHW-RGB888 input to YUV420SP(nv12) input for BPU models, mainly due to color space conversion. Incorporating this during training can mitigate such losses.
5. Slight discrepancies between Python and C/C++ interface accuracies arise from different handling of floating-point numbers during memcpy and conversions.
6. Test scripts can be found in the RDK Model Zoo eval section: [RDK Model Zoo Eval](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools)
7. This table reflects PTQ results using 50 images for calibration and compilation, simulating typical developer scenarios without fine-tuning or QAT, suitable for general validation needs but not indicative of maximum accuracy.

## Feedback
For any unclear expressions or further inquiries, please visit the DiGua Developer Community.

[DiGua Robotics Developer Community](developer.d-robotics.cc).

## References

[Ultralytics Documentation](https://docs.ultralytics.com/)