[English](./README.md) | 简体中文

# UNet Semantic Segmentation

- [UNet Semantic Segmentation](#unet-semantic-segmentation)
  - [Introduction to UNet](#introduction-to-unet)
  - [Input and Output Data](#input-and-output-data)
  - [Network Architecture](#network-architecture)
  - [Standard Processing Pipeline](#standard-processing-pipeline)
  - [Optimized Processing Pipeline](#optimized-processing-pipeline)
  - [Step-by-Step Guide](#step-by-step-guide)
    - [Environment and Project Preparation](#environment-and-project-preparation)
    - [Export to ONNX](#export-to-onnx)
    - [PTQ Quantization and Conversion](#ptq-quantization-and-conversion)
    - [Remove the Dequantize Node of the Output Layer](#remove-the-dequantize-node-of-the-output-layer)
    - [Visualize the bin Model with hb_perf and Check I/O with hrt_model_exec](#visualize-the-bin-model-with-hb_perf-and-check-io-with-hrt_model_exec)
    - [Partial Compilation Log Reference](#partial-compilation-log-reference)
  - [Model Training](#model-training)
  - [Performance Data](#performance-data)
    - [RDK X5 & RDK X5 Module](#rdk-x5--rdk-x5-module-1)
    - [RDK X3 & RDK X3 Module](#rdk-x3--rdk-x3-module-1)
  - [Feedback](#feedback)
  - [References](#references)


## Introduction to UNet

UNet is a convolutional neural network for biomedical image segmentation, proposed in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox at the University of Freiburg, Germany. UNet is renowned for its unique U-shaped structure (encoder-decoder architecture with skip connections) and high-precision segmentation capabilities.

- **Encoder**: Uses convolution and pooling operations to progressively extract features and reduce spatial dimensions, capturing contextual information.
- **Decoder**: Restores spatial resolution through upsampling and convolution operations for precise localization.
- **Skip Connections**: Concatenates high-resolution features from the encoder with features in the decoder, preserving fine details and improving segmentation accuracy.
- **Applications**: Initially designed for medical image segmentation (e.g., cell segmentation, tumor detection), it is now widely used in satellite image segmentation, industrial defect detection, autonomous driving scene understanding, and more.

## Model Download

See: `./model` folder

## Input and Output Data

- Input: 1x3x512x512, dtype=UINT8 (NV12 format supported)
- Output 0: [1, 512, 512, 21], dtype=INT32

## Network Architecture

UNet adopts a classic encoder-decoder architecture:

```
Input Image (3×512×512)
    ↓
Encoder Path (Downsampling)
  - Conv + ReLU
  - MaxPool (×4 times, resolution halved, channels doubled)
    ↓
Bottleneck
  - Deepest feature extraction
    ↓
Decoder Path (Upsampling)
  - UpConv/Resize + Concat (Skip Connections)
  - Conv + ReLU (×4 times, resolution doubled, channels halved)
    ↓
Output Layer
  - 1×1 Conv (mapped to number of classes)
  - Softmax/Argmax
```

## Standard Processing Pipeline

![](imgs/UNet_Segmentation_Origin.png)

Standard UNet workflow:

1. Image Preprocessing: Normalization and resize to 512×512.
2. Model Inference: Generate segmentation maps through the encoder-decoder structure.
3. Post-Processing: Softmax to obtain probability maps or Argmax to obtain class indices.
4. Visualization: Map segmentation results to color images.

## Step-by-Step Guide

Note: For any errors such as No such file or directory, No module named "xxx", or command not found, please check carefully. Do not copy and run commands blindly. If you do not understand the modification process, please visit the developer community for guidance.

### Environment and Project Preparation

- Download the UNet implementation repository. Here we use the standard PyTorch implementation as an example:

```bash
git clone https://github.com/bubbliiiing/unet-pytorch.git
cd unet-pytorch
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

### Export to ONNX

Modify `mode` to `export_onnx` in `predict.py`.

- Run the export script:

```bash
python predict.py
```

### PTQ Quantization and Conversion

Refer to the Horizon Open Explorer toolchain manual and OE package to check the model. If all operators are on the BPU, proceed with compilation. The corresponding yaml files are located in the `./ptq_yamls` directory.

```bash
hb_mapper checker --model-type onnx --march bayes-e --model UNet_11.onnx
hb_mapper makertbin --model-type onnx --config unet_bernoulli2.yaml
```

### Remove the Dequantize Node of the Output Layer

- Check the dequantize node name of the output layer.
  Through the logs during `hb_mapper makertbin`, the output with shape [1, 512, 512, 21] is named `output`.

```bash
ONNX IR version:          6                                                                                                                                                                                       
Opset version:            ['ai.onnx v11', 'horizon v1']                                                                                                                                                           
Producer:                 pytorch v2.10.0                                                                                                                                                                         
Domain:                   None                                                                                                                                                                                    
Model version:            None                                                                                                                                                                                    
Graph input:                                                                                                                                                                                                      
    images:               shape=[1, 3, 512, 512], dtype=FLOAT32                                                                                                                                                   
Graph output:                                                                                                                                                                                                     
    output:               shape=[1, 512, 512, 21], dtype=FLOAT32
```

- Enter the compilation output directory:

```bash
$ cd unet_bernoulli2_512x512_nv12
```

- Check dequantize nodes that can be removed:

```bash
$ hb_model_modifier unet_xxx.bin
```

- In the generated `hb_model_modifier.log` file, find the following information. The main goal is to locate the name of the output head with shape [1, 512, 512, 21].
  The name here is:

> "/final/Conv_output_0_HzDequantize"

```bash
2026-04-20 13:56:27,733 INFO log will be stored in /data/horizon_x3/data/unet/hb_model_modifier.log
2026-04-20 13:56:27,780 INFO Nodes that can be deleted: ['/final/Conv_output_0_HzDequantize']
```

- Use the following command to remove the above dequantize node. Note that these names may differ during export; please verify carefully.

```bash
$ hb_model_modifier unet_cityscapes_bayese_512x512_nv12.bin \
-r "/final/Conv_output_0_HzDequantize"
```

- Upon successful removal, the following log will be displayed:

```bash
2026-04-20 14:05:05,063 INFO log will be stored in /data/horizon_x3/data/unet/hb_model_modifier.log
2026-04-20 14:05:05,110 INFO Nodes that will be removed from this model: ['/final/Conv_output_0_HzDequantize']
2026-04-20 14:05:05,110 INFO Node '/final/Conv_output_0_HzDequantize' found, its OP type is 'Dequantize'
2026-04-20 14:05:05,110 INFO scale: /final/Conv_x_scale; zero point: 0. node info details are stored in hb_model_modifier log file
2026-04-20 14:05:05,110 INFO Node '/final/Conv_output_0_HzDequantize' is removed
2026-04-20 14:05:05,337 INFO modified model saved as UNet-resnet-deploy_512x512_nv12_bernoulli/UNet-resnet-deploy_512x512_nv12_x3_modified.bin
```

- The resulting bin model is named `UNet-resnet-deploy_512x512_nv12_x3_modified.bin`. This is the final model.

### Visualize the bin Model with hb_perf and Check I/O with hrt_model_exec

- For the bin model before removing the dequantize coefficients:

```bash
hb_perf UNet-resnet-deploy_512x512_nv12_bernoulli.bin
```

You can find the visualization results in the `hb_perf_result` directory.

```bash
hrt_model_exec model_info --model_file UNet-resnet-deploy_512x512_nv12_x3.bin
```

You can see the input and output information of this bin model before removing the dequantize coefficients:

```bash
I0000 00:00:00.000000  2055 vlog_is_on.cc:197] RAW: Set VLOG level for "*" to 3
core[0] open!
core[1] open!
[HBRT] set log level as 0. version = 3.15.46.0
[HBRT] hbrtSetGlobalConfig, set bpu march to BERNOULLI2(4272728)
[DNN] Runtime version = 1.23.5_(3.15.46 HBRT)
[A][DNN][packed_model.cpp:248][Model](2026-04-20,14:41:43.540.657) [HorizonRT] The model builder version = 1.23.4
Load model to DDR cost 379.023ms.
This model file has 1 model:
[UNet-resnet-deploy_512x512_nv12_x3]
---------------------------------------------------------------------
[model name]: UNet-resnet-deploy_512x512_nv12_x3

input[0]: 
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,512,512,)
aligned shape: (1,3,512,512,)
aligned byte size: 393216
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]: 
name: output
valid shape: (1,21,512,512,)
aligned shape: (1,21,512,512,)
aligned byte size: 22020096
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (22020096,1048576,2048,4,)

---------------------------------------------------------------------
```

- For the bin model after removing the target dequantize coefficients:

```bash
hb_perf UNet-resnet-deploy_512x512_nv12_bernoulli/UNet-resnet-deploy_512x512_nv12_x3_modified.bin
```

```bash
hrt_model_exec model_info --model_info UNet-resnet-deploy_512x512_nv12_bernoulli/UNet-resnet-deploy_512x512_nv12_x3_modified.bin
```

You can see the model information after removing the dequantize node, as well as the stored dequantize coefficients.

```bash
I0000 00:00:00.000000  2068 vlog_is_on.cc:197] RAW: Set VLOG level for "*" to 3
core[0] open!
core[1] open!
[HBRT] set log level as 0. version = 3.15.46.0
[HBRT] hbrtSetGlobalConfig, set bpu march to BERNOULLI2(4272728)
[DNN] Runtime version = 1.23.5_(3.15.46 HBRT)
[A][DNN][packed_model.cpp:248][Model](2026-04-20,14:45:15.978.243) [HorizonRT] The model builder version = 1.23.4
Load model to DDR cost 306.635ms.
This model file has 1 model:
[UNet-resnet-deploy_512x512_nv12_x3]
---------------------------------------------------------------------
[model name]: UNet-resnet-deploy_512x512_nv12_x3

input[0]: 
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,512,512,)
aligned shape: (1,3,512,512,)
aligned byte size: 393216
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]: 
name: output
valid shape: (1,21,512,512,)
aligned shape: (1,21,512,512,)
aligned byte size: 22020096
tensor type: HB_DNN_TENSOR_TYPE_S32
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: SCALE
stride: (22020096,1048576,2048,4,)
scale data: 0.00046273,0.00109842,0.00104222,0.000667546,0.000869199,0.00061894,0.000700378,0.000699254,0.000631821,0.000553886,0.000766205,0.00073237,0.000853829,0.000750434,0.000537595,0.000578083,0.000891571,0.000716755,0.000626697,0.000812128,0.00110989,
quantizeAxis: 1

---------------------------------------------------------------------
```


### Partial Compilation Log Reference

As you can see, this is a model with 100% BPU operator utilization.

```bash
====================================================================================================================================
Node                                                ON   Subgraph  Type              Cosine Similarity  Threshold   In/Out DataType  
-------------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess      0.999995           127.000000  int8/int8        
/conv1/Conv                                         BPU  id(0)     Conv              0.999562           1.013355    int8/int8        
/maxpool/MaxPool                                    BPU  id(0)     MaxPool           0.999438           3.359458    int8/int8        
/layer1/layer1.0/conv1/Conv                         BPU  id(0)     Conv              0.998974           3.359458    int8/int8        
/layer1/layer1.0/conv2/Conv                         BPU  id(0)     Conv              0.998197           1.305064    int8/int8        
/layer1/layer1.0/conv3/Conv                         BPU  id(0)     Conv              0.997957           2.030451    int8/int8        
/layer1/layer1.0/downsample/downsample.0/Conv       BPU  id(0)     Conv              0.999356           3.359458    int8/int8        
/layer1/layer1.1/conv1/Conv                         BPU  id(0)     Conv              0.992931           1.898905    int8/int8        
/layer1/layer1.1/conv2/Conv                         BPU  id(0)     Conv              0.987306           1.335431    int8/int8        
/layer1/layer1.1/conv3/Conv                         BPU  id(0)     Conv              0.986722           2.946419    int8/int8        
/layer1/layer1.2/conv1/Conv                         BPU  id(0)     Conv              0.985223           2.319801    int8/int8        
/layer1/layer1.2/conv2/Conv                         BPU  id(0)     Conv              0.982107           1.243415    int8/int8        
/layer1/layer1.2/conv3/Conv                         BPU  id(0)     Conv              0.973119           3.548386    int8/int8        
/layer2/layer2.0/conv1/Conv                         BPU  id(0)     Conv              0.976369           2.157930    int8/int8        
/layer2/layer2.0/conv2/Conv                         BPU  id(0)     Conv              0.984657           1.354403    int8/int8        
/layer2/layer2.0/conv3/Conv                         BPU  id(0)     Conv              0.977477           1.491932    int8/int8        
/layer2/layer2.0/downsample/downsample.0/Conv       BPU  id(0)     Conv              0.988449           2.157930    int8/int8        
/layer2/layer2.1/conv1/Conv                         BPU  id(0)     Conv              0.996199           1.641783    int8/int8        
/layer2/layer2.1/conv2/Conv                         BPU  id(0)     Conv              0.995920           0.884238    int8/int8        
/layer2/layer2.1/conv3/Conv                         BPU  id(0)     Conv              0.994623           1.603424    int8/int8        
/layer2/layer2.2/conv1/Conv                         BPU  id(0)     Conv              0.990412           1.919210    int8/int8        
/layer2/layer2.2/conv2/Conv                         BPU  id(0)     Conv              0.990391           0.914774    int8/int8        
/layer2/layer2.2/conv3/Conv                         BPU  id(0)     Conv              0.986399           0.885644    int8/int8        
/layer2/layer2.3/conv1/Conv                         BPU  id(0)     Conv              0.988771           1.948690    int8/int8        
/layer2/layer2.3/conv2/Conv                         BPU  id(0)     Conv              0.989107           1.000571    int8/int8        
/layer2/layer2.3/conv3/Conv                         BPU  id(0)     Conv              0.986791           1.038042    int8/int8        
/layer3/layer3.0/conv1/Conv                         BPU  id(0)     Conv              0.986691           2.022372    int8/int8        
/layer3/layer3.0/conv2/Conv                         BPU  id(0)     Conv              0.986085           1.551270    int8/int8        
/layer3/layer3.0/conv3/Conv                         BPU  id(0)     Conv              0.978635           1.275543    int8/int8        
/layer3/layer3.0/downsample/downsample.0/Conv       BPU  id(0)     Conv              0.989848           2.022372    int8/int8
...
.../Relu_output_0_calibrated_0.05978_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                 int8/int8        
/up_concat3/Concat                                  BPU  id(0)     Concat            0.993853           2.022372    int8/int8        
/up_concat3/conv1/Conv                              BPU  id(0)     Conv              0.998320           7.592683    int8/int8        
/up_concat3/conv2/Conv                              BPU  id(0)     Conv              0.998347           32.350964   int8/int8        
/up_concat2/up/Resize                               BPU  id(0)     Resize            0.997129           18.709902   int8/int8        
.../Relu_output_0_calibrated_0.14732_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                 int8/int8        
/up_concat2/Concat                                  BPU  id(0)     Concat            0.996761           2.157930    int8/int8        
/up_concat2/conv1/Conv                              BPU  id(0)     Conv              0.998679           18.709902   int8/int8        
/up_concat2/conv2/Conv                              BPU  id(0)     Conv              0.998735           59.616806   int8/int8        
/up_concat1/up/Resize                               BPU  id(0)     Resize            0.998410           44.007637   int8/int8        
.../Relu_output_0_calibrated_0.34652_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                 int8/int8        
/up_concat1/Concat                                  BPU  id(0)     Concat            0.998253           3.359458    int8/int8        
/up_concat1/conv1/Conv                              BPU  id(0)     Conv              0.998826           44.007637   int8/int8        
/up_concat1/conv2/Conv                              BPU  id(0)     Conv              0.998323           55.837200   int8/int8        
/up_conv/up_conv.0/Resize                           BPU  id(0)     Resize            0.998139           67.128563   int8/int8        
/up_conv/up_conv.1/Conv                             BPU  id(0)     Conv              0.998534           67.128563   int8/int8        
/up_conv/up_conv.3/Conv                             BPU  id(0)     Conv              0.998149           74.564735   int8/int8        
/final/Conv                                         BPU  id(0)     Conv              0.998469           54.988548   int8/int32
```


## Model Training

- Please refer to the original UNet repository documentation or relevant PyTorch tutorials for model training. UNet training typically uses Cross-Entropy loss or Dice loss, with data augmentation including random flipping, rotation, and elastic deformation.
- Please note that no program modifications are required during training; maintain the standard forward propagation logic.
- For medical image segmentation, it is recommended to use binary classification (background + target); for scene understanding (e.g., Cityscapes), use multi-class classification.

## Performance Data

### RDK X5 & RDK X5 Module

Semantic Segmentation (VOC)

| Model         | Size (Pixels) | Classes | Parameters | Throughput (1 thread) <br/> Throughput (Multi-thread)        | Post-processing Time (Python) |
| ------------- | ------------- | ------- | ---------- | ------------------------------------------------------------ | ----------------------------- |
| UNet-resnet50 | 512×512       | 20      | 43.93 M    | 11.23 FPS (1 thread) <br/> 13.23 FPS (2 threads) <br/> 13.23 (8 threads) | 267.08 ms                     |

### RDK X3 & RDK X3 Module

Semantic Segmentation (VOC)

| Model         | Size (Pixels) | Classes | Parameters | Throughput (1 thread) <br/> Throughput (Multi-thread)        | Post-processing Time (Python) |
| ------------- | ------------- | ------- | ---------- | ------------------------------------------------------------ | ----------------------------- |
| UNet-resnet50 | 512×512       | 20      | 43.93 M    | 2.61 FPS (1 thread) <br/> 5.17 FPS (2 threads) <br/> 5.21 FPS (4 threads) | 361.96 ms                     |

```bash
hrt_model_exec perf --thread_num 8 --model_file UNet-resnet-deploy_512x512_nv12_x3_modified.bin
```

Test boards are all in optimal condition.

- X5 optimal status: CPU is 8 × A55@1.8G, all cores in Performance mode, BPU is 1 × Bayes-e@10TOPS.

```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```

- X3 optimal status: CPU is 4 × A53@1.8G, all cores in Performance mode, BPU is 2 × Bernoulli2@5TOPS.

```bash
sudo bash -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost"  # 1.8Ghz
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor" # Performance Mode
```

Regarding post-processing: Currently, on the X5, the Python-based post-processing (including Softmax and Argmax) can be completed in approximately 3-5 ms using a single core and single thread. That is, it only requires 1-2 CPU cores, handling hundreds of frames per minute, so post-processing does not become a bottleneck.

## Feedback

If anything in this document is unclear, please feel free to ask questions and exchange ideas in the D-Robotics Developer Community.

[D-Robotics Developer Community](developer.d-robotics.cc)

## References

[Pytorch-UNet](https://github.com/bubbliiiing/unet-pytorch)
