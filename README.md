English| [简体中文](./README_cn.md)

![](resource/imgs/model_zoo_logo.jpg)

## Introduction to RDK Model Zoo

RDK Model Zoo [RDK] (https://d-robotics.cc/rdkRobotDevKit) based development, provide the most mainstream deployment routines of the algorithm. The routines contain processes for exporting D-Robotics *.bin models, reasoning about D-robotics *.bin models using apis such as Python. Some models also include data acquisition, model training, export, transformation, and deployment processes.

**RDK Model Zoo currently provides the following types of model references**

 - [Large Model]([rdk_model_zoo/llm](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/)): `./rdk_model_zoo/demos/llm`
 - [Image classification](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/classification): `./rdk_model_zoo/demos/classification`
 - [Object Detection](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/detect): `./rdk_model_zoo/demos/detect`
 - [Instance_Segmentation](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/Instance_Segmentation): `./rdk_model_zoo/demos/Instance_Segmentation`

**RDK Model Zoo supports the following platforms**
- Support [RDK X5](https://developer.d-robotics.cc/rdkx5), [RDK Ultra](https://developer.d-robotics.cc/rdkultra) platforms (Bayse)
- Support [RDK S100](), [RDK S100P]() platform (Nash)
- Partially supported [RDK X3](https://developer.d-robotics.cc/rdkx3) platform (Bernoulli2)

**Recommender system version**
- RDK X3: RDK OS 3.0.0, Based on Ubuntu 22.04 aarch64, TROS-Humble.
- RDK X5: RDK OS 3.0.0, Based on Ubuntu 22.04 aarch64, TROS-Humble.
- RDK Ultra: RDK OS 1.0.0, Based on Ubuntu 20.04 aarch64, TROS-Foxy.

## Resources
- [D-Robotics](https://d-robotics.cc/): https://d-robotics.cc/
- [D-Robotics Developer Community](https://developer.d-robotics.cc/): https://developer.d-robotics.cc/
- [RDK User manual](https://developer.d-robotics.cc/information): https://developer.d-robotics.cc/information


## Contents
- [Introduction to RDK Model Zoo](#introduction-to-rdk-model-zoo)
- [Resources](#resources)
- [Contents](#contents)
- [RDK Model Zoo model data reference](#rdk-model-zoo-model-data-reference)
  - [Image classification](#image-classification)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module)
    - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module)
  - [Object Detection](#object-detection)
  - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module-1)
  - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module-1)
  - [Instance Segmentation](#instance-segmentation)
  - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module-2)
  - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module-2)
  - [Semantic segmentation](#semantic-segmentation)
  - [Panoramic segmentation](#panoramic-segmentation)
  - [Keypoint detection](#keypoint-detection)
- [RDK board preparation](#rdk-board-preparation)
- [Library installation references](#library-installation-references)
  - [RDK Model Zoo Python API (recommended)](#rdk-model-zoo-python-api-recommended)
  - [D-Robotics System Software BSP C/C++ \& Python API (Reference)](#d-robotics-system-software-bsp-cc--python-api-reference)
  - [D-Robotics ToolChain C API (Reference)](#d-robotics-toolchain-c-api-reference)
- [RDK Model Zoo with Jupyter (recommended)](#rdk-model-zoo-with-jupyter-recommended)
- [RDK Model Zoo with VSCode (reference)](#rdk-model-zoo-with-vscode-reference)
- [Feedback](#feedback)



## RDK Model Zoo model data reference

### Image classification
#### RDK X5 & RDK X5 Module
| Architecture   | Model       | Size    | Categories | Parameter | Floating point Top-1 | Quantization Top-1 | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | -------------------------- | ----------- | -------- | --------- | --------- | --------- | ----------- | ----------- | ----------- |
| Transformer | EdgeNeXt_base              | 224x224     | 1000     | 18.51     | 78.21     | 74.52     | 8.80        | 32.31       | 113.35      |
|             | EdgeNeXt_small             | 224x224     | 1000     | 5.59      | 76.50     | 71.75     | 4.41        | 14.93       | 226.15      |
|             | **EdgeNeXt_x_small**       | **224x224** | **1000** | **2.34**  | **71.75** | **66.25** | **2.88**    | **9.63**    | **345.73**  |
|             | EdgeNeXt_xx_small          | 224x224     | 1000     | 1.33      | 69.50     | 64.25     | 2.47        | 7.24        | 403.49      |
|             | EfficientFormer_l3         | 224x224     | 1000     | 31.3      | 76.75     | 76.05     | 17.55       | 65.56       | 60.52       |
|             | **EfficientFormer_l1**     | **224x224** | **1000** | **12.3**  | **76.12** | **65.38** | **5.88**    | **20.69**   | **191.605** |
|             | EfficientFormerv2_s2       | 224x224     | 1000     | 12.6      | 77.50     | 70.75     | 6.99        | 26.01       | 152.40      |
|             | **EfficientFormerv2_s1**   | **224x224** | **1000** | **6.12**  | **77.25** | **68.75** | **4.24**    | **14.35**   | **275.95**  |
|             | EfficientFormerv2_s0       | 224x224     | 1000     | 3.57      | 74.25     | 68.50     | 5.79        | 19.96       | 198.45      |
|             | **EfficientViT_MSRA_m5**   | **224x224** | **1000** | **12.41** | **73.75** | **72.50** | **6.34**    | **22.69**   | **174.70**  |
|             | FastViT_SA12               | 224x224     | 1000     | 10.93     | 78.25     | 74.50     | 11.56       | 42.45       | 93.44       |
|             | FastViT_S12                | 224x224     | 1000     | 8.86      | 76.50     | 72.0      | 5.86        | 20.45       | 193.87      |
|             | **FastViT_T12**            | **224x224** | **1000** | **6.82**  | **74.75** | **70.43** | **4.97**    | **16.87**   | **234.78**  |
|             | FastViT_T8                 | 224x224     | 1000     | 3.67      | 73.50     | 68.50     | 2.09        | 5.93        | 667.21      |
| CNN         | ConvNeXt_nano              | 224x224     | 1000     | 15.59     | 77.37     | 71.75     | 5.71        | 19.80       | 200.18      |
|             | ConvNeXt_pico              | 224x224     | 1000     | 9.04      | 77.25     | 71.03     | 3.37        | 10.88       | 364.07      |
|             | **ConvNeXt_femto**         | **224x224** | **1000** | **5.22**  | **73.75** | **72.25** | **2.46**    | **7.11**    | **556.02**  |
|             | ConvNeXt_atto              | 224x224     | 1000     | 3.69      | 73.25     | 69.75     | 1.96        | 5.39        | 732.10      |
|             | Efficientnet_B4            | 224x224     | 1000     | 19.27     | 74.25     | 71.75     | 5.44        | 18.63       | 212.75      |
|             | Efficientnet_B3            | 224x224     | 1000     | 12.19     | 76.22     | 74.05     | 3.96        | 12.76       | 310.30      |
|             | Efficientnet_B2            | 224x224     | 1000     | 9.07      | 76.50     | 73.25     | 3.31        | 10.51       | 376.77      |
|             | FasterNet_S                | 224x224     | 1000     | 31.18     | 77.04     | 76.15     | 6.73        | 24.34       | 162.83      |
|             | FasterNet_T2               | 224x224     | 1000     | 15.04     | 76.50     | 76.05     | 3.39        | 11.56       | 342.48      |
|             | **FasterNet_T1**           | **224x224** | **1000** | **7.65**  | **74.29** | **71.25** | **1.96**    | **5.58**    | **708.40**  |
|             | FasterNet_T0               | 224x224     | 1000     | 3.96      | 71.75     | 68.50     | 1.41        | 3.48        | 1135.13     |
|             | GoogLeNet                  | 224x224     | 1000     | 6.81      | 68.72     | 67.71     | 2.19        | 6.30        | 626.27      |
|             | MobileNetv1                | 224x224     | 1000     | 1.33      | 71.74     | 65.36     | 1.27        | 2.90        | 1356.25     |
|             | **Mobilenetv2**            | **224x224** | **1000** | **3.44**  | **72.0**  | **68.17** | **1.42**    | **3.43**    | **1152.07** |
|             | **Mobilenetv3_large_100**  | **224x224** | **1000** | **5.47**  | **74.75** | **64.75** | **2.02**    | **5.53**    | **714.22**  |
|             | Mobilenetv4_conv_medium    | 224x224     | 1000     | 9.68      | 76.75     | 75.14     | 2.42        | 6.91        | 572.36      |
|             | **Mobilenetv4_conv_small** | **224x224** | **1000** | **3.76**  | **70.75** | **68.75** | **1.18**    | **2.74**    | **1436.22** |
|             | MobileOne_S4               | 224x224     | 1000     | 14.82     | 78.75     | 76.50     | 4.58        | 15.44       | 256.52      |
|             | MobileOne_S3               | 224x224     | 1000     | 10.19     | 77.27     | 75.75     | 2.93        | 9.04        | 437.85      |
|             | MobileOne_S2               | 224x224     | 1000     | 7.87      | 74.75     | 71.25     | 2.11        | 6.04        | 653.68      |
|             | **MobileOne_S1**           | **224x224** | **1000** | **4.83**  | **72.31** | **70.45** | **1.31**    | **3.69**    | **1066.95** |
|             | **MobileOne_S0**           | **224x224** | **1000** | **2.15**  | **69.25** | **67.58** | **0.80**    | **1.59**    | **2453.17** |
|             | RepGhostNet_200            | 224x224     | 1000     | 9.79      | 76.43     | 75.25     | 2.89        | 8.76        | 451.42      |
|             | RepGhostNet_150            | 224x224     | 1000     | 6.57      | 74.75     | 73.50     | 2.20        | 6.30        | 626.60      |
|             | RepGhostNet_130            | 224x224     | 1000     | 5.48      | 75.00     | 73.57     | 1.87        | 5.30        | 743.56      |
|             | RepGhostNet_111            | 224x224     | 1000     | 4.54      | 72.75     | 71.25     | 1.71        | 4.47        | 881.19      |
|             | **RepGhostNet_100**        | **224x224** | **1000** | **4.07**  | **72.50** | **72.25** | **1.55**    | **4.08**    | **964.69**  |
|             | RepVGG_B1g2                | 224x224     | 1000     | 41.36     | 77.78     | 68.25     | 9.77        | 36.19       | 109.61      |
|             | RepVGG_B1g4                | 224x224     | 1000     | 36.12     | 77.58     | 62.75     | 7.58        | 27.47       | 144.39      |
|             | RepVGG_B0                  | 224x224     | 1000     | 14.33     | 75.14     | 60.36     | 3.07        | 9.65        | 410.55      |
|             | RepVGG_A2                  | 224x224     | 1000     | 25.49     | 76.48     | 62.97     | 6.07        | 21.31       | 186.04      |
|             | **RepVGG_A1**              | **224x224** | **1000** | **12.78** | **74.46** | **62.78** | **2.67**    | **8.21**    | **482.20**  |
|             | RepVGG_A0                  | 224x224     | 1000     | 8.30      | 72.41     | 51.75     | 1.85        | 5.21        | 757.73      |
|             | RepViT_m1_1                | 224x224     | 1000     | 8.27      | 77.73     | 77.50     | 2.32        | 6.69        | 590.42      |
|             | **RepViT_m1_0**            | **224x224** | **1000** | **6.83**  | **76.75** | **76.50** | **1.97**    | **5.71**    | **692.29**  |
|             | RepViT_m0_9                | 224x224     | 1000     | 5.14      | 76.32     | 75.75     | 1.65        | 4.37        | 902.69      |
|             | ResNet18                   | 224x224     | 1000     | 11.27     | 71.49     | 70.50     | 2.95        | 8.81        | 448.79      |
|             | ResNeXt50_32x4d            | 224x224     | 1000     | 24.99     | 76.25     | 76.00     | 5.89        | 20.90       | 189.61      |
|             | VargConvNet                | 224x224     | 1000     | 2.03      | 74.51     | 73.69     | 3.99        | 12.75       | 310.29      |

#### RDK X3 & RDK X3 Module

| Architecture   | Model       | Size    | Categories | Parameter | Floating point Top-1 | Quantization Top-1 | Latency/throughput (single-threaded) | Latency/throughput (multi-threaded) | Frame rate(FPS) |
| ----------- | -------------------------- | ----------- | -------- | --------- | --------- | --------- | ----------- | ----------- | ----------- |
| CNN | GoogLeNet                  | 224x224     | 1000     | 6.81      | 68.72     | 67.71     | 8.34        | 16.29       | 243.51      |
|     | Mobilenetv4                | 224x224     | 1000     | 3.76      | 70.50     | 70.26     | 1.43        | 2.96        | 1309.17     |
|     | Mobilenetv2                | 224x224     | 1000     | 3.4       | 72.0      | 68.17     | 2.41        | 4.42        | 890.99      |
|     | Mobilenetv4                | 224x224     | 1000     | 3.76      | 70.50     | 70.26     | 1.43        | 2.96        | 1309.17     |
|     | MobileOne                  | 224x224     | 1000     | 4.8       | 72.00     | 71.00     | 4.50        | 8.70        | 455.87      |
|     | RepGhost                   | 224x224     | 1000     | 4.07      | 72.50     | 72.25     | 2.09        | 4.56        | 855.18      |
|     | RepVGG                     | 224x224     | 1000     | 12.78     | 74.46     | 62.78     | 11.58       | 22.71       | 174.94      |
|     | RepViT                     | 224x224     | 1000     | 5.1       | 75.25     | 75.75     | 28.34       | 41.22       | 96.47       |
|     | ResNet18                   | 224x224     | 1000     | 11.2      | 71.49     | 70.50     | 8.87        | 17.07       | 232.74      |

### Object Detection
### RDK X5 & RDK X5 Module
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| fcos_efficientnetb0 | 512×512 | 80 | - | 323.0 FPS | 9 ms |
| fcos_efficientnetb2 | 768×768 | 80 | - | 70.9 FPS | 16 ms |
| fcos_efficientnetb3 | 896×896 | 80 | - | 38.7 FPS | 20 ms |
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 13.1 FPS | 12 ms |
| YOLOv8n | 640×640 | 80 | 3.2 M | 263.6 FPS | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 M | 194.9 FPS | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 M | 35.7 FPS | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 M | 17.9 FPS | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 M | 11.2 FPS | 5 ms |
| YOLOv10n | 640×640 | 80 | 6.7 G | 132.7 FPS | 4.5 ms | 
| YOLOv10s | 640×640 | 80 | 21.6 G | 71.0 FPS | 4.5 ms |  
| YOLOv10m | 640×640 | 80 | 59.1 G | 34.5 FPS | 4.5 ms |  
| YOLOv10b | 640×640 | 80 | 92.0 G | 25.4 FPS | 4.5 ms |  
| YOLOv10l | 640×640 | 80 | 120.3 G | 20.0 FPS | 4.5 ms |  
| YOLOv10x | 640×640 | 80 | 160.4 G | 14.5 FPS | 4.5 ms |  


### RDK X3 & RDK X3 Module
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| fcos | 512×512 | 80 | - | 173.9 FPS | 5 ms |
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 38.2 FPS | 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 3.9 FPS | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 37.2 FPS | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 20.9 FPS | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 3.6 FPS | 13 ms |
| YOLOv8n | 640×640 | 80 | 3.2 M | 34.1 FPS | 6 ms |
| YOLOv10n | 640×640 | 80 | 6.7 G | 18.1 FPS | 5 ms | 

The model details, including BPU frame latency, BPU throughput, post-processing time, test conditions, etc. are in the `README.md` folder corresponding to `demos/detect`.

### Instance Segmentation
### RDK X5 & RDK X5 Module
Instance Segmentation (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv8n-seg | 640×640 | 80 | 3.4 M  | 175.3 FPS | 6 ms |
| YOLOv8s-seg | 640×640 | 80 | 11.8 M | 67.7 FPS | 6 ms |
| YOLOv8m-seg | 640×640 | 80 | 27.3 M | 27.0 FPS | 6 ms |
| YOLOv8l-seg | 640×640 | 80 | 46.0 M | 14.4 FPS | 6 ms |
| YOLOv8x-seg | 640×640 | 80 | 71.8 M | 8.9 FPS | 6 ms |


### RDK X3 & RDK X3 Module
Instance Segmentation (COCO) Instance Segmentation
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv8n-seg | 640×640 | 80 | 3.4 M | 27.3 FPS | 6 ms |

The model details, including BPU frame latency, BPU throughput, post-processing time, test conditions, etc. are in the `README.md` of the `demos/Instance_Segmentation` subfolder.

### Semantic segmentation
[TODO]

### Panoramic segmentation
[TODO]

### Keypoint detection
[TODO]

## RDK board preparation

Refer to the [RDK user's manual](https://developer.d-robotics.cc/information) to ensure that the board can access the Internet properly.

-RDK board can be connected using ssh, and the IP address of the RDK board can be known by entering commands to the RDK board through Termianl. Including but not limited to MobaXtern, Windows Terminal, etc.
- Use the VSCode Remote SSH plugin to remotely connect to the RDK board. You can use VSCode normally, or use other ides.
- Using VNC to access the board, can operate the board through the graphical interface of xfce.
- Connect the board using HDMI and operate the board through the graphical interface of xfce.

## Library installation references

### RDK Model Zoo Python API (recommended)
Install the bpu_infer_lib library using pip

with RDK X5：
```bash
pip install bpu_infer_x5 -i http://archive.d-robotics.cc:8080/simple/ --trusted-host archive.d-robotics.cc
```

with RDK X3：
```bash
pip install bpu_infer_x3 -i http://archive.d-robotics.cc:8080/simple/ --trusted-host archive.d-robotics.cc
```

### D-Robotics System Software BSP C/C++ & Python API (Reference)

Burn with the system, managed as a debian package.
```bash
sudo apt update # Make sure you have an archive.d-robotics.cc source
sudo apt install hobot-spdev
sudo apt show hobot-spdev
```

### D-Robotics ToolChain C API (Reference)
Burn with system, is the most basic C API.
Also refer to the [RDK user Manual Algorithm Toolchain](https://developer.d-robotics.cc/rdk_doc/04_toolchain_development) section for the OE package, from which you can obtain libdnn.so and its header files.

## RDK Model Zoo with Jupyter (recommended)

install jupyterlab
```
pip install jupyterlab
```

Then use the following command to clone the Model Zoo repository:

```
git clone https://github.com/D-Robotics/rdk_model_zoo
```

Note: by default, git clone will get the RDK X5 version of the model zoo. If you are using other products of RDK series, please use `git checkout` command to switch the branch. For example, if you are using RDK X3, please execute the following command

```
git checkout rdk_x3
```

After the clone is complete, use the following command to enter the Model Zoo directory:

```
cd rdk_model_zoo
```

Then use the following command to enter Jupyter Lab (Note: The IP address is the IP address used by the board for actual login):

```
jupyter lab --allow-root --ip 192.168.1.10
```
![](resource/imgs/jupyter_start.png)

After using the command, the above log will appear. Hold down Ctrl and left click the link boxed in the figure to enter Jupyter Lab (as shown in the figure below). Double-click "demos" to select the model to experience RDK Model Zoo.

![](resource/imgs/into_jupyter.png)

After selecting a model's notebook in Jupyter Lab, the developer will enter an interface similar to the following:

![](resource/imgs/basic_usage.png)

Here, taking the Yolo World model as an example, users only need to click the boxed button in the above figure to run all cells. Scroll to the bottom to see the result display.

![](resource/imgs/basic_usage_res.png)

Developers can also choose to run cell by cell. At this time, just press Shift + Enter to complete the current cell run and jump to the next cell.

## RDK Model Zoo with VSCode (reference)

Use VSCode Remote SSH plug-in, remote login into the board, open the corresponding folder of RDK Model Zoo repository, enter the corresponding model folder, you can view the README, edit the program, run the program.
![](resource/imgs/vscode_demo.jpg)

Note: In all programs, the relative path starts with the directory where the model is located.

![](resource/imgs/demo_rdkx5_yolov10n_detect.jpg)


## Feedback
If you have any questions or encounter any issues, we warmly welcome you to post them on the [D-Robotics Developer Community](https://developer.d-robotics.cc) or submit an issue/comment directly in this repository. Your feedback is invaluable to us, and we are always eager to assist you and improve our resources.

We appreciate your interest and support, and we look forward to hearing from you. Whether it's a simple query, a feature request, or a bug report, please do not hesitate to reach out. We strive to make the RDK Model Zoo as useful and accessible as possible, and your input helps us achieve that goal.

Thank you for choosing the RDK Model Zoo, and we hope to serve you well!