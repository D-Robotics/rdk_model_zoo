[English](./README.md) | 简体中文

RDK X5 Model Zoo
=======

## 1. 产品介绍

本产品为 RDK 系列开发板的模型样例总仓（Model Zoo），旨在为开发者提供能直接上板部署的，丰富多样的模型案例。

通过该仓库，开发者可以访问以下资源：
1. **多样化的地瓜异构模型**：仓库中包含了各类可直接上板部署，适用与多种场景、通用性较强的地瓜异构模型，包括但不限于图像分类、目标检测、语义分割、自然语言处理等领域的.bin模型。这些模型经过精心挑选和优化，具有高效的性能。
2. **详细的使用指南**：每个模型都配有一个Jupyter Notebook，其中附带详细的模型介绍、使用说明、示例代码和注释，帮助开发者快速上手。同时，对于部分模型，我们还提供了模型的性能评估报告和调优建议，方便开发者根据具体需求进行定制和优化。
3. **集成的开发工具**：我们为开发者提供了可在RDK 系列开发板上快速部署模型的一套python接口，bpu_infer_lib，通过学习仓库内模型配备的Jupyter Notebook，如数据预处理脚本和推理方法，开发者能快速掌握对该接口的使用，大大简化了模型开发和部署的流程。

## 2. 环境准备

开发者首先根据所在分支，准备一块对应的RDK 开发板，并前往[地瓜机器人官网](https://developer.d-robotics.cc/)完成硬件准备、驱动安装、软件下载、和镜像烧录。

在完成硬件连接和网络配置后，使用MobaXTerm登录开发板，对开发板进行网络连接，即可使用如下命令拉取Model Zoo仓库：
```
git clone https://github.com/D-Robotics/rdk_model_zoo
```

拉取完成后，使用如下命令进入Model Zoo目录：
```
cd rdk_model_zoo
```

随后使用如下命令启动Jupyter Lab（注：ip地址为板子实际登录时使用的ip）:
```
jupyter lab --allow-root --ip 192.168.1.10
```
![](resource/imgs/jupyter_start.png)

使用命令后，会出现以上日志，按住Ctrl，鼠标左键点击上图所示的链接，即可进入Jupyter Lab（如下图所示），双击demos后，即可选择模型体验RDK Model Zoo。

![](resource/imgs/into_jupyter.png)

## 3. 模块介绍

RDK 系列 Model Zoo总体分为如下模块：

1. **[大模型](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/llm)**
2. **[图像分类](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/classification)**
3. **[目标检测](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/detect)**
4. **[实例分割](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/Instance_Segmentation)**

开发者可跳转至对应的模块，体验模型在RDK 系列开发板的部署。

## 4. 使用指南

在Jupyter Lab中选择一个模型的notebook进入后，开发者会进入到类似如下的界面：

![](resource/imgs/basic_usage.png)

这里以yolo world模型为例，用户只需要点击上图中的双三角按钮，即可运行全部cell。鼠标拖动到下方，即可看到结果展示：

![](resource/imgs/basic_usage_res.png)

开发者也可以选择逐cell运行，此时只需要按下Shift + Enter，即可完成当前cell运行，并跳转至下一个cell。