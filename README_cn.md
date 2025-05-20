
![](resource/imgs/model_zoo_logo.jpg)

[English](./README.md) | 简体中文

# ⭐️ 点个Star不迷路, 感谢您的关注 ⭐️

## RDK Model Zoo 简介

RDK Model Zoo 基于[RDK](https://d-robotics.cc/rdkRobotDevKit)开发, 提供大多数主流算法的部署例程. 例程包含导出D-Robotics *.bin模型, 使用 Python 等 API 推理 D-Robotics *.bin模型的流程. 部分模型还包括数据采集, 模型训练, 导出, 转化, 部署流程.

**RDK Model Zoo 目前提供以下类型模型参考.**

 - [图像分类](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/classification): `./rdk_model_zoo/demos/classification`
  
 - [目标检测](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/detect): `./rdk_model_zoo/demos/detect`

 - [实例分割](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/Instance_Segmentation): `./rdk_model_zoo/demos/Instance_Segmentation`

 - [大模型](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/llm): `./rdk_model_zoo/demos/llm`

**RDK Model Zoo 支持如下平台.**
 - 支持 [RDK X5](https://developer.d-robotics.cc/rdkx5), [RDK Ultra](https://developer.d-robotics.cc/rdkultra) 平台 (Bayse)
 - 支持 [RDK S100](), [RDK S100P]() 平台 (Nash)
 - 部分支持 [RDK X3](https://developer.d-robotics.cc/rdkx3) 平台 (Bernoulli2)

**推荐系统版本**
- RDK X3: RDK OS 3.0.0, Based on Ubuntu 22.04 aarch64, TROS-Humble.
- RDK X5: RDK OS 3.0.0, Based on Ubuntu 22.04 aarch64, TROS-Humble.
- RDK Ultra: RDK OS 1.0.0, Based on Ubuntu 20.04 aarch64, TROS-Foxy.

## ⭐️ RDK板卡准备

参考[RDK用户手册](https://developer.d-robotics.cc/information), 使得板卡能正常访问互联网从, 确保能做到以下条件之一.

 - 利用ssh连接RDK板卡, 可以通过Termianl向RDK板卡输入命令, 知道RDK板卡的IP地址. 包括但是不限于MobaXtern, Windows Terminal等.
 - 利用VSCode Remote SSH插件远程连接RDK板卡, 可以正常的使用VSCode, 也可使用其他的IDE.
 - 利用VNC访问板卡, 能通过xfce的图形化界面操作板卡.
 - 利用HDMI连接板卡, 能通过xfce的图形化界面操作板卡.

## ⭐️ 依赖库安装参考

### RDK Model Zoo Python API (推荐)
使用pip完成bpu_infer_lib库的安装

如使用RDK X5：
```bash
pip install bpu_infer_lib_x5 -i  http://sdk.d-robotics.cc:8080/simple/  --trusted-host sdk.d-robotics.cc
```

使用 RDK X3：
```bash
pip install bpu_infer_lib_x3 -i  http://sdk.d-robotics.cc:8080/simple/  --trusted-host sdk.d-robotics.cc
```

### D-Robotics System Software BSP C/C++ & Python API (参考)

随系统烧录, 以debian包的形式管理.
```bash
sudo apt update # 确保有 archive.d-robotics.cc 源
sudo apt install hobot-spdev
sudo apt show hobot-spdev
```

### D-Robotics ToolChain C API (参考)
随系统烧录, 是最基本的C API.
还可以参考[RDK用户手册算法工具链](https://developer.d-robotics.cc/rdk_doc/04_toolchain_development)章节, 获取OE包, 从OE包中获取libdnn.so及其头文件.

## ⭐️ 使用 Jupyter 体验RDK Model Zoo (推荐)
安装jupyterlab
```bash
pip install jupyterlab
```

随后即可使用如下命令拉取Model Zoo仓库：
```bash
git clone https://github.com/D-Robotics/rdk_model_zoo
```

注：这里git clone拉下来的分支默认为RDK X5分支，如实际使用的开发板为RDK系列的其他产品，请使用git checkout命令进行切换，这里以X3为例，如想切换至RDK X3对应的分支，请执行如下命令：

```bash
git checkout rdk_x3
```

拉取完成后，使用如下命令进入Model Zoo目录：
```bash
cd rdk_model_zoo
```

随后使用如下命令启动Jupyter Lab（注：ip地址为板子实际登录时使用的ip）:
```bash
jupyter lab --allow-root --ip 192.168.1.10
```
![](resource/imgs/jupyter_start.png)

使用命令后，会出现以上日志，按住Ctrl，鼠标左键点击上图所示的链接，即可进入Jupyter Lab（如下图所示），双击demos后，即可选择模型体验RDK Model Zoo。

![](resource/imgs/into_jupyter.png)

开发者可跳转至对应的模块，体验模型在RDK 系列开发板的部署。

在Jupyter Lab中选择一个模型的notebook进入后，开发者会进入到类似如下的界面：

![](resource/imgs/basic_usage.png)

这里以yolo world模型为例，用户只需要点击上图中的双三角按钮，即可运行全部cell。鼠标拖动到下方，即可看到结果展示：

![](resource/imgs/basic_usage_res.png)

开发者也可以选择逐cell运行，此时只需要按下Shift + Enter，即可完成当前cell运行，并跳转至下一个cell。

## ⭐️ 使用VSCode体验RDK Model Zoo (参考)

使用VSCode Remote SSH插件, 远程登录进板卡, 打开RDK Model Zoo的仓库对应的文件夹, 进入对应的模型文件夹, 可以查看README, 编辑程序, 运行程序.
![](resource/imgs/vscode_demo.jpg)

注: 所有的程序中, 相对路径均以模型所在的目录开始计算.

![](resource/imgs/demo_rdkx5_yolov10n_detect.jpg)

## ⭐️ RDK参考资源

[地瓜机器人](https://d-robotics.cc/)

[地瓜开发者社区](https://developer.d-robotics.cc/)

[RDK用户手册](https://developer.d-robotics.cc/information)

[社区资源中心](https://developer.d-robotics.cc/resource)

[RDK X3 算法工具链社区手册](https://developer.d-robotics.cc/api/v1/fileData/horizon_xj3_open_explorer_cn_doc/index.html)

[RDK X3 OpenExplore 产品发布](https://developer.d-robotics.cc/forumDetail/136488103547258769)

[RDK Ultra 算法工具链社区手册](https://developer.d-robotics.cc/api/v1/fileData/horizon_j5_open_explorer_cn_doc/index.html)

[RDK Ultra OpenExplore 产品发布](https://developer.d-robotics.cc/forumDetail/118363912788935318)

[RDK X5 算法工具链社区手册](https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)

[RDK X5 OpenExplore 产品发布](https://developer.d-robotics.cc/forumDetail/251934919646096384)

## ⭐️ 反馈
如果您有任何问题或遇到任何问题, 我们热烈欢迎您将它们发布到[地瓜开发者社区](https://developer.d-robotics.cc)或直接在此仓库中提交issue/comment. 您的反馈对我们来说是无价的, 我们一直渴望帮助您, 并改善我们的资源.

## ⭐️ FAQ

### 自己训练模型的精度不满足预期

- 请检查OpenExplore工具链Docker, 板端libdnn.so的版本是否均为目前发布的最新版本.
- 请检查在导出模型时，是否有按照对应examples的文件夹内的README的要求进行。
- 每一个输出节点的余弦相似度是否均达到0.999以上(保底0.99).

### 自己训练模型的速度不满足预期

- Python API 的推理性能会较弱一些，请基于 C/C++ API 测试性能。
- 性能数据不包含前后处理，与完整demo的耗时是存在差异的，一般来说采用nv12输入的模型可以做到end2end吞吐量等于BPU吞吐量。
- 板子是否已经定频到对应README内的最高频率。
- 是否有其他应用占用了 CPU/BPU 及 DDR 带宽资源，这会导致推理性能减弱。

### 如何解决模型量化掉精度问题

- 根据平台版本，先参考对应平台的文档，参考PTQ章节的精度debug章节进行精度debug。
- 如果是模型结构特性、权重分布导致 int8 量化掉精度，请考虑使用混合量化或QAT量化。

### Can't reshape 1354752 in (1,3,640,640)
您好，请修改同级目录下preprocess.py文件中的分辨率，修改为准备转化的onnx一样大小的分辨率，并删除所有的校准数据集，再重新运行02脚本，生成校准数据集。
目前这个示例的校准数据集来自../../../01common/calibration data/coco目录，生成在./calibration_data_rgb_f32目录

### 为什么其他模型没有demo，是因为不支持吗

你好，不是。

- 受限于项目排期，为了照顾大部分地瓜开发者的需求，我们挑选了提问频率较高的模型作为demo示例。如果有更好的模型推荐，欢迎前往地瓜开发者社区反馈。
- 同时，BPU及算法工具链相关资源均已经在开发者社区进行释放，自定义的模型完全可以自己进行转化。

### mAP 精度相比ultralytics官方的结果低一些

- ultralytics官方测mAP时，使用动态shape模型, 而BPU使用了固定shape模型，map测试结果会比动态shape的低一些。
- RDK Solutions使用pycocotools计算的精度比ultralytics计算的精度会低一些是正常现象, 主要原因是两者计算方法有细微差异, 我们主要是关注同样的一套计算方式去测试定点模型和浮点模型的精度, 从而来评估量化过程中的精度损失.
- BPU 模型在量化和NCHW-RGB888输入转换为YUV420SP(nv12)输入后, 也会有一部分精度损失。

### 不修改YOLO模型结构直接导出的ONNX可以使用吗

可以，但不推荐。

- 公版的模型结构或者自己设计的输出头结构，需要自行对后处理代码进行程序设计。
- RDK Solutions仓库提供的模型结构的调整方式是经过了精度和性能的考虑, 其他的修改方法暂未经过测试，也欢迎大家探索更多的高性能和高精度的修改方法。

### 模型要先转onnx才能量化吗/地平线工具链如何使用
PTQ方案下需要先导出为onnx或者caffe，将onnx或者caffe转化为bin模型。QAT方案下需要重新搭建torch模型进行量化感知训练，从pt模型转为hbm模型。

### 训练的时候需要修改输出头吗？
训练的适合全部按照公版的来，只有导出的时候再修改，这样训练的适合和训练的损失函数计算那套就能对上，部署的适合就能和板子上跑的代码的后处理那套对上。

### 模型进行推理时会进行cpu处理吗
你好，在模型转化的过程中，无法量化的算子或者不满足BPU约束，不满足被动量化逻辑的算子会回退到CPU计算。特别的，对于一个全部都是BPU算子的bin模型，bin模型的前后会有量化和反量化节点，负责将float转int，和int转float，这两种节点是由CPU来计算的。
