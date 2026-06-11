[English](./README.md) | 简体中文

# Ultralytics YOLO 模型转换与编译指南

本目录提供 Ultralytics YOLO 模型导出、量化编译和 HBM 产物检查所需的脚本、资源和说明，目标产物为适配 RDK S100/S100P 的 BPU 量化 `.hbm` 模型。

## 目录结构

```text
.
├── export_monkey_patch.py          # Ultralytics YOLO ONNX 导出脚本
├── mapper.py                       # 准备校准数据并调用 OpenExplore 编译工具
├── imgs/                           # 转换流程说明图片
├── README.md
└── README_cn.md
```

## 模型编译环境

模型编译请在 x86 Linux 主机的 RDK S100 OpenExplore Docker 环境中完成，不建议在板端安装编译工具链。

工具链入口：

- OE Docker 下载文档：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链下载：<https://toolchain.d-robotics.cc/>

### 1. 安装 Docker

按照 Docker 官方说明安装并验证：<https://docs.docker.com/engine/install/>

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. 获取并加载离线镜像

请访问 [D-Robotics 开发者文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F) 下载适配 RDK S100 系列的 CPU 版本 Docker 镜像。

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

请将 `ai_toolchain_ubuntu_22_s100_xxx.tar` 替换为实际下载的文件名。

### 3. 启动容器

建议使用以下命令启动容器，将当前工作目录挂载到容器中，并增大共享内存以避免编译过程中的内存问题。

```bash
# 假设当前位于 rdk_mode_zoo_mc_rdks 根目录
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

`<docker-image-name>` 可通过 `sudo docker images` 查看加载后的镜像名称和标签。
## 转换流程
### 高性能计算流程介绍
#### 目标检测 (Obeject Detection)
![](./imgs/ultralytics_yolo_detect_dataflow.png)

公版处理流程中, 是会对8400个bbox完全计算分数, 类别和xyxy坐标, 这样才能根据GT去计算损失函数. 但是我们在部署中, 只需要合格的bbox就好了, 并不需要对8400个bbox完全计算.
优化处理流程中, 主要就是利用Sigmoid函数单调性做到了先筛选, 再计算. 对DFL和特征解码的部分也做到了先筛选, 再计算, 节约了大量的计算. 从而使得inference time大大缩短.

 - Classify部分,ReduceMax操作
ReduceMax操作是沿着Tensor的某一个维度找到最大值,此操作用于找到8400个Grid Cell的80个分数的最大值. 操作对象是每个Grid Cell的80类别的值,在C维度操作. 注意,这步操作给出的是最大值,并不是80个值中最大值的索引.
激活函数Sigmoid具有单调性,所以Sigmoid作用前的80个分数的大小关系和Sigmoid作用后的80个分数的大小关系不会改变.
$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
综上,bin模型直接输出的最大值(反量化完成)的位置就是最终分数最大值的位置,bin模型输出的最大值经过Sigmoid计算后就是原来onnx模型的最大值.

 - Classify部分,Threshold(TopK)操作
此操作用于找到8400个Grid Cell中,符合要求的Grid Cell. 操作对象为8400个Grid Cell,在H和W的维度操作. 如果您有阅读我的程序,你会发现我将后面H和W维度拉平了,这样只是为了程序设计和书面表达的方便,它们并没有本质上的不同.
我们假设某一个Grid Cell的某一个类别的分数记为$x$,激活函数作用完的整型数据为$y$,阈值筛选的过程会给定一个阈值,记为$C$,那么此分数合格的**充分必要条件**为:

$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$

由此可以得出此分数合格的**充分必要条件**为:

$$x > -ln\left(\frac{1}{C}-1\right)$$

此操作会符合条件的Grid Cell的索引(indices)和对应Grid Cell的最大值,这个最大值经过Sigmoid计算后就是这个Grid Cell对应类别的分数了.

 - Classify部分,GatherElements操作和ArgMax操作
使用Threshold(TopK)操作得到的符合条件的Grid Cell的索引(indices),在GatherElements操作中获得符合条件的Grid Cell,使用ArgMax操作得到具体是80个类别中哪一个最大,得到这个符合条件的Grid Cell的类别.

 - Bounding Box部分,GatherElements操作
使用Threshold(TopK)操作得到的符合条件的Grid Cell的索引(indices), 在GatherElements操作中获得符合条件的Grid Cell, 得到1×64×k×1的bbox信息.

 - Bounding Box部分,DFL: SoftMax+Conv操作
每一个Grid Cell会有4个数字来确定这个框框的位置,DFL结构会对每个框的某条边基于anchor的位置给出16个估计,对16个估计求SoftMax,然后通过一个卷积操作来求期望,这也是Anchor Free的核心设计,即每个Grid Cell仅仅负责预测1个Bounding box. 假设在对某一条边偏移量的预测中,这16个数字为 $ l_p $ 或者$(t_p, t_p, b_p)$,其中$p = 0,1,...,15$那么偏移量的计算公式为:

$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

 - Bounding Box部分,Decode: dist2bbox(ltrb2xyxy)操作
此操作将每个Bounding Box的ltrb描述解码为xyxy描述,ltrb分别表示左上右下四条边距离相对于Grid Cell中心的距离,相对位置还原成绝对位置后,再乘以对应特征层的采样倍数,即可还原成xyxy坐标,xyxy表示Bounding Box的左上角和右下角两个点坐标的预测值.
![](./imgs/ltrb2xyxy.jpg)

图片输入为$Size=640$,对于Bounding box预测分支的第$i$个特征图$(i=1, 2, 3)$,对应的下采样倍数记为$Stride(i)$,在YOLOv8 - Detect中,$Stride(1)=8, Stride(2)=16, Stride(3)=32$,对应特征图的尺寸记为$n_i = {Size}/{Stride(i)}$,即尺寸为$n_1 = 80, n_2 = 40 ,n_3 = 20$三个特征图,一共有$n_1^2+n_2^2+n_3^3=8400$个Grid Cell,负责预测8400个Bounding Box.
对特征图i,第x行y列负责预测对应尺度Bounding Box的检测框,其中$x,y \in [0, n_i)\bigcap{Z}$,$Z$为整数的集合. DFL结构后的Bounding Box检测框描述为$ltrb$描述,而我们需要的是$xyxy$描述,具体的转化关系如下:

$$x_1 = (x+0.5-l)\times{Stride(i)}$$

$$y_1 = (y+0.5-t)\times{Stride(i)}$$

$$x_2 = (x+0.5+r)\times{Stride(i)}$$

$$y_1 = (y+0.5+b)\times{Stride(i)}$$

最终的检测结果,包括类别(id),分数(score)和位置(xyxy).

#### 实例分割 (Instance Segmentation)
![](./imgs/ultralytics_yolo_seg_dataflow.png)

 - Mask Coefficients 部分, 两次GatherElements操作,
用于得到最终符合要求的Grid Cell的Mask Coefficients信息, 也就是32个系数.
这32个系数与Mask Protos部分作一个线性组合, 也可以认为是加权求和, 就可以得到这个Grid Cell对应目标的Mask信息.

#### 姿态估计 (Pose Estimation)
![](./imgs/ultralytics_yolo_pose_dataflow.png)

Ultralytics YOLO Pose 的关键点基于目标检测, kpt的定义参考如下
```python
COCO_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
```

Ultralytics YOLO Pose 模型的目标检测部分与 Ultralytics YOLO Detect一致, 对应的感受野会多出Channel = 57的特征图, 对应着17个Key Points, 分别是相对于特征图下采样倍数的坐标x, y和这个点对应的分数score.

我们通过目标检测部分, 得知在某个位置的Key Points符合要求后, 将其乘以对应感受野的下采样倍数, 即可得到基于输入尺寸的Key Points坐标.


### 1. 环境准备与模型训练

注：模型训练与 ONNX 导出操作应在具有硬件加速（如 NVIDIA GPU）的 x86 机器上进行。推荐使用 Ubuntu 22.04, Python 3.10 环境。

1. **配置环境**：
   下载 `ultralytics/ultralytics` 仓库，并参考 [Ultralytics 官方文档](https://docs.ultralytics.com/quickstart/) 配置训练环境。
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip install -e .
   ```

2. **模型训练**：
   参考 [Ultralytics 训练指南](https://docs.ultralytics.com/modes/train/) 完成模型训练。训练过程中无需修改任何模型代码或 `forward` 逻辑。

3. **获取权重**：
   使用训练得到的 `.pt` 权重文件，或从 [Ultralytics Releases](https://github.com/ultralytics/assets/releases) 下载预训练权重。

### 2. 导出 ONNX

**重要提示**：ONNX 导出必须在安装了 `ultralytics` 及其所有依赖的训练/开发环境中进行，**严禁在板端执行此步骤**。

1. **运行导出脚本**：
   将本目录下的 `export_monkey_patch.py` 拷贝至 `ultralytics` 训练环境。
   该脚本利用猴子补丁（Monkey Patch）技术，在不修改原始库代码的情况下，将模型适配为 BPU 友好的计算流（如先 TopK 筛选再 DFL 解码）。

   ```bash
   # 以 YOLO11n 为例
   python3 export_monkey_patch.py --pt yolo11n.pt
   ```

2. **产物验证**：
   导出的 `yolo11n.onnx` 会保存在 `.pt` 文件同级目录，该模型已包含 BPU 适配逻辑。

### 3. 模型编译 (mapper)

模型编译请在 RDK S100/S100P OpenExplore 工具链环境中执行。建议使用 OE Docker 离线镜像，不在板端安装和运行编译工具链。

工具链入口：

- OE Docker 下载：[S100 算法工具链](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
- OE 工具链在线手册：[https://toolchain.d-robotics.cc/](https://toolchain.d-robotics.cc/)

在 OpenExplore 工具链环境中运行本目录下的 `mapper.py`。需要准备校准图片和 ONNX 模型；脚本会自动准备校准数据和编译 YAML 配置文件，转换完成的 `.hbm` 模型会保存在 ONNX 模型同级目录或 `--output-dir` 指定目录。

```bash
cd samples/vision/ultralytics_yolo/conversion

# RDK S100 (Nash-E)
python3 mapper.py --onnx yolo11n.onnx --cal-images ./cal_images --march nash-e

# RDK S100P (Nash-M)
python3 mapper.py --onnx yolo11n.onnx --cal-images ./cal_images --march nash-m
```

这个脚本暴露了一些常见参数，默认值已经满足大多数需求。

```bash
$ python3 mapper.py -h
usage: mapper.py [-h] [--cal-images CAL_IMAGES] [--onnx ONNX] [--quantized QUANTIZED] [--jobs JOBS] [--optimize-level OPTIMIZE_LEVEL]
                 [--cal-sample CAL_SAMPLE] [--cal-sample-num CAL_SAMPLE_NUM] [--save-cache SAVE_CACHE] [--cal CAL] [--ws WS]

options:
  -h, --help                        show this help message and exit
  --cal-images CAL_IMAGES           *.jpg, *.png calibration images path, 20 ~ 50 pictures is OK.
  --onnx ONNX                       origin float onnx model path.
  --march MARCH                     S100: nash-e, S100P: nash-m
  --quantized QUANTIZED             int8 first / int16 first
  --jobs JOBS                       model combine jobs.
  --optimize-level OPTIMIZE_LEVEL   O0, O1, O2
  --cal-sample CAL_SAMPLE           sample calibration data or not.
  --cal-sample-num CAL_SAMPLE_NUM   num of sample calibration data.
  --save-cache SAVE_CACHE           remove bpu output files or not.
  --cal CAL                         calibration_data_temporary_folder
  --ws WS                           temporary workspace
```

生成的 `.hbm` 文件建议命名为：

- S100: `*_nashe_*_nv12.hbm`
- S100P: `*_nashm_*_nv12.hbm`

模型文件需放入 sample 的 `model/nash-e/` 或 `model/nash-m/` 目录，供 `runtime/python/run.sh` 和 `runtime/python/main.py` 使用。
## 输入输出协议

### 输入协议

Ultralytics YOLO runtime 使用 NV12 输入，并固定为两个输入 tensor：

- `input[0]`: Y plane
- `input[1]`: UV plane

转换侧生成的模型必须保持该输入协议。

### 输出协议

Python runtime 按固定索引解析输出：

- Detection: `[cls, box] * 3`
- YOLOv10 Detection: `[bbox, score, class_id]`
- Segmentation: `[cls, box, mask_coeff] * 3 + proto`
- Pose: `[cls, box, keypoints] * 3`
- Classification: 单个分类输出 tensor

当前 runtime 覆盖以下模型族和任务组合：

| 模型族 | Detection | Segmentation | Pose | Classification |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv5u | 支持 | 不支持 | 不支持 | 不支持 |
| YOLOv8 | 支持 | 支持 | 支持 | 支持 |
| YOLOv10 | 支持 | 不支持 | 不支持 | 不支持 |
| YOLO11 | 支持 | 支持 | 支持 | 支持 |
| YOLO12 | 支持 | 不支持 | 不支持 | 不支持 |

具体后处理实现见 `runtime/python/yolo_*.py`。

## 编译结果检查

```bash
hrt_model_exec model_info --model_file yolo11n_detect_nashm_640x640_nv12.hbm
hrt_model_exec perf --model_file yolo11n_detect_nashm_640x640_nv12.hbm --thread_num 1
```

## 常见问题

- **权限问题**：宿主机复制回文件时出现权限错误，可检查文件属主或使用 `sudo chown -R`。
- **内存/IPC 报错**：启动 Docker 容器时请添加 `--shm-size=15g`。
- **优化等级报错**：Nash 架构不支持 `O3` 时，请使用 `O0`、`O1` 或 `O2`。

## License

本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
