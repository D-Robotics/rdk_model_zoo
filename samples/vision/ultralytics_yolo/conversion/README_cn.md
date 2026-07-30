[English](./README.md) | 简体中文

# Ultralytics YOLO 模型转换与编译指南

本目录提供 Ultralytics YOLO 模型导出、量化编译和 BIN 产物检查所需的脚本、资源和说明，目标产物为适配 RDK X5 的 BPU 量化 `.bin` 模型。

## 目录结构

```text
.
├── export_monkey_patch.py          # Ultralytics YOLO ONNX 导出脚本
├── mapper.py                       # 准备校准数据并调用 OpenExplorer 编译工具
├── imgs/                           # 转换说明使用的流程图
├── config.yaml                     # 参考 Mapper 配置
├── requirements.txt                # Python 依赖
├── README.md
└── README_cn.md
```

## 模型编译环境

模型编译请在 x86 Linux 主机上进行，不建议在 RDK X5 板端安装或运行编译工具链。RDK X5 支持通过 pip 安装轻量工具链，也可以使用 OpenExplorer Docker 离线镜像。

### 1. Pip 安装工具链

推荐使用 Ubuntu 22.04 + Python 3.10，并通过 Conda 或其他虚拟环境隔离依赖。

```bash
conda create -n rdk_env python=3.10 -y
conda activate rdk_env
pip install rdkx5-yolo-mapper
hb_mapper --version
```

如果下载较慢，可以使用 PyPI 镜像源：

```bash
pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### 2. 获取并加载离线 Docker 镜像

如需使用 Docker，请下载 RDK X5 OpenExplorer 1.2.8 离线镜像包。CPU 镜像用于常规模型转换，GPU 镜像可用于带 GPU 依赖的扩展环境。

```bash
# CPU 镜像
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz

# GPU 镜像（按需下载和加载）
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz

docker images
```

也可以前往地瓜开发者社区获取离线版本的 Docker 镜像：[https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)

### 3. 启动容器

使用下列命令启动容器，将当前仓库挂载到容器中，并增大 shared memory，降低编译期间出现内存或 IPC 问题的概率。

```bash
# 假设当前位于 rdk_model_zoo_mc_rdkx5 仓库根目录
docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

`openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8` 可通过 `docker images` 查看加载后的镜像名称和标签。

## 转换流程

### 高性能计算流程介绍

#### 目标检测 (Object Detection)

![](./imgs/ultralytics_yolo_detect_dataflow.png)

在标准处理流程中，会完整计算全部 8400 个 Bounding Box（bbox）的 scores、categories 和 xyxy 坐标，用于结合 GT 计算 loss。但部署阶段只需要保留满足阈值要求的 bbox，因此没有必要对全部 8400 个 bbox 做完整计算。

这里的优化主要利用 Sigmoid 函数的单调性，在计算前先进行筛选。该思路也用于 DFL 和特征解码阶段：先筛选，再计算，从而节省大量计算量并降低推理耗时。

- **分类部分：ReduceMax 操作**

ReduceMax 用于在指定维度上取最大值。在 YOLO 检测头中，该操作用于在 8400 个 Grid Cell 的 80 个类别分数中找最大值，操作维度为 C 维。该操作输出最大值，而不是最大值对应的类别索引。

Sigmoid 函数具有单调性，因此 80 个分数在 Sigmoid 前后的相对大小关系不变：

$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$

$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$

因此，bin 模型直接输出的最大值位置与最终 score 最大值的位置一致；该最大值经过 Sigmoid 后，与原始 ONNX 模型中的最大 score 一致。

- **分类部分：Threshold(TopK) 操作**

Threshold(TopK) 用于筛选满足阈值要求的 Grid Cell，操作对象是 8400 个 Grid Cell，对 H/W 维度进行筛选。代码中可能将 H/W 展平，便于实现和表达，但本质上没有区别。

设某个 Grid Cell 某一类别的原始分数为 $x$，经过 Sigmoid 后的值为 $y$，阈值为 $C$，则该分数满足要求的充要条件为：

$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$

进一步可得：

$$x > -ln\left(\frac{1}{C}-1\right)$$

该操作得到满足阈值的 Grid Cell 索引及其对应最大值。最大值经过 Sigmoid 后，即为该 Grid Cell 的类别 score。

- **分类部分：GatherElements 和 ArgMax 操作**

利用 Threshold(TopK) 得到的索引，GatherElements 取出满足要求的 Grid Cell，ArgMax 判断 80 个类别中最大值所在类别，从而得到该 Grid Cell 的类别。

- **Bounding Box 部分：GatherElements 操作**

利用 Threshold(TopK) 得到的 Grid Cell 索引，GatherElements 取出对应 bbox 信息，得到形状为 `1×64×k×1` 的 bbox 特征。

- **Bounding Box 部分：DFL：SoftMax + Conv 操作**

每个 Grid Cell 使用 4 个数描述 bbox 位置。DFL 结构会为某条边相对 anchor 位置的 offset 提供 16 个估计值。对这 16 个估计值执行 SoftMax，再通过卷积计算期望值。该结构是 Anchor Free 的核心设计，每个 Grid Cell 只负责预测一个 Bounding Box。以某条边的 offset 为例，其 16 个估计值为 $l_p$，其中 $p=0,1,...,15$，offset 的计算公式为：

$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

- **Bounding Box 部分：Decode：dist2bbox(ltrb2xyxy) 操作**

该操作将每个 Bounding Box 的 ltrb 描述解码为 xyxy 描述。ltrb 表示左、上、右、下边相对于 Grid Cell 中心点的距离。恢复相对位置为绝对位置并乘以对应特征层的下采样倍率后，即可得到 xyxy 坐标。

![](./imgs/ltrb2xyxy.jpg)

#### 实例分割 (Instance Segmentation)

![](./imgs/ultralytics_yolo_seg_dataflow.png)

实例分割基于目标检测流程扩展而来。在目标检测分支筛选出满足要求的 bbox 后，再取出对应的 mask coefficients，并与 proto 分支输出进行矩阵运算生成实例 mask。因此，检测部分的 ReduceMax、Threshold(TopK)、GatherElements、DFL 和 Decode 优化仍然适用。

#### 姿态估计 (Pose Estimation)

![](./imgs/ultralytics_yolo_pose_dataflow.png)

Ultralytics YOLO Pose 的关键点基于目标检测结果。COCO keypoint 定义如下：

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

Ultralytics YOLO Pose 的目标检测部分与 Detect 模型一致，额外增加一个 Channel = 57 的特征图，对应 17 个关键点，每个关键点包含相对特征图下采样倍率的 x、y 坐标以及该点 score。

在目标检测部分确定某个位置的 Key Points 满足要求后，将其乘以对应感受野的下采样倍率，即可得到基于输入尺寸的关键点坐标。

### 1. 环境准备与模型训练

该操作在 x86 机器上完成。推荐使用支持 CUDA 的 GPU 机器进行模型训练和验证，确保 `torch.cuda.is_available()` 为 True。推荐系统为 Ubuntu 22.04，Python 版本为 3.10。

下载 `ultralytics/ultralytics` 仓库，并参考官方文档配置训练环境：

```bash
git clone https://github.com/ultralytics/ultralytics.git
```

模型训练请参考 Ultralytics 官方文档。源 `.pt` 权重应使用 `ultralytics/ultralytics` 仓库训练得到，也可以使用 Ultralytics 官方发布的预训练权重。训练过程中无需修改程序，也不要修改模型的 `forward` 方法。

Ultralytics 官方文档：

- Quick Start: [https://docs.ultralytics.com/quickstart/](https://docs.ultralytics.com/quickstart/)
- Model Training: [https://docs.ultralytics.com/modes/train/](https://docs.ultralytics.com/modes/train/)

### 2. 导出 ONNX

该操作在 x86 机器上完成，推荐 Ubuntu 22.04 和 Python 3.10 环境。

进入本地 Ultralytics 仓库，下载 Ultralytics 官方预训练权重，或者使用训练流程产出的 `.pt` 权重。以下以 YOLO11n-Detect 为例：

```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

在 Ultralytics YOLO 训练环境中，运行本目录下的 `export_monkey_patch.py` 导出模型。该步骤依赖 `ultralytics.YOLO`、PyTorch 和导出相关依赖，因此应在训练/导出环境中执行，不在板端执行。脚本使用 `ultralytics.YOLO` 类加载 YOLO `.pt` 模型，通过 monkey patch 在 PyTorch 层替换模型计算流，然后调用 `ultralytics.YOLO.export` 导出 ONNX。导出的 ONNX 模型会保存在 `.pt` 模型同级目录。

```bash
python3 export_monkey_patch.py --pt yolo11n.pt
```

### 3. 模型编译 (mapper)

模型编译请在 RDK X5 OpenExplorer 工具链环境中执行。可以使用 pip 安装的 `rdkx5-yolo-mapper`，也可以使用 OpenExplorer Docker 离线镜像。不在板端安装和运行编译工具链。

在 OpenExplorer 工具链环境中运行本目录下的 `mapper.py`。需要准备校准图片和 ONNX 模型；脚本会自动准备校准数据和编译 YAML 配置文件，转换完成的 `.bin` 模型会保存在 ONNX 模型同级目录或 `--output-dir` 指定目录。

```bash
cd samples/vision/ultralytics_yolo/conversion

python3 mapper.py \
  --onnx yolo11n.onnx \
  --cal-images ./cal_images \
  --output-dir ./output
```

这个脚本暴露了一些常见参数，默认值已经满足大多数需求。

```bash
$ python3 mapper.py -h
usage: mapper.py [-h] [--cal-images CAL_IMAGES] [--onnx ONNX] [--output-dir OUTPUT_DIR] [--quantized QUANTIZED] [--jobs JOBS] [--optimize-level OPTIMIZE_LEVEL]
                 [--cal-sample CAL_SAMPLE] [--cal-sample-num CAL_SAMPLE_NUM] [--save-cache SAVE_CACHE] [--cal CAL] [--ws WS]

options:
  -h, --help                        show this help message and exit
  --cal-images CAL_IMAGES           *.jpg, *.png calibration images path, 20 ~ 50 pictures is OK.
  --onnx ONNX                       origin float onnx model path.
  --output-dir OUTPUT_DIR           output directory for converted model.
  --quantized QUANTIZED             int8 first / int16 first
  --jobs JOBS                       model combine jobs.
  --optimize-level OPTIMIZE_LEVEL   O0, O1, O2, O3
  --cal-sample CAL_SAMPLE           sample calibration data or not.
  --cal-sample-num CAL_SAMPLE_NUM   num of sample calibration data.
  --save-cache SAVE_CACHE           remove bpu output files or not.
  --cal CAL                         calibration_data_temporary_folder
  --ws WS                           temporary workspace
```

生成的 `.bin` 文件建议命名为：

- `*_bayese_*_nv12.bin`

模型文件需放入 sample 的 `model/` 目录，供 `runtime/python/run.sh` 和 `runtime/python/main.py` 使用。

## 输入输出协议

### 输入协议

Ultralytics YOLO runtime 使用 NV12 输入。转换侧生成的模型必须保持 NV12 runtime 输入协议，并与训练侧 RGB/NCHW 输入、`data_scale=1/255` 的编译配置保持一致。

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
| YOLOv9 | 支持 | 支持 | 不支持 | 不支持 |
| YOLOv10 | 支持 | 不支持 | 不支持 | 不支持 |
| YOLO11 | 支持 | 支持 | 支持 | 支持 |
| YOLO12 | 支持 | 不支持 | 不支持 | 不支持 |
| YOLO13 | 支持 | 不支持 | 不支持 | 不支持 |

具体后处理实现见 `runtime/python/ultralytics_yolo_*.py`。

## 编译结果检查

```bash
hb_model_info ./output/yolo11n_bayese_640x640_nv12.bin
hrt_model_exec model_info --model_file ./output/yolo11n_bayese_640x640_nv12.bin
hrt_model_exec perf --model_file ./output/yolo11n_bayese_640x640_nv12.bin --thread_num 1
```

## 常见问题

- **权限问题**：宿主机复制回文件时出现权限错误，可检查文件属主或使用 `sudo chown -R`。
- **内存/IPC 报错**：启动 Docker 容器时请添加 `--shm-size=15g`。
- **优化等级报错**：如果当前模型或工具链不支持 `O3`，可尝试使用 `O0`、`O1` 或 `O2`。

## License

本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
