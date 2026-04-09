[English](./README.md) | [简体中文](./README_cn.md)

# 模型转换

本文档介绍如何将训练模型转换为 D-Robotics BPU 可运行的 BIN 模型文件。

---

## 目录结构

```

```conversion/
├── README.md              # 英文
├── README_cn.md           # 中文（本文件）
└── ptq_yamls/             # PTQ 配置文件
    ├── paddleocr_det_config.yaml  # 检测模型配置文件
    └── paddleocr_rec_config.yaml  # 识别模型配置文件

---

## 转换流程

```
浮点预训练模型 → ONNX导出 → 校准数据集准备 → PTQ量化 → 编译为BIN
```

---

## 编译环境

为了转换模型，您需要安装 **RDK X5 OpenExplore 工具链**。我们提供两种安装方式，**推荐使用方式一**。

### 方式一：Pip 安装 (推荐)

此方式在 x86 Linux 机器上安装经过裁剪的轻量级工具链，建议配合 Miniconda 使用。

**注意**: 此操作仅在 x86 开发机（推荐 Ubuntu 22.04）上进行，**切勿**在 RDK 板端安装。

1. **创建 Python 环境 (Miniconda)**
   
   强烈建议使用虚拟环境以避免依赖冲突。
   ```bash
   # 创建名为 rdk_env 的 Python 3.10 环境
   conda create -n rdk_env python=3.10 -y
   
   # 激活环境
   conda activate rdk_env
   ```

2. **安装工具链**
   
   ```bash
   pip install rdkx5-yolo-mapper
   ```
   
   *(可选) 如果下载速度慢，请使用阿里云镜像:*
   ```bash
   pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
   ```

3. **验证安装**
   
   ```bash
   hb_mapper --version
   # 预期输出: hb_mapper, version 1.24.3 (或更新版本)
   ```

**常见问题**: 如果出现 `incomplete-download` 或下载失败的错误，通常是网络连接不稳定导致的。重新运行安装命令即可，Pip 会自动跳过已下载的包。

---

### 方式二：Docker 安装 (备选)

如果您希望环境完全隔离，或者方式一遇到依赖问题，可以使用官方 Docker 镜像。

**RDK X5 OpenExplore 1.2.8**
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```
或者前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

**启动容器**:
```bash
# 挂载您的 model zoo 目录到容器中
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

---


## 一键转换脚本 (推荐)

我们提供了 `mapper.py` 脚本，可以自动完成校准数据准备、配置文件生成以及调用 `hb_mapper` 进行编译的全过程。

### 准备工作

- 已经导出为 BPU 适配的 ONNX 模型。
- 准备一个文件夹，包含 20~50 张用于量化校准的图片（`.jpg` 或 `.png`）。

### 运行转换

```bash
python3 mapper.py --onnx model.onnx --cal-images /path/to/calibration/images
```

转换成功后，生成的 `.bin` 模型文件将位于 ONNX 模型的同级目录下。

### 脚本参数说明

```bash
python3 mapper.py -h
```

| 参数 | 说明 |
| :--- | :--- |
| `--onnx` | 原始浮点 ONNX 模型的路径。 |
| `--cal-images` | 包含校准图片的目录路径（建议 20~50 张）。 |
| `--quantized` | 量化精度：`int8`（默认，推荐）或 `int16`。 |
| `--jobs` | 模型编译时的并发任务数。 |
| `--optimize-level` | 编译器优化等级：`O0`, `O1`, `O2` (默认), `O3`。 |
| `--cal-sample` | 是否从目录中采样图片（默认：True）。 |
| `--save-cache` | 是否保留 BPU 编译过程中的临时文件（默认：False）。 |

---

## 生成校准数据集

模型量化需要校准数据集来统计激活值分布。

### 数据集要求

- 数量：建议 20-50 张图片
- 来源：从训练数据集中选取典型样本
- 格式：RGB 图像，JPEG/PNG 格式
- 尺寸：与模型输入分辨率一致（如 640x640）

### 准备步骤

```bash
# 1. 创建校准数据目录
mkdir -p calibration_data_rgb_f32_640

# 2. 将图片放入目录（每张图片需 resize 到模型输入尺寸）
#    图片命名格式：0001.jpg, 0002.jpg, ...

# 3. 验证数据
ls calibration_data_rgb_f32_640 | wc -l
```

> **注意**：请勿使用验证集/测试集图片作为校准数据，应从训练集中随机选取。

---

## 模型检查

在编译前验证模型算子是否被 BPU 支持：

```bash
hb_mapper checker --model-type onnx --march bayes-e --model model.onnx
```

### 常见问题处理

- **算子不支持**：部分算子会 fallback 到 CPU 计算，可能影响性能
- **精度问题**：检查 cosine similarity 是否 ≥ 0.999

---

## 模型编译

### 使用配置文件编译

```bash
hb_mapper makertbin --model-type onnx --config <config>.yaml
```

---

## 输出产物

编译完成后，在配置的 `build_dir` 目录下生成以下文件：

```
output/
├── model_name.bin           # 可运行的 BIN 模型文件
└── hb_mapper_makertbin.log  # 转换日志
```

### 日志关键信息

日志中包含以下重要信息，请务必保存：
- 输入输出 tensor 的名称和形状
- 量化参数（scale/zero_point）
- 算子分布统计（BPU/CPU）

---

## 验证方法

### 使用 hb_perf 可视化

```bash
hb_perf model_name.bin
```

生成性能分析报告，包括算子分布、内存占用等信息。

### 使用 hrt_model_exec 查看模型信息

```bash
hrt_model_exec model_info --model_file model_name.bin
```

输出示例：
```
input[0]:
  name: images
  valid shape: (1,3,640,640,)
  tensor type: HB_DNN_IMG_TYPE_NV12

output[0]:
  name: output0
  valid shape: (1,80,80,255,)
  tensor type: HB_DNN_TENSOR_TYPE_F32
```

---

## 相关文档

- [RDK X5 算法工具链手册](https://developer.d-robotics.cc)
- [OpenExplorer 工具链下载](https://developer.d-robotics.cc)
- [地瓜开发者社区](https://forum.d-robotics.cc)

---

## License

本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。