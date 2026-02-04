# YOLO26 模型转换与编译指南

[English](./README.md) | 简体中文

本目录提供了将 YOLO26 模型（源自 Ultralytics 框架）转换为适配地瓜机器人（D-Robotics）RDK 硬件的 BPU 量化模型（`.bin`）的工具与说明。

## 模型编译环境

为了转换模型，您需要安装 **RDK X5 OpenExplore 工具链**。我们提供两种安装方式，**推荐使用方式一**。

### 方式一：Pip 安装 (推荐)

此方式在 x86 Linux 机器上安装经过裁剪的轻量级工具链，建议配合 Miniconda 使用。

**注意**: 此操作仅在 x86 开发机（推荐 Ubuntu 22.04）上进行，**切勿**在 RDK 板端安装。

1.  **创建 Python 环境 (Miniconda)**
    强烈建议使用虚拟环境以避免依赖冲突。
    ```bash
    # 创建名为 rdk_env 的 Python 3.10 环境
    conda create -n rdk_env python=3.10 -y
    
    # 激活环境
    conda activate rdk_env
    ```

2.  **安装工具链**
    ```bash
    pip install rdkx5-yolo-mapper
    ```
    *(可选) 如果下载速度慢，请使用阿里云镜像:*
    ```bash
    pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    ```

3.  **验证安装**
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

## 转换流程

### 1. 一键转换脚本 (推荐)

我们提供了 `mapper.py` 脚本，可以自动完成校准数据准备、配置文件生成以及调用 `hb_mapper` 进行编译的全过程。

**准备工作**:
- 已经导出为 BPU 适配的 ONNX 模型（参考 `onnx_export/`）。
- 准备一个文件夹，包含 20~50 张用于量化校准的图片（`.jpg` 或 `.png`）。

**运行转换**:
```bash
python3 mapper.py --onnx [model.onnx] --cal-images [校准图片目录]
```
转换成功后，生成的 `.bin` 模型文件将位于 ONNX 模型的同级目录下。

### 2. 脚本参数说明

`mapper.py` 暴露了一些常用参数以满足定制需求：

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

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。