# HGNetV2 模型转换与编译指南

[English](./README.md) | 简体中文

本目录提供了将 HGNetV2 模型转换为适配地瓜机器人（D-Robotics）`RDK X5` 硬件的 BPU 量化模型（`.bin`）的工具与说明。共支持 **b0、b1、b2、b3、b4** 五个变种。

## 模型编译环境

为了转换模型，您需要安装 **RDK X5 OpenExplore 工具链**。

### Docker 安装

**RDK X5 OpenExplore 1.2.8**
```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```
或者前往地瓜开发者社区获取离线版本的 Docker 镜像: [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)

**启动容器**（将 model zoo 挂载进容器以共享工作目录）:
```bash
docker run -it --rm \
  -v /path/to/rdk_model_zoo:/data \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

### ONNX 导出的 Python 依赖

export 脚本使用 `timm` 库加载 PP-HGNetV2 预训练权重。在容器内（或任意带 PyTorch ≥ 1.13 的 Python 3 环境）执行：

```bash
pip install timm
```

首次运行 `export_hgnetv2_b*_bpu.py` 时会从 Hugging Face 下载权重。若网络受限,可在执行前设置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 转换流程

### 1. PyTorch (timm) → ONNX

对 `b0 b1 b2 b3 b4` 中的每个 `${VARIANT}`,执行：

```bash
cd onnx_export
python3 export_hgnetv2_${VARIANT}_bpu.py
```

每个变种会在 `onnx_export/` 下产出 `hgnetv2_${VARIANT}.onnx`。脚本除了 timm 模型 id 不同,逻辑完全一致。

### 2. 准备校准数据

`hb_mapper` 需要 20–50 张代表性的 ImageNet 风格图片做 INT8 量化校准。yaml 中 `cal_data_dir: '../cal_data'`,所以请在 `conversion/` 同级建立该目录：

```bash
mkdir -p ../cal_data
# 拷贝 20–50 张 ImageNet val 风格的 JPEG 图片到该目录
```

### 3. ONNX → BIN

对每个变种,从本目录执行：

```bash
hb_mapper makertbin --model-type onnx --config hgnetv2_${VARIANT}.yaml
```

产物 `hgnetv2_${VARIANT}_224x224_nv12.bin` 会写到 `hgnetv2_${VARIANT}_224x224_nv12/` 子目录,把它拷贝或软链到 `../model/`,运行时即可直接使用：

```bash
cp hgnetv2_${VARIANT}_224x224_nv12/hgnetv2_${VARIANT}_224x224_nv12.bin ../model/
```

---

## 支持的变种

| 变种 | timm 模型 id | 输出 `.bin` |
| --- | --- | --- |
| b0 | `hgnetv2_b0.ssld_stage2_ft_in1k` | `hgnetv2_b0_224x224_nv12.bin` |
| b1 | `hgnetv2_b1.ssld_stage2_ft_in1k` | `hgnetv2_b1_224x224_nv12.bin` |
| b2 | `hgnetv2_b2.ssld_stage2_ft_in1k` | `hgnetv2_b2_224x224_nv12.bin` |
| b3 | `hgnetv2_b3.ssld_stage2_ft_in1k` | `hgnetv2_b3_224x224_nv12.bin` |
| b4 | `hgnetv2_b4.ssld_stage2_ft_in1k` | `hgnetv2_b4_224x224_nv12.bin` |

---

## License
本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
