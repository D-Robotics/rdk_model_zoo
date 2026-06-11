[English](./README.md) | 简体中文

# YOLO26 模型转换与编译指南

本目录提供 YOLO26 模型导出、量化编译和 HBM 产物检查所需的脚本和说明，目标产物为适配 RDK S100/S100P 的 BPU 量化 `.hbm` 模型。

## 目录结构

```text
.
├── mapper.py                       # 调用 OpenExplore 编译工具生成 HBM
├── onnx_export/                    # YOLO26 各任务 ONNX 导出脚本
│   ├── export_yolo26_cls_bpu.py
│   ├── export_yolo26_detect_bpu.py
│   ├── export_yolo26_obb_bpu.py
│   ├── export_yolo26_pose_bpu.py
│   └── export_yolo26_seg_bpu.py
├── README.md
└── README_cn.md
```

## 模型编译环境

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore Docker 环境中完成，不建议在板端安装编译工具链。

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

请访问 [D-Robotics 开发者文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview#docker-%E9%95%9C%E5%83%8F) 下载适配 RDK S100 系列的 CPU 版本 Docker镜像。

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

### 1. 环境准备与模型训练

注：模型训练与 ONNX 导出操作应在具有硬件加速（如 NVIDIA GPU）的 x86 机器上进行。推荐使用 Ubuntu 22.04, Python 3.10 环境。

1. **配置环境**：
   YOLO26 基于 `ultralytics` 框架。请参考 [Ultralytics 官方文档](https://docs.ultralytics.com/quickstart/) 配置训练环境。
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   pip install -e .
   ```

2. **模型训练**：
   参考 [Ultralytics 训练指南](https://docs.ultralytics.com/modes/train/) 完成模型训练。训练过程中无需针对 BPU 修改模型结构。

3. **获取权重**：
   使用训练得到的 `.pt` 权重文件。

### 2. 导出 ONNX

**重要提示**：ONNX 导出必须在安装了 `ultralytics` 及其所有依赖的训练/开发环境中进行，**严禁在板端执行此步骤**。

根据任务类型选择 `onnx_export/` 下的导出脚本，生成与 BPU 后处理协议匹配的 ONNX：

```bash
# 目标检测
python3 onnx_export/export_yolo26_detect_bpu.py --weights yolo26n.pt --imgsz 640
# 实例分割
python3 onnx_export/export_yolo26_seg_bpu.py --weights yolo26n-seg.pt --imgsz 640
# 姿态估计
python3 onnx_export/export_yolo26_pose_bpu.py --weights yolo26n-pose.pt --imgsz 640
# 旋转框检测 (OBB)
python3 onnx_export/export_yolo26_obb_bpu.py --weights yolo26n-obb.pt --imgsz 640
# 图像分类
python3 onnx_export/export_yolo26_cls_bpu.py --weights yolo26n-cls.pt --imgsz 224
```

导出的 ONNX 模型会自动包含适配 BPU 的后处理转换逻辑（如 DFL 节点提取等）。导出的 ONNX 模型会保存在 `.pt` 模型同级目录下。

### 3. 准备校准数据

准备 20 到 50 张覆盖目标场景的 `.jpg` 或 `.png` 图片（建议使用训练集的子集）：

```text
cal_images/
├── 000001.jpg
├── 000002.jpg
└── ...
```

### 4. 编译 HBM 模型 (mapper)

进入当前目录后运行 `mapper.py`。`--march` 用于选择目标架构：

```bash
cd samples/vision/ultralytics_yolo26/conversion

# RDK S100 (Nash-E)
python3 mapper.py --onnx yolo26n_detect.onnx --cal-images ./cal_images --march nash-e

# RDK S100P (Nash-M)
python3 mapper.py --onnx yolo26n_detect.onnx --cal-images ./cal_images --march nash-m
```

生成的 `.hbm` 文件建议命名为：

- S100: `*_nashe_*_nv12.hbm`
- S100P: `*_nashm_*_nv12.hbm`

模型文件需放入 sample 的 `model/nash-e/` 或 `model/nash-m/` 目录，供 `runtime/python/run.sh` 和 `runtime/python/main.py` 使用。

### 5. 脚本参数说明

```bash
python3 mapper.py -h
```

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--onnx` | 浮点 ONNX 模型路径。 | `./yolo11n.onnx` |
| `--output-dir` | 转换后模型输出目录。 | `.` |
| `--cal-images` | 校准图片目录。 | `./cal_images` |
| `--march` | 目标架构：`nash-e` 为 RDK S100，`nash-m` 为 RDK S100P。 | `nash-e` |
| `--quantized` | 量化精度：`int8` 或 `int16`。 | `int8` |
| `--jobs` | 编译并发任务数。 | `16` |
| `--optimize-level` | 编译优化等级，Nash 架构支持 `O0` 到 `O2`。 | `O2` |
| `--cal-sample` | 是否从校准目录采样图片。 | `True` |
| `--cal-sample-num` | 采样图片数量。 | `20` |
| `--save-cache` | 是否保留临时工作目录。 | `False` |

## 输入输出协议

### 输入协议

YOLO26 runtime 使用 NV12 输入，并固定为两个输入 tensor：

- `input[0]`: Y plane
- `input[1]`: UV plane

转换侧生成的模型必须保持该输入协议。

### 输出协议

Python runtime 按固定索引解析输出，不做输出顺序猜测：

- Detection: `[cls, box] * 3`
- Segmentation: `[cls, box, mask_coeff] * 3 + proto`
- Pose: `[cls, box, keypoints] * 3`
- OBB: `[cls, box, angle] * 3`
- Classification: 单个分类输出 tensor

具体后处理实现见 `runtime/python/yolo26_*.py`。

## 编译结果检查

```bash
hrt_model_exec model_info --model_file yolo26n_detect_nashm_640x640_nv12.hbm
hrt_model_exec perf --model_file yolo26n_detect_nashm_640x640_nv12.hbm --thread_num 1
```

## 常见问题

- **权限问题**：宿主机复制回文件时出现权限错误，可检查文件属主或使用 `sudo chown -R`。
- **内存/IPC 报错**：启动 Docker 容器时请添加 `--shm-size=15g`。
- **优化等级报错**：Nash 架构不支持 `O3` 时，请使用 `O0`、`O1` 或 `O2`。

## License

本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。
