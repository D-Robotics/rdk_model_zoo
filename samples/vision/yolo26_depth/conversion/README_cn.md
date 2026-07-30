[English](./README.md) | [简体中文](./README_cn.md)

# YOLO26 Depth 模型转换与编译指南

本目录提供 Ultralytics YOLO26 Depth 模型导出、量化编译和 RDK X5 BPU 部署 `.bin` 产物检查所需的脚本、资源和说明。

## 目录结构

```text
.
├── export.py                  # Ultralytics YOLO26 Depth ONNX 导出脚本
├── extract_sunrgbd_subset.py  # 确定性 SUN RGB-D 子集提取工具
├── prepare_calibration.py     # 校准 tensor 准备工具
├── mapper.py                  # 准备 Mapper YAML 并调用 X5 OpenExplorer 编译器
├── ptq_yamls/                 # N/S/M/L/X 规格的参考 PTQ YAML
├── requirements.txt           # 导出和校准工具所需 Python 依赖
├── README.md
└── README_cn.md
```

## 编译环境

请在 x86 Linux 主机和对应的 RDK X5 OpenExplorer 环境中执行模型编译。不建议在 RDK X5 板端安装或运行编译工具链。

工具链入口：

- RDK X5 OpenExplorer / 算法工具链文档：<https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/overview>
- OE 工具链下载与手册：<https://toolchain.d-robotics.cc/>

### 1. 安装 Docker

按照 Docker 官方文档安装 Docker，并验证安装结果：

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. 下载并加载 X5 工具链离线镜像

使用与预期编译器版本匹配的 RDK X5 OpenExplorer 离线镜像。本样例的验证数据由 OpenExplorer v1.2.8 / Mapper 1.24.3 生成。

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
```

### 3. 启动容器

将仓库和外部工作目录挂载到容器中。建议增加 shared memory，减少大模型规格编译失败概率。

```bash
# 假设当前目录是 rdk_model_zoo_mc_rdkx5 仓库根目录
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  -v /path/to/external/work:/work \
  --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

如果镜像名或 tag 与示例不同，请使用 `sudo docker images` 中显示的实际名称。

### 4. 检查工具链和 Python 依赖

```bash
hb_mapper --version
hb_model_info --help
python3 -m pip install -r samples/vision/yolo26_depth/conversion/requirements.txt
```

## 转换流程

### 高性能计算流程说明

#### 单目深度估计

YOLO26 Depth 根据单张 RGB 图片预测稠密相对深度图。原始浮点模型中包含适合训练的任务解码和细化逻辑，但这些逻辑并不都适合直接放到板端部署图中。

部署时，转换流程将 BPU 图聚焦在计算量较大的 convolutional backbone、neck 和 depth head 上。导出的图在 `768x768` 输入下输出 `1x192x192x1` 的低分辨率 calibrated log-depth tensor。Runtime 再在 CPU 上执行轻量后处理：

- 对 calibrated log-depth 执行 `exp(log_depth)`；
- 将 `192x192` 深度图 resize 回 padded model input 尺寸；
- 去除 114-value letterbox padding 并恢复到原图几何；
- 对恢复后的相对深度进行可视化或序列化。

这种划分可以保持编译模型简洁，稳定 runtime tensor 协议，并避免将生成的中间产物提交到仓库。Mapper 配置还将尾部 depth convolution 输出保留为 `int16`，以降低 log-depth 图的量化损失。

### 1. 环境准备和模型训练

该操作在 x86 机器上完成，推荐 Ubuntu 22.04 和 Python 3.10 环境。训练或验证自定义权重时可以使用 GPU 环境，模型编译则由 X5 工具链完成。

下载 `ultralytics/ultralytics` 仓库，并按照 Ultralytics 官方文档配置训练/导出环境。

```bash
git clone https://github.com/ultralytics/ultralytics.git
```

模型训练请参考 Ultralytics Depth 官方文档。源 `.pt` 权重应由 `ultralytics/ultralytics` 仓库训练得到，也可以使用兼容的 YOLO26 Depth 预训练权重。训练阶段不需要修改程序，也不要在训练仓库中修改模型 `forward` 方法。

Ultralytics 文档：

- Quick Start：<https://docs.ultralytics.com/quickstart/>
- Model Training：<https://docs.ultralytics.com/modes/train/>
- Depth Task：<https://docs.ultralytics.com/tasks/depth/>

### 2. 导出 ONNX

该操作在 x86 机器和 Ultralytics 训练/导出环境中完成。准备 YOLO26 Depth `.pt` 文件后，运行本目录下的 `export.py`。

`export.py` 使用 `ultralytics.YOLO` 加载 `.pt` 模型，执行适配 X5 导出的 Python-side patch，并调用 `ultralytics.YOLO.export`。导出的 ONNX 和 `export-report.json` 会写入外部输出目录。

```bash
cd samples/vision/yolo26_depth/conversion

python3 export.py \
  --weights /work/weights/yolo26n-depth.pt \
  --variant n \
  --imgsz 768 \
  --opset 11 \
  --output-dir /work/yolo26_depth/export_n
```

导出脚本支持 `n`、`s`、`m`、`l` 和 `x` 五种规格。每个规格都需要使用匹配的权重文件和 `--variant` 参数。

### 3. 准备校准数据

已验证配置使用 SUN RGB-D train split 中固定选取的 100 张图片。校准 tensor 为 RGB CHW uint8，并使用与 runtime 预处理一致的 114-value letterbox 策略。Mapper 在编译时执行 `data_scale=1/255`。

提取确定性的 SUN RGB-D 子集：

```bash
python3 extract_sunrgbd_subset.py \
  --archive /work/datasets/SUNRGBD.zip \
  --split train \
  --count 100 \
  --seed 20260725 \
  --output /work/yolo26_depth/sunrgbd_train100
```

打包校准二进制：

```bash
python3 prepare_calibration.py \
  --images /work/yolo26_depth/sunrgbd_train100/images \
  --output /work/yolo26_depth/calibration_768 \
  --count 100 \
  --seed 20260725 \
  --size 768 \
  --manifest /work/yolo26_depth/calibration_768_manifest.json \
  --report /work/yolo26_depth/calibration_768_report.md
```

如果无法使用 SUN RGB-D，也可以使用等价的代表性 RGB 图片；但在对比结果时，应保持相同的预处理和校准数量策略。

### 4. 模型编译

在 RDK X5 OpenExplorer 工具链环境中执行模型编译。运行 `mapper.py` 前需要准备导出的 ONNX 模型和已打包的校准目录。

`mapper.py` 会生成 YAML 配置，执行 `hb_mapper checker`、`hb_mapper makertbin`，将最终 `.bin` 和量化 ONNX 复制到 `artifacts/`，并在 `reports/` 下写出日志和 `compile-report.json`。

```bash
cd samples/vision/yolo26_depth/conversion

python3 mapper.py \
  --onnx /work/yolo26_depth/export_n/yolo26n-depth-log.onnx \
  --variant n \
  --calibration /work/yolo26_depth/calibration_768 \
  --size 768 \
  --optimize-level O3 \
  --output /work/yolo26_depth/compile_n
```

这个脚本暴露了一些常见参数，默认值覆盖已验证配置。

```bash
$ python3 mapper.py -h
usage: mapper.py [-h] --onnx ONNX --variant {n,s,m,l,x} --calibration CALIBRATION --output OUTPUT --size {768} [--jobs JOBS] [--optimize-level {O0,O1,O2,O3}]

options:
  -h, --help                        show this help message and exit
  --onnx ONNX                       exported floating-point ONNX model path
  --variant {n,s,m,l,x}             YOLO26 Depth model size
  --calibration CALIBRATION         packed calibration binary directory
  --output OUTPUT                   new external output directory
  --size {768}                      model input size; this sample validates 768 only
  --jobs JOBS                       Mapper compilation jobs, default: 16
  --optimize-level {O0,O1,O2,O3}    compiler optimization level, default: O3
```

推荐的 `.bin` 文件名：

- `yolo26n_depth_bayese_768x768_nv12.bin`
- `yolo26s_depth_bayese_768x768_nv12.bin`
- `yolo26m_depth_bayese_768x768_nv12.bin`
- `yolo26l_depth_bayese_768x768_nv12.bin`
- `yolo26x_depth_bayese_768x768_nv12.bin`

模型文件需放入 sample 的 `model/bayes-e/` 目录，供 `runtime/python/run.sh`、`runtime/cpp/run.sh` 和对应 `main` 程序使用。

## 输入输出协议

### 输入协议

Runtime 使用一个名为 `images` 的 NV12 pyramid 输入 tensor。

- Mapper 预处理前的训练/导出布局：`NCHW` RGB。
- Runtime 布局：`NHWC` NV12 pyramid。
- 已验证输入尺寸：`768x768`。
- 校准预处理：114-value letterbox padding 和 `data_scale=1/255`。

转换侧生成的模型必须保持该输入协议，否则 Python 和 C++ runtime 会在 shape 或 tensor type 检查阶段失败。

### 输出协议

Runtime 期望一个 dequantized float32 输出 tensor：

- 输出 shape：`1x192x192x1`；
- 语义：calibrated log-depth；
- 后处理：`depth = exp(log_depth)`、双线性插值、letterbox 还原；
- 最终输出：原图尺寸的稠密相对深度。

模型输出为相对深度，不是标定后的米制绝对深度。因此，数据集级精度评测在和米制真值比较前需要执行 scale 或 scale-shift 对齐。

## 编译结果检查

使用 `hb_model_info` 或 `hrt_model_exec` 检查生成的 `.bin` 模型。`../evaluator/README.md` 中的性能数据在 RDK X5 板端测得，仅统计模型执行时间。

```bash
hb_model_info /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec model_info --model_file /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec perf --model_file /work/yolo26_depth/compile_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin --thread_num 1
```

Mapper 成功执行后，会在仓库外部写出以下文件：

```text
compile_n/
├── artifacts/
│   ├── yolo26n_depth_bayese_768x768_nv12.bin
│   └── yolo26n_depth_bayese_768x768_nv12_quantized_model.onnx
├── config/
│   └── yolo26n_depth_bayese_768x768_nv12.yaml
├── reports/
│   ├── checker.log
│   ├── makertbin.log
│   ├── hb_model_info.log
│   └── compile-report.json
└── working/
```

`compile-report.json` 会记录模型 hash、输出余弦相似度、编译器延迟/FPS 估计、DDR 估计以及生成产物路径。

## 常见问题

- **权限问题**：从 Docker 复制回宿主机的文件属主异常时，可检查文件属主或对外部工作目录执行 `sudo chown -R`。
- **内存/IPC 报错**：启动 Docker 容器时请添加 `--shm-size=15g`。
- **优化等级不支持**：如果本地 X5 编译器包不支持当前图的 `O3`，可尝试 `O0`、`O1` 或 `O2`。
- **缺少输出余弦**：检查 `reports/makertbin.log`；当 Mapper 没有报告输出余弦时，不应发布该模型。
- **Runtime 几何不一致**：确认校准和 runtime 使用相同的 114-padding letterbox 策略。

## License

本目录下的工具遵循仓库顶层许可证。Ultralytics 模型和 SUN RGB-D 数据仍分别遵循其上游许可证。
