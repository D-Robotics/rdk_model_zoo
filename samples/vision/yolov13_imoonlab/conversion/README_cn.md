[English](./README.md) | 简体中文

# YOLOv13 iMoonLab 模型转换与编译指南

本目录提供 YOLOv13 Detect 模型在 RDK S100 平台上的 ONNX 导出、校准数据准备、HBM 编译和结果检查说明，并附带参考 YAML 和编译日志。

## 目录结构

```bash
.
├── config_yolov13_detect_nv12.yaml
├── hb_compile_yolov13.txt
├── hb_model_info_yolov13.txt
├── hrt_model_exec_model_info_yolov13.txt
├── README.md
└── README_cn.md
```

## 编译环境

模型转换请在 x86 Linux 主机的 OpenExplore 环境中完成，不建议在板端安装编译工具链。

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

### 1. 安装 Docker

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. 获取并加载离线镜像

请访问 OE 资源入口，下载适配 RDK S100 系列的 CPU 版本 Docker 镜像。

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
```

### 3. 启动容器

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

## 转换流程

### 1. 准备训练环境与权重

YOLOv13 的 ONNX 导出需要在 iMoonLab/Ultralytics 训练环境中完成，源 `.pt` 权重应来自官方仓库训练流程或官方发布的预训练权重。

```bash
git clone https://github.com/iMoonLab/yolov13.git
cd yolov13
wget https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt
```

训练请参考 Ultralytics 官方文档：

- <https://docs.ultralytics.com/modes/train/>

训练阶段无需修改程序，也无需修改 `forward`。

### 2. 导出 ONNX

建议先卸载环境中通过 `pip` 或 `conda` 安装的 `ultralytics` 命令行包，确保你修改的是实际生效的源码目录。

```bash
conda list | grep ultralytics
pip list | grep ultralytics
conda uninstall ultralytics
pip uninstall ultralytics
```

如需确认当前环境加载的 `ultralytics` 路径，可执行：

```python
import ultralytics
print(ultralytics.__path__)
```

然后修改 `ultralytics/nn/modules/head.py` 中 `Detect` 类的 `forward`，把三个特征层的分类输出和框输出拆开，形成 6 个输出头：

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

如果导出的输出顺序与参考模型相反，可交换 `cv2` 与 `cv3` 的追加顺序后重新导出：

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

完成修改后执行导出：

```python
from ultralytics import YOLO
YOLO('yolov13n.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

如果遇到 `No module named onnxsim`，安装对应依赖即可。若导出的 ONNX IR 版本过高，可以继续使用 `simplify=False`。

### 3. 准备校准数据

请准备 20 到 50 张覆盖目标场景的图片作为 PTQ 校准输入。也可以参考 OE 开发包中的相关示例生成校准数据。

## 转换参考

ONNX 导出
PTQ 配置生成

### 4. 确认移除反量化节点名称

使用 Netron 打开导出的 ONNX：

- <https://netron.app/>

查看大小为 `[1, 80, 80, 64]`、`[1, 40, 40, 64]`、`[1, 20, 20, 64]` 的三个输出名称，并将它们对应地填写到 YAML 的 `remove_node_name` 中。一个常用经验是优先关注名称中对应 `64 = 4 * REG` 的 Dequantize 节点，但不同版本导出的节点名可能不同，不能直接硬套。

![Netron example](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/test_data/netron_conv_example.jpeg)

参考 YAML 片段如下：

```yaml
model_parameters:
  onnx_model: 'ultralytcs_YOLO.onnx'
  march: nash-e
  layer_out_dump: False
  working_dir: 'ultralytcs_YOLO_output'
  output_model_file_prefix: 'ultralytcs_YOLO'
  remove_node_name: "/model.32/cv2.0/cv2.2.2/Conv;/model.32/cv2.1/cv2.1.2/Conv;/model.32/cv2.2/cv2.2.2/Conv;"
```

### 5. 编译 HBM

```bash
hb_compile --config config_yolov13_detect_nv12.yaml
```

当前目录中提供了以下参考日志，便于对照自己的模型导出与编译结果：

- [hb_compile_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hb_compile_yolov13.txt)
- [hb_model_info_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hb_model_info_yolov13.txt)
- [hrt_model_exec_model_info_yolov13.txt](/D:/20_Dev_Projects/21_RDK_MODEL_ZOO/rdk_model_zooo_mccc/rdk_mode_zoo_mc_rdks/samples/vision/yolov13_imoonlab/conversion/hrt_model_exec_model_info_yolov13.txt)

## 异常处理

如果你自己编译出的模型输出顺序与参考模型不一致，通常是 `remove_node_name` 设置错误。可以通过快速生成 `bc` 模型并检查可移除节点信息：

```bash
hb_compile --fast-perf --march nash-e --skip compile --model yolov13n.onnx
hb_model_info yolov13n_quantized_model.bc
```

典型输出示例：

```bash
2025-06-24 03:17:30,044 INFO ############# Removable node info #############
2025-06-24 03:17:30,044 INFO Node Name                    Node Type
2025-06-24 03:17:30,045 INFO ---------------------------- ----------
2025-06-24 03:17:30,045 INFO /model.32/cv3.0/cv3.0.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.0/cv2.0.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv3.1/cv3.1.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.1/cv2.1.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv3.2/cv3.2.2/Conv Dequantize
2025-06-24 03:17:30,045 INFO /model.32/cv2.2/cv2.2.2/Conv Dequantize
```

## License

本目录内容遵循仓库顶层 `LICENSE`。
