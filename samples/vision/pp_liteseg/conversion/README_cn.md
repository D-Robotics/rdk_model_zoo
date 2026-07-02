# PP-LiteSeg-STDC1 模型转换

本目录提供从 PaddleSeg / ONNX 到 RDK X5 `.bin` 模型的完整转换流程。

## 环境

需要两个环境：

1. **ONNX 导出环境**：安装 PaddlePaddle、PaddleSeg、`paddle2onnx`、`onnx`、`onnxsim` 的 Python 环境。
2. **OpenExplorer（OE）环境**：D-Robotics OpenExplorer v1.2.8 Docker 环境，提供 `hb_mapper`、`hb_perf` 等工具。

所有 `hb_mapper` 和 `hb_perf` 命令都必须在 OpenExplorer Docker 容器内执行。

### 1.1 ONNX 导出环境安装

在本地 Python 环境（推荐 Python 3.8–3.10）中安装依赖：

```bash
pip install paddlepaddle==3.0.0 paddle2onnx onnx onnxsim

# 从 Gitee 镜像安装 PaddleSeg（部分地区 GitHub 访问受限）
git clone --depth=1 https://gitee.com/paddlepaddle/PaddleSeg.git
cd PaddleSeg && pip install -e .
```

### 1.2 OpenExplorer Docker 安装

**方式 A — Docker Hub（无需登录）：**

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

**方式 B — 离线 tar.gz（无法访问网络时使用）：**

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

**可选 — OE SDK 与文档：**

```bash
# 完整 SDK 包
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz

# 中文文档
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip
# 英文文档
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip
```

### 1.3 启动 OE Docker 容器

将仓库根目录挂载进容器并进入交互式 shell：

```bash
docker run -it --rm \
  -v $(pwd):/open_explorer \
  -w /open_explorer \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
  /bin/bash
```

在容器内验证工具链：

```bash
hb_mapper --version
hb_perf --version
```

## 1. 导出 ONNX

在本目录执行导出脚本：

```bash
cd samples/vision/pp_liteseg/conversion
bash onnx_export/export_pp_liteseg_stdc1_onnx.sh
```

默认脚本导出 PaddleSeg 配置：

```text
configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml
```

预期 ONNX 输出：

```text
conversion/onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
```

如果你已经有 ONNX 模型，可以直接放到上述路径，或修改 `ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml` 中的 `model_parameters.onnx_model`。

## 2. 准备校准数据

准备 20 到 50 张有代表性的道路场景图片。脚本会导出 NCHW RGB float32 raw tensor，不在脚本中做归一化；归一化由 YAML 完成。

```bash
cd samples/vision/pp_liteseg/conversion
python3 prepare_calibration.py \
  --src /path/to/cityscapes_or_custom_images \
  --out calibration_data_rgb_f32_1024x512 \
  --width 1024 \
  --height 512 \
  --num 50
```

每个校准文件大小应为：

```text
1 * 3 * 512 * 1024 * 4 = 6291456 bytes
```

## 3. 检查 ONNX 算子

在 OpenExplorer Docker 容器内执行：

```bash
cd samples/vision/pp_liteseg/conversion
hb_mapper checker \
  --model-type onnx \
  --march bayes-e \
  --model onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
```

查看 `hb_mapper_checker.log`，确认没有 unsupported operators。如果出现不支持算子，优先简化 ONNX 图，避免把后处理节点导入模型。

## 4. 编译 BIN

```bash
cd samples/vision/pp_liteseg/conversion
hb_mapper makertbin \
  --config ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml \
  --model-type onnx
```

预期输出：

```text
conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```

## 5. 一键编译

ONNX 已存在后，可以执行：

```bash
cd samples/vision/pp_liteseg/conversion
CAL_SRC=/path/to/calibration/images bash build_bin.sh
```

如果校准数据已经准备好，可以省略 `CAL_SRC`：

```bash
bash build_bin.sh
```

## 6. 性能检查

```bash
hb_perf ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```

板端验证：

```bash
hrt_model_exec model_info \
  --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin

hrt_model_exec perf \
  --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
  --core_id=0 \
  --thread_num=1 \
  --profile_path="."
```

## 运行协议

- 运行时输入类型：`nv12`
- 训练输入类型：`rgb`
- 训练输入布局：`NCHW`
- 输入尺寸：`1024x512`
- 归一化：`(pixel - mean) * scale`
- Mean：`123.675, 116.28, 103.53`
- Scale：`1/58.395, 1/57.12, 1/57.375`
- 输出：语义分割 logits，通常在类别维度执行 `argmax` 解码

## 常见问题

- 如果 `checker` 报 resize 或 argmax 不支持，确认 ONNX 只包含神经网络本体，不包含后处理。
- 如果 cosine similarity 较低，可以尝试 `calibration_type: mix` 或准备更有代表性的校准图片。
- 如果输出 shape 不符合预期，使用 Netron 检查 ONNX 并同步调整运行时后处理。
