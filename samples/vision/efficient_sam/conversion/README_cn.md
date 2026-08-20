[English](./README.md) | 简体中文

# EfficientSAM 模型转换与编译指南

本目录提供 EfficientSAM 从上游 PyTorch 权重导出、为编码器与解码器分别准备校准数据、并编译得到适配 RDK S100 / S100P / S600 的两个 `.hbm` 模型所需的脚本与说明。

## 目录结构

```text
.
├── scripts/
│   ├── export_encoder_onnx.py                    # 导出 ViT-Tiny 图像编码器 ONNX
│   ├── export_decoder_onnx.py                    # 导出固定提示(点)掩码解码器 ONNX
│   ├── prepare_calibration.py                    # 编码器校准:图片 → RGB/255 NCHW npy
│   ├── dump_encoder_embedding.py                 # dump 一份真实编码器 embedding(.bin)
│   ├── prepare_efficient_decoder_calibration.py  # 解码器校准:embedding → featuremap npy
│   └── quantize.py                               # hb_compile 入口(每个 march 编译编码器+解码器)
├── configs/
│   ├── efficient_sam_encoder_{nashe,nashm,nashp}_config.yaml
│   └── efficient_sam_decoder_{nashe,nashm,nashp}_config.yaml
├── README.md
└── README_cn.md
```

## 模型编译环境

模型转换请在 x86 Linux 主机的 OpenExplore Docker 环境中完成，不建议在板端安装编译工具链。RDK S100/S100P/S600 共用同一套 Nash 工具链（`hb_compile`），仅 `--march` 不同。

工具链文档与下载：

- OE 在线手册：<https://developer.d-robotics.cc/oe_s_doc/index.html>
- RDK S100 工具链文档：<https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=4.0.5&p=RDK+S100>
- RDK S600 工具链文档：<https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=5.1.0&p=RDK+S600>

### 1. 安装 Docker

按照 Docker 官方说明安装并验证：

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. 获取并加载离线镜像

下载 OpenExplore CPU 版 Docker 镜像（S100/S100P/S600 共用）并加载：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
sudo docker images
```

也可直接在线拉取：

```bash
docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

> **注意**：若上方下载地址过期或失效，请到 OE 在线手册查看最新地址，或在文档站点提交 issue 反馈修复。

### 3. 启动容器

建议挂载仓库目录并增大共享内存，避免编译过程中的内存问题：

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

`<docker-image-name>` 可通过 `sudo docker images` 查看加载后的镜像名称与标签。

## 转换流程

EfficientSAM 被拆分为两个 `.hbm` 模型——图像编码器与掩码解码器，二者必须严格按顺序准备。解码器的校准输入是编码器的输出 featuremap（`image_embeddings`，形状 `1×256×32×32`），而不是图片；该 featuremap 只能由编码器运行得到。因此，必须先导出（最好再运行一次）编码器，才能准备解码器的校准数据。请按以下步骤顺序执行。

### 1. 导出 ONNX

ONNX 模型不随仓库分发，请先克隆上游仓库并放置权重：

```bash
cd samples/vision/efficient_sam/conversion
git clone https://github.com/yformer/EfficientSAM.git workspace/EfficientSAM
# 将权重放置到 workspace/EfficientSAM/weights/efficient_sam_vitt.pt
```

然后导出两个 ONNX 模型：

```bash
python3 scripts/export_encoder_onnx.py --output ./efficient_sam_vitt_encoder_512_op11.onnx
python3 scripts/export_decoder_onnx.py --output ./efficient_sam_vitt_decoder_512_op11.onnx
```

若仓库路径或权重路径与默认值不同，可用 `--repo` / `--checkpoint` 覆盖。完整参数见 `python3 scripts/export_encoder_onnx.py -h`。

### 2. 准备编码器校准数据

准备 20 到 50 张有代表性的 RGB 图片，再转换为编码器的 NCHW float32 输入 tensor（RGB，缩放到 `1/255`）：

```text
calibration_images/
├── 000001.jpg
├── 000002.jpg
└── ...
```

```bash
python3 scripts/prepare_calibration.py --src ./calibration_images --out ./calibration_data --num 30
```

tensor 写入 `./calibration_data/batched_images/*.npy`，至少需要 20 张图片。

### 3. 获取编码器 embedding

解码器的校准输入是一份真实编码器 embedding（形状 `1×256×32×32`，float32），而不是图片。必须先运行编码器才能得到。二选一：

- **主机上运行浮点编码器**（最简单，无需板卡）：用 committed 的 `dump_encoder_embedding.py` 脚本在第 1 步导出的编码器 ONNX 上跑一张图，把 `image_embeddings` 输出写成原始 `.bin` 文件：

  ```bash
  python3 scripts/dump_encoder_embedding.py \
    --image ./calibration_images/000001.jpg \
    --output ./encoder_embedding.bin
  ```

  若未安装 `onnxruntime`，先 `pip install onnxruntime`。

- **板端运行已编译编码器**（保真度最高）：在第 5 步编译出编码器后，用 `hrt_model_exec` 跑一次并把 `image_embeddings` 输出 dump 成同一个 `.bin`。这样解码器按量化后编码器的真实输出分布校准。

### 4. 准备解码器校准数据

把这一份 embedding 交给 `prepare_efficient_decoder_calibration.py`，脚本会生成 `--num` 份派生的 featuremap（小幅缩放扰动），写入 `./decoder_calibration/image_embeddings/`：

```bash
python3 scripts/prepare_efficient_decoder_calibration.py \
  --embedding ./encoder_embedding.bin \
  --out ./decoder_calibration \
  --num 30
```

EfficientSAM 解码器在导出时已把点提示烘焙为常量，因此这里无需准备提示 tensor。

### 5. 编译 HBM 模型

若 embedding 来自**浮点编码器**（第 3 步），则两组校准数据都已就绪，可一次性编译编码器与解码器：

```bash
# RDK S100 (Nash-E)
python3 scripts/quantize.py --march nash-e

# RDK S100P (Nash-M)
python3 scripts/quantize.py --march nash-m

# RDK S600 (Nash-P)
python3 scripts/quantize.py --march nash-p

# 三个 march 全部编译
python3 scripts/quantize.py
```

若希望 embedding 来自**已编译编码器**，则先单独编译编码器、dump 出 embedding、准备解码器校准数据，再编译解码器：

```bash
python3 scripts/quantize.py --config configs/efficient_sam_encoder_nashe_config.yaml
# 运行一次编码器，把 image_embeddings dump 为 encoder_embedding.bin(第 3 步板端方案)
python3 scripts/prepare_efficient_decoder_calibration.py --embedding ./encoder_embedding.bin --out ./decoder_calibration
python3 scripts/quantize.py --config configs/efficient_sam_decoder_nashe_config.yaml
```

生成的 `.hbm` 位于 `bpu_model_output_encoder_nashe/` 与 `bpu_model_output_decoder_nashe/`。将其拷贝到模型目录，供 `runtime/python/run.sh` 与 `runtime/python/main.py` 直接使用：

```bash
cp bpu_model_output_encoder_nashe/efficient_sam_vitt_encoder_512x512_nashe.hbm ../model/nash-e/
cp bpu_model_output_decoder_nashe/efficient_sam_vitt_decoder_512_nashe.hbm ../model/nash-e/
```

对 `nash-m`、`nash-p` 重复上述操作。输出文件名已与 runtime 期望一致，无需重命名。

### 6. 脚本参数说明

完整参数请运行 `python3 <脚本名> -h`。

**`quantize.py`**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--march` | 目标架构：`nash-e`（S100）、`nash-m`（S100P）、`nash-p`（S600）。省略则编译全部。 | 全部 |
| `--config` | 编译单个 committed YAML(覆盖 `--march`)。 | 无 |

**`export_encoder_onnx.py` / `export_decoder_onnx.py`**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--repo` | 已克隆的上游仓库路径。 | `./workspace/EfficientSAM` |
| `--checkpoint` | `efficient_sam_vitt.pt` 路径。 | `./workspace/EfficientSAM/weights/efficient_sam_vitt.pt` |
| `--output` | 输出 ONNX 路径。 | `./efficient_sam_vitt_{encoder,decoder}_512_op11.onnx` |
| `--size` | 正方形输入尺寸。 | `512` |
| `--opset` | ONNX opset 版本。 | `11` |
| `--points`(仅解码器) | 烘焙进解码器的两个正点提示 `x1 y1 x2 y2`。 | `248 210 302 315` |

**`prepare_calibration.py`**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--src` / `--image-dir` | 校准图片目录。 | 必填 |
| `--out` / `--output-dir` | 输出根目录(写入 `<out>/batched_images/`)。 | 必填 |
| `--num` | 校准 tensor 数量。 | `30` |
| `--size` / `--image-size` | 正方形输入尺寸。 | `512` |

**`dump_encoder_embedding.py`**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--onnx` | 编码器 ONNX 路径。 | `./efficient_sam_vitt_encoder_512_op11.onnx` |
| `--image` | 单张输入图片路径。 | 必填 |
| `--output` | 输出原始 embedding `.bin` 路径。 | `./encoder_embedding.bin` |
| `--size` | 正方形输入尺寸。 | `512` |

**`prepare_efficient_decoder_calibration.py`**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--embedding` | 原始 float32 编码器 embedding `1×256×32×32`(即 `.bin` 文件)。 | 必填 |
| `--out` | 输出根目录(写入 `<out>/image_embeddings/`)。 | `./decoder_calibration` |
| `--num` | 校准 featuremap 数量。 | `30` |

## 输入输出协议

导出与量化必须保持一致的下述 tensor 链。

**编码器**(`efficient_sam_vitt_encoder_512_op11.onnx`)：

- 输入 `batched_images`：`1×3×512×512`，RGB，float32，缩放到 `1/255`。
- 输出 `image_embeddings`：`1×256×32×32`，float32。

**解码器**(`efficient_sam_vitt_decoder_512_op11.onnx`)：

- 输入 `image_embeddings`：`1×256×32×32`，float32——即编码器输出。
- 输出 `low_res_masks` + `iou_predictions`：低分辨率掩码 logits(由 `1×1×256×256` 上采样至 `1×1×512×512`)与 IoU 分数。

点提示已在导出时烘焙为解码器的常量 buffer，因此编译后的解码器只接收 `image_embeddings`。两个网络均通过 committed 配置中的 `calibration_parameters.optimization: set_all_nodes_int16` 量化为 int16。

## 编译结果检查

```bash
hrt_model_exec model_info --model_file efficient_sam_vitt_encoder_512x512_nashe.hbm
hrt_model_exec perf --model_file efficient_sam_vitt_encoder_512x512_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file efficient_sam_vitt_decoder_512_nashe.hbm --thread_num 1
```

## 常见问题

- **权限问题**：宿主机复制回文件时出现权限错误，可检查文件属主或使用 `sudo chown -R`。
- **内存/IPC 报错**：启动 Docker 容器时请添加 `--shm-size=15g`。
- **优化等级报错**：Nash 架构不支持 `O3` 时，请使用 `O0`、`O1` 或 `O2`。
- **"No calibration images found"**：请把 `--src` 指向至少含 20 张 `.jpg`/`.png` 的目录；编码器校准至少需要 20 个文件。
- **解码器 embedding 形状报错**：`--embedding` 必须是原始 float32 数组，恰好 `1×256×32×32`(262144) 个值——是编码器的输出，而不是图片或 `.npy` 文件。
- **形状不一致**：导出与校准都请保持 `--size 512`(得到 `image_embedding_size 32`)，否则 `1×256×32×32` 的 tensor 契约会被破坏。

## License

本目录下的工具遵循 [Apache 2.0 License](../../../../LICENSE)。