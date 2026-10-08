[English](./README.md) | 简体中文

# PaddleOCR 模型转换

转换在与目标匹配的 RDK OpenExplorer（OE）环境内的 x86 Linux 主机上
执行，绝不在板端进行。提供两套完整、按目标分开的配方族——选择与
目标板匹配的一套：

| 目标 | 模型族 | 工具链 / march | 随仓配方 | 必须复现的运行时契约 |
| --- | --- | --- | --- | --- |
| X5 | PP-OCRv3 英文 | `hb_mapper`，`bayes-e`（OE X5 v1.2.8） | `x5/ptq_yamls/*.yaml` | packed NV12 检测器输入；`[1,40,97,1]` 识别器输出 |
| S100 | PP-OCRv6 | `hb_compile`，`nash-e`（S OpenExplore） | `s100/*_configs.yaml` | split NV12 检测器输入；`[1,40,18710]` F32 识别器输出；检测器尾部 `Dequantize` |

图文件、输出名、词典与输入协议均按目标区分。不要把 X5 检测器与
S100 识别器组合，也不能互相替换词典。S100 YAML 也可以把 `march`
改为 `nash-m`（S100P）或 `nash-p`（S600）重新编译；已发布 manifest
行只覆盖 X5（PP-OCRv3）与 S100（PP-OCRv6），S100P/S600 制品需先经
该编译路径准备后再使用。

<a id="source-model"></a>
## 源模型

- X5 模型对：PaddleOCR 上游 PP-OCRv3 英文推理模型
  （`en_PP-OCRv3_det_infer`、`en_PP-OCRv3_rec_infer`）。本仓库**不**
  提供 PP-OCRv3 导出器；从确切的上游 release 取得导出文件，编译前
  核对图输入。
- S100 模型对：PaddleOCR 上游 PP-OCRv6 推理模型，由[导出](#export)
  中的 Paddle2ONNX opset 19 命令导出。
- 词典是模型契约的一部分：X5 识别器按固定 96 字符字母表解码；
  S100 识别器按
  [`test_data/s100/ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt)
  （加 blank 与末尾空格）解码。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── s100/  # s100 相关文件
├── scripts/  # scripts 相关文件
├── x5/  # x5 相关文件
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标匹配的 OE Docker/工具链，并记录镜像 tag、OE 版本与主机
日期。

X5（OE X5 v1.2.8）——下载、加载并启动官方镜像：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

或在[地瓜开发者社区](https://forum.d-robotics.cc/t/topic/35229)下载离线 Docker 镜像。

S（OpenExplore）——启动匹配的 S 系列 OE 容器，在 `/workspace` 工作：

```bash
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <docker-image-name> /bin/bash
```

OE 资源入口：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>。
OE 工具链在线手册：<https://toolchain.d-robotics.cc/>。

`hb_mapper`、`hb_compile`、`hrt_model_exec` 由这些环境提供。随仓
S100 配方使用 `nash-e`；把 `march` 字段改为 `nash-m`（S100P）或
`nash-p`（S600）即可编译其他 S 系列 SoC，输出前缀保持不变。

<a id="export"></a>
## 导出

S100 PP-OCRv6 ONNX 导出（上游 PaddlePaddle；克隆与安装在主机、OE
步骤之前完成，此处涉及联网）：

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
python3 -m pip install -e .
python3 -m pip install paddle2onnx onnxruntime

# 输出：det_onnx/model_detv6.onnx 与 rec_onnx/model_recv6.onnx
paddle2onnx --model_dir ./inference/PP-OCRv6_det_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/det_onnx/model_detv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True

paddle2onnx --model_dir ./inference/PP-OCRv6_rec_infer \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./inference/rec_onnx/model_recv6.onnx \
  --opset_version 19 \
  --enable_onnx_checker True
```

把结果复制到 S100 YAML 期望的路径（或只修改文档声明的
`onnx_model` 字段）：`conversion/onnx/model_detv6.onnx` 与
`conversion/onnx/model_recv6.onnx`。

X5 PP-OCRv3：本仓库无自有导出器。配方期望来自确切 PP-OCRv3 release
的 `en_PP-OCRv3_det_infer.onnx` 与 `en_PP-OCRv3_rec_infer.onnx`；
本 sample 不把猜测的命令伪装成受支持的 X5 导出配方。

<a id="calibration"></a>
## 校准

识别器合适的校准语料是
[ICDAR2019-LSVT 数据集](https://ai.baidu.com/broad/introduction?dataset=lsvt)
（45 万张中文街景图：5 万张全标注 bbox+文本、40 万张弱标注仅文本；
[下载](https://ai.baidu.com/broad/download?dataset=lsvt)）。来自部署
领域的任意代表性图像集均可。

随仓辅助脚本把一个 BGR 图像目录转成随仓配方期望的输入张量（不转换
模型、不联网）。在 `samples/vision/paddle_ocr/conversion` 下执行：

```bash
# X5 hb_mapper：两个独立空目录避免检测/识别张量混淆；先把 det/rec
# YAML 的 cal_data_dir 指向它们。
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./x5/calibration_data_detector \
  --target x5 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops --output ./x5/calibration_data_recognizer \
  --target x5 --stage recognizer

# S100 hb_compile：这两个输出即随仓 YAML 中的 ../calibration_data 与
# ../calibration_data_rec_new/cropped_images_npy。
python3 scripts/prepare_calibration.py \
  --images /data/calibration/images --output ./calibration_data \
  --target s100 --stage detector
python3 scripts/prepare_calibration.py \
  --images /data/recognition/crops \
  --output ./calibration_data_rec_new/cropped_images_npy \
  --target s100 --stage recognizer
```

检测器张量缩放到 640×640 RGB float32；识别器张量缩放到 320×48
RGB float32、值域 `[0,1]`，对应运行时的 `no_preprocess` 输入。X5
输出扩展名 `.rgbchw`，S100 为 `.npy`。每个输出目录必须为空，第二个
阶段无法静默复用残留张量。请使用来自模型训练域的代表性图像/裁剪；
本脚本不决定精度，也不决定所需样本数。

<a id="compile"></a>
## 编译

X5 PP-OCRv3：在 OE X5 v1.2.8 环境内、放置好两个 ONNX 文件与校准
目录后，于 `conversion/x5` 执行（输出：`model_output/*.bin`；成功：
`hb_mapper` 退出码 0）：

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_det_infer.onnx
hb_mapper checker --model-type onnx --march bayes-e \
  --model ./en_PP-OCRv3_rec_infer.onnx

hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_det_config.yaml
hb_mapper makertbin --model-type onnx \
  --config ptq_yamls/paddleocr_rec_config.yaml
```

生成文件名：

```text
model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
model_output/en_PP-OCRv3_rec_infer-deploy_48x320_rgb.bin
```

检测器 YAML 保留观测到的 RGB mean/scales、`nv12` 运行时输入、
`bayes-e` 与 O3 编译级别；识别器 YAML 保留 `featuremap`
NCHW/no-preprocess 与三处观测到的 `p2o.Softmax.*` int16 节点映射。

S100 PP-OCRv6：在 S OpenExplore 环境内、于 `conversion/s100` 执行
（输出：`model_output/*.hbm`；成功：`hb_compile` 退出码 0）：

```bash
hb_compile -c paddleocr_det_configs.yaml
hb_compile -c paddleocr_rec_configs.yaml
```

生成文件名：

```text
model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

检测器配方保留尾部 `Dequantize` 节点——**不要**设置
`remove_node_type: "Dequantize"`：运行时把 `fetch_name_0` 当作 F32
概率图直接阈值化，删除该节点会改变契约。识别器保留三处
`p2o.Softmax.*` int16 节点映射与 `set_all_nodes_int16`；其类别数必须
与随仓词典加 blank、末尾空格保持对齐。

为其他 S 系列 SoC 编译 S 配方时，改 YAML 中的 `march`：

| 目标平台 | `march` | 输出前缀不变 |
| --- | --- | --- |
| RDK S100 | `nash-e` | `PP-OCRv6_*-deploy_*` |
| RDK S100P | `nash-m` | `PP-OCRv6_*-deploy_*` |
| RDK S600 | `nash-p` | `PP-OCRv6_*-deploy_*` |

各平台延迟/FPS 用以下命令获取：

```bash
hrt_model_exec perf --model_file model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
hrt_model_exec perf --model_file model_output/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

<a id="validation"></a>
## 转换后验证

上板前逐一检查生成制品（完整记录输出）：

```bash
hrt_model_exec model_info --model_file \
  model_output/en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin
hrt_model_exec model_info --model_file \
  model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
```

把张量名称、形状、dtype 与输出名同所选模型对的契约（见
[阶段 I/O](../runtime/python/README.md#stage-io)）比对；只登记完整
引用与元数据都匹配的制品。再在匹配板卡上用运行时与随仓测试图确认。

<a id="artifacts"></a>
## 产物

| 阶段 | 需保留的产物 |
| --- | --- |
| export | 两个 `.onnx` 文件、导出命令/版本（S100）、上游 release 身份（X5） |
| calibration | 校准目录、图像来源与辅助脚本调用 |
| checker | `hb_mapper checker` 命令与日志 |
| compile | 所用 YAML、OE 版本/镜像 tag、`model_output/` 清单 |
| deployment | [编译](#compile)列出的四个清单文件名 |
| validation | `hrt_model_exec model_info` 输出、板端运行命令、预测结果 |

<a id="known-gaps"></a>
## 缺失项

- 仓库无自有的 PP-OCRv3（X5）导出器；从确切的上游 release 取得 ONNX
  文件并记录所用 release 身份。
- 原始校准语料未入库；辅助脚本准备张量但不固定语料与样本数，因此不
  声明再生成制品与已发布制品等价。
- 非随仓 march 取值属于未声明的实验。
