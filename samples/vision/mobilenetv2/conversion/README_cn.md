# MobileNetV2 模型转换

模型转换在 x86 Linux 主机上的 RDK OpenExplore (OE) 环境中执行，不是板卡
操作。随附的 `mobilenetv2_config.yaml` 面向 **S100**（`march: "nash-e"`）；
S600 构建与 S100 使用同一源 ONNX 与同一量化配置——仅把 `march` 改为
`nash-p`。

<a id="source-model"></a>
## 源模型

MobileNetV2（[论文](https://arxiv.org/abs/1801.04381)，
[timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)）。
上游流程使用 `timm` 导出 ONNX。下文的导出、校准与 x86 推理命令为原始
配方：请在自备的导出工作区中执行，该工作区需提供辅助脚本
`runtime/python/get_mobilenetv2_onnx.py`、
`runtime/python/timm2onnx_local.py`、`runtime/python/get_calibration_data.py`、
`runtime/python/x86_inference.py` 与 `runtime/python/s100_inference.py`。
仓库侧携带量化 YAML（`mobilenetv2_config.yaml`）与测试图，相关引用按
原样使用。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── mobilenetv2_config.yaml  # 配置
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标板卡匹配的 OE Docker/工具链版本。权威入口：
[RDK S 工具链总览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。
目标：X5 用 `hb_mapper` 编译，march `bayes-e`；S100 用 `hb_compile`，
march `nash-e`；S600 为 `nash-p`。容器挂载仓库到 `/workspace` 并给足共享
内存（`--shm-size=15g`）。

<a id="export"></a>
## 导出

上游流程使用 `timm`（PyTorch Image Models）。安装依赖、登录 Hugging Face
（脚本从 `timm/mobilenetv2_100.ra_in1k` 拉取）后导出：

```bash
# cwd：提供 runtime/python/get_mobilenetv2_onnx.py 的导出工作区
pip install timm onnx
huggingface-cli login
python runtime/python/get_mobilenetv2_onnx.py
```

无法配置代理时，从
[timm/mobilenetv2_100.ra_in1k](https://huggingface.co/timm/mobilenetv2_100.ra_in1k)
手动下载模型并本地转换：

```bash
# cwd：提供 runtime/python/timm2onnx_local.py 的导出工作区
python runtime/python/timm2onnx_local.py
```

导出完成后脚本打印模型 metadata：

```text
input: (3, 224, 224)
mean (0.485, 0.456, 0.406)
std (0.229, 0.224, 0.225)
Simplified model is valid.
Simplified model saved to mobilenetv2_100.onnx
Total number of parameters in the model: 3487818
```

两个导出脚本都在[源模型](#source-model)所述的导出工作区内运行，产出
`mobilenetv2_100.onnx`；请把结果放到本目录 YAML 旁边再编译。

<a id="calibration"></a>
## 校准

模型使用 [ImageNet](https://image-net.org/) ILSVRC2012 数据集（训练集约
120 万张 / 验证集 50,000 张 / 测试集 100,000 张，1000 类）。原始流程从
100 张验证图（`ILSVRC2012_val_*.JPEG`）生成 float32 校准目录：

```text
imagenet/
├── calibration_data/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...  (100 images)
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── val.txt
```

```bash
# cwd：提供 runtime/python/get_calibration_data.py 的导出工作区
python runtime/python/get_calibration_data.py
```

随附 YAML 消费 `../calibration_data_bgr` 的 float32 数据（BGR 顺序，
`mean_value: 103.53 116.28 123.675`，`scale_value: 0.017429 0.017507
0.017124`）；校准生成脚本未随本交付提供。再生成时必须记录所用图像清单
与预处理，其产物才能与已发布制品比较。

<a id="compile"></a>
## 编译

完整编译前先快速校验 ONNX 模型：

```bash
hb_compile --model mobilenetv2_100.onnx --march nash-e
```

使用参考 YAML 与校准数据执行量化编译：

```bash
hb_compile --config conversion/mobilenetv2_config.yaml
```

编译产物写入 `model_output/mobilenetv2_224x224_nv12.hbm`（按随附 YAML
的 `working_dir`/前缀）。S600 只需把 march 改为 `nash-p`。对比目标、输入 metadata、输出
shape/dtype 与数值结果之后，才能把再生成制品视为与已发布制品等价。

| 配置 | 目标 | 命令（OE 容器内） |
| --- | --- | --- |
| `mobilenetv2_config.yaml` | s100 | `hb_compile --config mobilenetv2_config.yaml` |

<a id="validation"></a>
## 验证

对再生成制品，按 OE 手册执行 `hb_perf` 与 `hrt_model_exec` 并保留完整
输出；随后在匹配板卡上用样例运行时确认契约：S100/S600 暴露 Y
`[1,224,224,1]`、UV `[1,112,112,2]` 与 F32 `[1,1000]` 输出；输出语义为
softmax 之后的概率。原始配方还记载两个工作区脚本：
`runtime/python/x86_inference.py` 在 x86 上做 ONNX/HBIR/HBM 推理并支持
验证集精度校验；`runtime/python/s100_inference.py` 在板上用 HBM 推理。
请在导出工作区内执行：

```bash
# cwd：导出工作区
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  -i test_data/zebra_cls.jpg
```

```bash
# cwd：导出工作区
python3 runtime/python/x86_inference.py \
  -m model_output/mobilenetv2_224x224_nv12_quantized_model.bc \
  --validate \
  -d ../../../imagenet/val \
  -l ../../../imagenet/val.txt
```

原 S 构建的已发布量化记录（量化后余弦相似度）：

```text
+------------+-------------------+------------------+
| TensorName | Calibrated Cosine | Quantized Cosine |
+------------+-------------------+------------------+
| output     | 0.993383          | 0.988877         |
+------------+-------------------+------------------+
```

原 S 构建的工具链性能参考：

```text
FPS (1 core): 4968.89
Latency: 0.2 ms (201.3 us)
BPU conv original OPs per run: 601,548,544
```

<a id="artifacts"></a>
## 保留材料

- `mobilenetv2_config.yaml`

<a id="known-gaps"></a>
## 补充准备

从[源模型](#source-model)列出的、含辅助脚本的操作者工作区运行导出、校准和 x86 推理命令。X5 使用 `hb_mapper`、march `bayes-e` 和匹配的 X5 量化配置，通过 X5 OE 流程构建。S 系列 YAML 与配方命令见上文。
