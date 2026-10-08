[English](README.md) | [简体中文](README_cn.md)

# LaneNet 转换

本目录提供编译 YAML、校准准备和编译包装工具。按下文准备配套模型与导出代码，
生成 ONNX，再使用校准和编译命令处理该 ONNX。

<a id="source-model"></a>
## 源模型与前提

源 checkpoint 地址为
[best_model.pth](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth)，
请取得匹配的 LaneNet 模型/导出代码及其 `test.py` 入口，并记录 checkpoint
摘要、代码版本和模型结构。导出入口使用 `test.py --img... --model best_model.pth`，
图结构与前处理需符合下文运行时契约。

导出依赖包括 Python≥3.6、Torch≥1.2、torchvision≥0.4.0、NumPy≥1.7、OpenCV、
pandas 和 matplotlib；导出时记录实际安装版本。校准与配置准备工具使用 Python、
NumPy、OpenCV、PyYAML。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── compile.py  # Python 脚本
├── config.yaml  # 配置
└── prepare_calibration.py  # Python 脚本
```

<a id="toolchain-targets"></a>
## 工具链与目标

`config.yaml` 声明 **nash-e/S100**、latency 模式、O2、
`set_all_nodes_int16`。按以下链接选择匹配的 OE Docker/编译器环境：
[OE 环境](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
和[工具链手册](https://toolchain.d-robotics.cc/)。请显式准备匹配的 x86 Linux 工具链，
包装不会安装环境。

已发布模型面向 S100。面向 S100P/S600 时，需准备对应目标的图、march/配置和
运行时绑定。图内 int16 量化设置与公开 embedding/二值张量类型分别检查。

<a id="export"></a>
## 导出契约与配置

提供导出后的非空 ONNX，输入为 float32 RGB NCHW `[1,3,256,512]`，
INTER_AREA 拉伸、`/255`、ImageNet 均值/标准差。运行时消费命名 float32
`instance_seg_logits` 和 int64 `binary_seg_pred`。检查编译产物的输出元数据，
按实际名称、形状及类型保留辅助输出。

YAML 引用 `../log/best_model.onnx`、`../cal_data` 和输出前缀
`lanenet256x512_nv12`，运行/训练输入均为 **featuremap NCHW**。包装另写配置，
使用调用者绝对路径和 `lanenet256x512` 前缀。prepare-only 检查 ONNX 文件和
校准身份；图结构与输出语义在模型验证中检查。

<a id="calibration"></a>
## 显式校准准备

在本地准备
[TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3) 校准图片，并在
校准清单中记录许可、数据划分、图片 ID、选择种子和变换代码。

从仓库根目录执行，输入为已有图片目录，输出为新路径：

```bash
python -m samples.vision.lanenet.conversion.prepare_calibration \
  --images /work/tusimple/images --count 100 --output /work/lanenet/calibration
```

递归发现 JPG/JPEG/PNG/BMP，按路径字典序选前100张。count 必须为正且数量足够；
选中的图片不可解码时直接报错。它调用**与运行时相同的图片转张量函数**：BGR→RGB、
area 缩放256×512、float32 `/255`、通道均值`[.485,.456,.406]`和标准差
`[.229,.224,.225]`，再转 NCHW。

`data/000000.npy` 等文件保存 `[1,3,256,512]` float32。`manifest.json` 记录图片
路径/摘要、实际图片形状、张量形状/类型/摘要、数量和协议。需确认 ONNX
是否自带归一化，重复应用会改变契约。

<a id="compile"></a>
## 先准备配置，再编译

先生成配置检查，不调用 OE：

```bash
python -m samples.vision.lanenet.conversion.compile \
  --onnx /work/lanenet/best_model.onnx \
  --calibration-manifest /work/lanenet/calibration/manifest.json \
  --output /work/lanenet/config-review --prepare-only
```

再在兼容的 OE 环境中换一个新输出路径执行：

```bash
python -m samples.vision.lanenet.conversion.compile \
  --onnx /work/lanenet/best_model.onnx \
  --calibration-manifest /work/lanenet/calibration/manifest.json \
  --output /work/lanenet/compiled
```

包装检查校准声明/实际形状、float32 数值、归一化范围、摘要、唯一条目及数据目录
文件集合完全匹配，使用 YAML 的 march/量化/编译设置，然后对生成配置调用 `hb_compile -c`。
捕获 stdout/stderr；非零退出或预期 HBM 缺失/为空均失败，即便编译器返回零。
`--prepare-only` 只生成配置并校验输入，编译需另行调用 `hb_compile`。

<a id="validation"></a>
## 验证与参考性能

真实运行需检查公开张量名称/布局/类型，对相同输入比较原始 embedding、精确二值
标签，记录所有辅助输出，并验证 S100 执行。评估车道实例精度时，需对嵌入进行实例聚类，
再按匹配规则与数据集标签对照。

已发布的三个输出对照说明其余弦相似度较高。HRT 参考性能使用
200 帧、平均模型延迟 14.245 ms、69.894 FPS，固件、模型摘要和完整参数
未注明；在板端测量本 sample 时需记录这些条件。
参见[评估](../evaluator/README_cn.md)。

<a id="artifacts"></a>
## 制品与来源

准备步骤写入 `config.yaml`、`report.json`。真实成功编译还必须产生
`artifacts/lanenet256x512.hbm`，记录实测摘要并保留 `stdout.log`、`stderr.log`。
报告包含 ONNX/校准/模板/配置摘要和完整命令，制品来源明确写为
`caller-converted; not published asset authentication`。

运行时外部副本模式使用精确的 S100 契约身份。形状、名字或语义变化需匹配的绑定。
[模型准备](../model/README_cn.md)说明已发布制品路径。

<a id="known-gaps"></a>
## 准备要求

随模型准备导出代码/版本、checkpoint 摘要、校准清单、OE 版本和精度参考。
编译前检查生成配置，再验证 ONNX 输出及 S100 上的编译模型。
