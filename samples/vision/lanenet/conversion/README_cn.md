[English](README.md) | [简体中文](README_cn.md)

# LaneNet 转换

本目录保留源 YAML，并增加显式校准/配置准备。源导出脚本缺失。主机替身测试覆盖
准备过程和假编译器，本轮没有真实 ONNX 导出、OE 编译或板端执行。

<a id="source-model"></a>
## 源模型与前提

源 checkpoint 地址为
[best_model.pth](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/best_model.pth)，
未提供摘要、模型代码版本和架构绑定。源以 `test.py --img ... --model best_model.pth`
说明导出，但 `test.py` 不存在。复现发布制品前需取得匹配模型/导出代码；权重文件名
不能确定 ONNX 图或前处理边界。

源列出 Python≥3.6、Torch≥1.2、torchvision≥0.4.0、NumPy≥1.7、OpenCV、pandas、
matplotlib 等历史导出依赖。这些宽泛旧下限不是验证过的环境锁定。新主机准备工具
需要 Python、NumPy、OpenCV、PyYAML，不加载 Torch 或板端 SDK。

<a id="toolchain-targets"></a>
## 工具链与目标

保留的 `config.yaml` 声明 **nash-e/S100**、latency 模式、O2、
`set_all_nodes_int16`。源未固定 OE Docker/编译器版本，资源链接为
[OE 环境](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
和[工具链手册](https://toolchain.d-robotics.cc/)。请显式准备匹配的 x86 Linux 工具链，
包装不会安装环境。

只有 S100 有发布模型。新增 S100P/S600 编译需独立验证图、march/配置和运行时绑定；
改名或使用 S100 制品不构成移植。图内 int16 量化不说明公开 embedding/二值张量类型。

<a id="export"></a>
## 导出边界与源模板

解决缺失导出前提后，提供非空 ONNX。源输入为 float32 RGB NCHW `[1,3,256,512]`，
INTER_AREA 拉伸、`/255`、ImageNet 均值/标准差。运行时消费命名 float32
`instance_seg_logits` 和 int64 `binary_seg_pred`，真实制品元数据仍未观察。
源转换文字提到三个输出，却没给出第三个身份，不应虚构名字或丢掉实测辅助输出。

原 YAML 逐字节保留。它引用 `../log/best_model.onnx`、`../cal_data`，输出前缀
`lanenet256x512_nv12` 有误导性：运行/训练输入均为 **featuremap NCHW**，不是 NV12。
包装另写配置，使用调用者绝对路径和 `lanenet256x512` 前缀，不改源文件。
prepare-only 检查文件/校准身份，不验证 ONNX 图正确性或输出语义。

<a id="calibration"></a>
## 显式校准准备

源引用 TuSimple 和缺失的 `get_calibration_data.py`，没给原划分、图片 ID、种子或
变换代码。请合法取得本地图片；新选择的图片不能声称就是源校准集。

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
路径/摘要、实际图片形状、张量形状/类型/摘要、数量和协议，不下载模型或数据集。
这是新增显式流程，不是恢复了缺失源脚本。需确认 ONNX 是否自带归一化，重复应用会
改变契约。

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
文件集合完全匹配，保留源 march/量化/编译设置，然后对生成配置调用 `hb_compile -c`。
捕获 stdout/stderr；非零退出或预期 HBM 缺失/为空均失败，即便编译器返回零。
`--prepare-only` 不调用编译器，不代表成功转换。

<a id="validation"></a>
## 验证与历史证据

真实运行需检查公开张量名称/布局/类型，对相同输入比较原始 embedding、精确二值
标签，记录所有辅助输出，并验证 S100 执行。颜色不能证明车道实例精度，源没有聚类
或稳定车道 ID 解码。

源称三个输出余弦相似度较高，但所引用 `test_data/readme_img/result.jpg` 缺失，
也无数值表、数据集或原始日志。不虚构替代图或相似度。历史源 HRT 记录为200帧、
平均14.245ms、69.894FPS，固件、模型摘要和完整参数不明，不是新流程的基准。
参见[评估](../evaluator/README_cn.md)。

<a id="artifacts"></a>
## 制品与来源

准备步骤写入 `config.yaml`、`report.json`。真实成功编译还必须产生
`artifacts/lanenet256x512.hbm`，记录实测摘要并保留 `stdout.log`、`stderr.log`。
报告包含 ONNX/校准/模板/配置摘要和完整命令，制品来源明确写为
`caller-converted; not published asset authentication`。

运行时外部副本模式仍需精确契约身份和实际 S100 门禁。发布者摘要未知，不应把任意
自制权重标为原始发布制品。形状、名字或语义变化需新绑定。
[模型准备](../model/README_cn.md)单独说明发布路径。

<a id="known-gaps"></a>
## 剩余缺口

源导出代码/版本/checkpoint 摘要、原校准集、OE 版本和精度图仍缺失。原 README 还
链接了不存在的中文转换页，本双语说明提供当前流程。生成配置和假编译器测试不能
证明真实 OE 兼容、ONNX 正确或板端一致，这些验证仍为 not-run。
