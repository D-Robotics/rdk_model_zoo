[English](README.md) | 简体中文

# UNetMobileNet 转换边界

<a id="source-model"></a>
## 源模型

S 源提供预编译 HBM 和架构介绍，但没有训练框架版本、checkpoint、精确训练仓库或导出代码。U-Net/MobileNet 论文描述模型家族，不能据此还原本部署权重。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

发布制品仅有 S100 与 S600。平台 profile 分别标识 nash-e、nash-p，但没有本模型的 OE 版本、YAML 或编译配置。S100P 无发布制品，不推定存在 Nash-M 配方。

<a id="export"></a>
## 导出

钉住的源中没有可执行导出配方。补齐前需要：有许可的训练权重、精确架构／版本、导出依赖、张量名称和验证过的静态图。不能把下载的 HBM 当作 ONNX 导出输入。

<a id="calibration"></a>
## 校准

量化前准备代表性校准数据集、训练时的归一化和预处理配置，以及对应目标的量化配置。随附两张图片用于运行示例。

<a id="compile"></a>
## 编译

缺少源输入／配置，因此不提供模型编译命令。原生 runtime 的 CMake 编译是另一件事，参见 [C++ 构建](../runtime/cpp/README_cn.md#build)。当前推理使用[发布模型准备流程](../model/README_cn.md#preparation)。

<a id="validation"></a>
## 验证

转换模型须使用 uint8 Y [1,1024,2048,1] 与 UV [1,512,1024,2] 输入，以及 NHWC [1,H,W,19] int32 或 F32 分数和对应量化元数据。对齐前处理，在相同输入上比较参考结果与解码 mask。按下方步骤运行 ONNX/BPU 数值对照。

<a id="artifacts"></a>
## 产物

当前产物仅包括外部下载的两份 HBM（发布 SHA-256 未知）及 runtime 输出；未随附 ONNX、checkpoint、编译日志或校准数据。建立真实转换流程时记录这些来源材料。

<a id="known-gaps"></a>
## 补充准备

缺少精确 checkpoint／架构、导出脚本／依赖、校准和归一化、OE 版本／目标配置、编译模型对应关系与真实验证。这些缺口明确记录，不用泛化工具链命令填充。
