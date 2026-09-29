[English](README.md) | 简体中文

# UNetMobileNet 转换边界

<a id="source-model"></a>
## 源模型

S 源提供预编译 HBM 和架构介绍，但没有训练框架版本、checkpoint、精确训练仓库或导出代码。U-Net/MobileNet 论文描述模型家族，不能据此还原本部署权重。

<a id="toolchain-targets"></a>
## 工具链与目标

发布制品仅有 S100 与 S600。平台 profile 分别标识 nash-e、nash-p，但没有本模型的 OE 版本、YAML 或编译配置。S100P 无发布制品，不推定存在 Nash-M 配方。

<a id="export"></a>
## 导出

钉住的源中没有可执行导出配方。补齐前需要：有许可的训练权重、精确架构／版本、导出依赖、张量名称和验证过的静态图。不能把下载的 HBM 当作 ONNX 导出输入。

<a id="calibration"></a>
## 校准

没有随附校准数据集、归一化策略、前处理配方或量化配置。运行时 BGR→NV12 输入处理不能证明训练归一化方式，两张随附图片不构成代表性校准集。

<a id="compile"></a>
## 编译

缺少源输入／配置，因此不提供模型编译命令。原生 runtime 的 CMake 编译是另一件事，参见 [C++ 构建](../runtime/cpp/README_cn.md#build)。当前推理使用[发布模型准备流程](../model/README_cn.md#preparation)。

<a id="validation"></a>
## 验证

未来转换模型须证明目标身份、uint8 Y [1,1024,2048,1]/UV [1,512,1024,2]、NHWC [1,H,W,19] int32 或 F32 分数以及正确量化描述。对齐预处理，并在真实输入上与可信参考比较解码 mask；仅有主机 fixture 不够。本轮未执行 ONNX/BPU 数值对照。

<a id="artifacts"></a>
## 产物

当前产物仅包括外部下载的两份 HBM（发布 SHA-256 未知）及 runtime 输出，不声称已有 ONNX、checkpoint、编译日志或校准数据。后续建立真实转换流程时须保留这些来源材料。

<a id="known-gaps"></a>
## 已知缺口

缺少精确 checkpoint／架构、导出脚本／依赖、校准和归一化、OE 版本／目标配置、编译模型对应关系与真实验证。这些是源缺口，明确记录，不用泛化工具链命令填充。板端验证仍为 not-run。
