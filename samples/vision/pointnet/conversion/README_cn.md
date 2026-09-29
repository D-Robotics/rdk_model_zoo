[English](README.md) | 简体中文

# PointNet 模型转换说明

<a id="source-model"></a>
## 源模型与网络

源交付参考[S100 PointNet 项目](https://gitee.com/chenguanzhong/rdk_-s100_-point-net_-official)，
未固定训练 commit、框架版本、checkpoint 身份或权重校验值。已发布 HBM 不等于可复现转换配方。
PointNet 通过共享 MLP、最大值聚合和局部/全局特征拼接预测逐点标签，本例只包含四种椅子部件。

![网络结构](../test_data/readme_img/image-1.png)
![分割网络](../test_data/readme_img/image.png)

<a id="toolchain-targets"></a>
## 工具链与目标

| Target | march / OE 版本 | 编译配置 |
| --- | --- | --- |
| s100 | 源配方未固定 | 缺失 |

转换应在配套 OpenExplore 的 x86 Linux 主机执行，不属于板端推理类。
[OE 资源](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
和[工具链手册](https://toolchain.d-robotics.cc/)提供环境说明，但不能补齐本模型专属配方。

<a id="export"></a>
## ONNX 导出

未提供 checkpoint、导出脚本或固定导出环境。原算子说明包含 Conv、BatchNorm、ReLU；
下图保留为历史参考，不能证明新导出的全部算子都支持。运行边界是 float32 `(1,3,N)` 输入、
`(1,N,4)` 部件 logits；N 由编译制品固定，不是运行时可随意设置的选项。

![历史 ONNX 图](../test_data/readme_img/char_static.png)

<a id="calibration"></a>
## 校准

缺失校准数据集、子集规模、准备脚本和配置。源资料记录 int16 量化、“trans > 0.9999”、
“pred > 0.98”，但未给出指标定义与完整条件，不能将这些数值当作分割准确率。

![历史量化记录](../test_data/readme_img/pixpin_2025-07-07_20-44-37.jpg)

<a id="compile"></a>
## 编译

当前没有可复现该 HBM 的编译配置/命令。请通过[模型准备指南](../model/README_cn.md)
取得已发布制品，不能用任意通用编译命令冒充已验证流程。

<a id="validation"></a>
## 验证边界

```bash
# cwd: repository root; on S100 with the published HBM already prepared
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --no-plot --output-dir outputs/pointnet-check
```

这是功能 smoke 命令，统一入口目前 not-run。新编译制品须先建立来源、目标和张量契约；
通过形状检查并不证明与发布模型等价。声明精度前，需要使用相同点序、归一化方法，与固定浮点参考比较逐点标签。

<a id="artifacts"></a>
## 制品

| 文件 | Target | 默认位置 |
| --- | --- | --- |
| pointnet.hbm | s100 | `samples/vision/pointnet/model/s100/pointnet.hbm` |

<a id="known-gaps"></a>
## 已知缺口

缺少固定训练源码/checkpoint、导出代码/环境、校准集/配置、编译器版本/march/配置、
发布方校验值和精度对照协议。本目录保留有用的网络/算子记录，不提供端到端转换流程。
代码许可为 [Apache-2.0](../../../../LICENSE)，外部模型和源码条款另行适用。
