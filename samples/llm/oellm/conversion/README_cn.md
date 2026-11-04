[English](README.md) | 简体中文

# 官方预编译制品

<a id="source-model"></a>
## 模型来源

使用 [模型清单](../model/README_cn.md#artifacts) 中的 SDK 官方 S600 ZIP。上游模型与逐变体许可保存在各 manifest；HF 完整 revision 未披露时保持 null。

<a id="directory"></a>
## 目录

本目录解释官方制品来源；校验值在 `../model/manifests/`，SDK 配置在 `../runtime/cpp/configs/`。

<a id="toolchain-targets"></a>
## 工具链与目标

SDK 2.0.0 / S600 nash-p。每个包的实际 HBDK 编译器、核心数、prefill chunk 与上下文分别登记，不把其他尺寸的编译规格套过来。

<a id="export"></a>
## 导出

当前方案使用已导出的官方 HBM，无需导出上游模型。

<a id="calibration"></a>
## 校准

无需为已发布包重复校准。官方包未附完整校准语料或实验日志，不沿用其他制品的量化精度/PPL。

<a id="compile"></a>
## 编译

HBM 已预编译。改变目标、核数、上下文或分块时应建立独立 SDK 转换，并登记新的制品身份。

<a id="validation"></a>
## 文件验证

按 [模型准备](../model/README_cn.md#preparation) 校验 ZIP 和全部成员；HBM 元数据在工具链主机读取，未执行计算图。

<a id="artifacts"></a>
## 产物

OSS ZIP 保持官方原始字节；本仓库提供清单、原生启动器和部署配置。

<a id="known-gaps"></a>
## 定制转换准备

若要定制量化，需另行固定上游 revision、SDK 配方、校准集与新制品，不能修改当前包的身份后沿用官方性能。
