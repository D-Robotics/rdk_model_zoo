[English](README.md) | 简体中文

# 官方预编译模型

[Sample](../README_cn.md)

<a id="source-model"></a>
## 模型来源

官方预编译包来自 OELLM SDK 2.0.0 S600 发布线。上游模型为 Google Gemma4 E2B/E4B；官方未提供 ZIP 对应的 HF 完整 revision，清单中保持 null。

<a id="directory"></a>
## 目录

本目录说明预编译制品来源；四个完整清单在 `../model/manifests/`，配套 SDK 配置在 `../runtime/cpp/configs/`。

<a id="toolchain-targets"></a>
## 工具链与目标

S600/nash-p、4 核、chunk 512、context 4096。各包实际 HBDK 编译器和 HBM 元数据保存在对应清单；运行使用配套 SDK 2.0.0 Runtime。

<a id="export"></a>
## 导出

当前入口直接使用已导出的官方 HBM，无需重新导出。定制模型应从 SDK Gemma 发布配方建立独立实验。

<a id="calibration"></a>
## 校准与量化

SDK Gemma 路线为 AutoClip/GPTQ → HPP → HBDK。官方 ZIP 没有提供这四个制品的校准语料或完整复现记录；不沿用旧自转换包的校准或精度结果。

<a id="compile"></a>
## 编译

使用官方包不需要重新编译 HBM。调整核数、上下文或分块时，须使用 SDK 的相应配方生成新的制品和配置。

<a id="validation"></a>
## 文件验证

按 [模型准备](../model/README_cn.md#preparation) 校验完整 ZIP SHA256 与各文件 SHA256；本步骤不执行板端推理。

<a id="artifacts"></a>
## 产物

官方 ZIP 保持原始字节。准备脚本额外生成绝对路径 `oellm_config.json`、`model-manifest.json` 和 `SAMPLE_LICENSE`。旧自转换 W4/W8 包已退出当前绑定。

<a id="known-gaps"></a>
## 定制转换准备

如需重新量化，请另行固定上游 revision、SDK 配方、校准集、精度结果和新制品校验值。当前 Sample 交付官方预编译方案。
