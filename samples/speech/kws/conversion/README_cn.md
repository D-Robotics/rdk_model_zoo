[English](README.md) | 简体中文

# KWS conversion availability

<a id="source-model"></a>
## 源模型

固定的 S100 源描述了 PaddlePaddle/PaddleAudio 体系的 MDTC 关键词模型，其中未包含训练权重和导出脚本。使用已发布模型推理时，按[模型说明](../model/README_cn.md)准备。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

已发布部署制品为 S100 HBM。转换需要目标平台的编译环境和配置；Runtime 前端依赖不能替代工具链选择与配置。X5、S100P、S600 没有已发布的 KWS 部署制品。

<a id="export"></a>
## 导出前提

从训练唤醒词权重和模型架构导出，并使用匹配的前处理定义与导出工具。记录权重/图 SHA-256、输入输出名称、形状和概率语义；如果图输出已是概率，应保留对应激活，运行时不会再次 sigmoid。

<a id="calibration"></a>
## 校准前提

在许可的校准划分上准备有代表性的正负音频。“hey snips”随附片段用于演示。匹配单声道 16 kHz PCM 缩放、60000 点截断/补零和固定 80-bin fbank，并记录源 ID、前端版本及特征摘要。

<a id="compile"></a>
## 编译前提

S100 编译需明确目标、特征输入布局、量化精度和最终输出语义；将编译日志及产物摘要与 HBM 一并保存，制品身份保持为 S100。

<a id="validation"></a>
## 验证流程

先在留出的正负数据上对照浮点图与原模型，再比较编译模型 metadata 和浮点图分数。分数容差、阈值判定、误唤醒/漏唤醒及延迟分别报告。

<a id="artifacts"></a>
## 转换输出

记录权重/源码身份、导出命令、图契约、校准清单与特征、编译配置及日志、产物摘要和浮点/编译对照结果。已发布 HBM 的下载和源性能见[模型](../model/README_cn.md)及[评估](../evaluator/README_cn.md)说明。

<a id="known-gaps"></a>
## 已发布运行制品

按[运行说明](../runtime/python/README_cn.md)使用已发布 S100 HBM 和匹配板端 SDK。新转换从上述训练权重、模型图和校准数据开始。
