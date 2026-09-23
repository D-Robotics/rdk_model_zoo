# YOLOWorld 转换

<a id="source-model"></a>
## 源模型

固定 X5 源提供已编译 `yolo_world.bin` 协议和离线词向量 JSON，但没有 checkpoint、
导出脚本、校准集、Bayes-E YAML 或可复现编译配方。复制的
`source/yoloworld_det.py` 仅用于保留 runtime 数学来源，不是转换工具。

<a id="toolchain-targets"></a>
<a id="export"></a>
## 工具链目标与导出

目标为 RDK X5，输入 640，图片 F32 NCHW 加文本 F32[1,32,512,1]，输出为两个
F32 张量。重新构建需要源模型/权重和匹配的 OpenExplorer 包；仓库没有发布这些
内容，因此不编造导出命令。

<a id="calibration"></a>
<a id="compile"></a>
## 校准与编译

源目录没有校准数据集或量化配置。发布产物按原样使用；不能从 `.bin` 文件名推断
INT8 scale。未来转换记录必须在声称一致前记录 exporter/OE 版本、源 checkpoint
身份、词表生成方式、输入输出名称/形状、校准数据和 SHA-256。

<a id="validation"></a>
<a id="artifacts"></a>
## 验证与产物

使用前先按两个输入和 `classes_score` / `bboxes` 形状校验实际 metadata。在 X5
运行 `evaluator/compare.py`，对拍源/统一 raw 和结果。当前唯一发布转换产物是
清单模型；离线词向量是另一个必需输入。

<a id="known-gaps"></a>
## 已知缺口

Checkpoint、导出、校准、编译日志和发布者模型摘要均未知。因此转换状态为“无法由
本树复现”；不宣称已转换、板端或精度结果。
