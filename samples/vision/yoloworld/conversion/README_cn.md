# YOLOWorld 转换

<a id="source-model"></a>
## 源模型

X5 源交付提供已编译 `yolo_world.bin` 协议和离线词向量 JSON，但没有 checkpoint、
导出脚本、校准集、Bayes-E YAML 或可复现编译配方。复制的
`source/yoloworld_det.py` 记录 runtime 数学的来源，不是转换工具。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── source/  # source 相关文件
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>

<a id="export"></a>
## 工具链目标与导出

目标为 RDK X5，输入 640，图片 F32 NCHW 加文本 F32[1,32,512,1]，输出为两个
F32 张量。重新构建需要源模型/权重和匹配的 OpenExplorer 包；OE 离线 Docker 镜像
的地瓜开发者社区讨论见 <https://forum.d-robotics.cc/t/topic/35229>。仓库没有发布
这些内容，因此不提供导出命令。

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
运行 `evaluator/compare.py`，与源实现进行 raw 和结果对拍。唯一发布的转换产物是
清单模型；离线词向量是另一个必需输入。

<a id="known-gaps"></a>
## 补充准备

Checkpoint、导出、校准、编译日志和发布者模型摘要均未知；转换无法由本仓库复现。
