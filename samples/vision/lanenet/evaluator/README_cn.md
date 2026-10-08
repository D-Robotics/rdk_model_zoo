[English](README.md) | [简体中文](README_cn.md)

# LaneNet 评估边界

本目录说明 LaneNet 输出对照和性能测量方法。数据集评估需准备标注图片并定义车道实例匹配规则。

<a id="dataset"></a>
## 数据集

内置的 [lane.jpg](../test_data/lane.jpg) 用作演示输入，四幅显示 PNG 用作可视化示例。数据集评估需准备标注图片，并记录训练/验证划分、标注转换、数据集校验和、许可、前处理和车道实例匹配规则。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

使用配套 S100 HBM、板端 SDK、Python、NumPy、OpenCV 和 PyYAML。C++ 比较需按 [C++ 说明](../runtime/cpp/README_cn.md) 构建原生程序。

<a id="command"></a>
## 命令

在 S100 上准备模型后，从仓库根目录运行：

```bash
bash samples/vision/lanenet/model/download.sh --target s100
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --output outputs/lanenet-eval
```

输出目录必须是新目录。运行成功返回 0，并写入原始张量和报告。

<a id="metrics"></a>
## 指标

相同模型和图片下，数值比较嵌入张量，声明所用容差；精确比较二值标签，单独比较可视化。数据集评估须先定义聚类、曲线拟合和车道实例匹配规则，再使用带标注数据评分。

<a id="outputs"></a>
## 输出与证据

单测将检查结果输出到终端。运行结果格式见 [Python 结果](../runtime/python/README_cn.md#results)和[原生结果](../runtime/cpp/README_cn.md#results-interpretation)：保留 NPZ 名称映射或原生角色索引，不能只留截图。

<a id="reference-results"></a>
## 参考结果

已发布的 HRT 参考测量使用 200 帧：模型延迟 14.245 ms、69.894 FPS，对应板卡镜像、运行库/工具链版本及制品摘要未注明。Python/C++ 端到端延迟需在目标环境测量，并随结果记录这些条件。模型准备见[转换说明](../conversion/README_cn.md)。



<a id="boundaries"></a>
## 适用范围

数据集精度与车道实例指标在标注数据集上测量：先定义车道实例匹配规则，对嵌入执行明确定义的聚类与曲线拟合，再对照数据集标注计算指标。嵌入显示颜色是可视化渲染特征，车道实例归属由所选聚类/拟合算法给出。运行加速比与跨语言板端对照分别以目标环境中的实测记录为准。
