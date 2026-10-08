[English](README.md) | 简体中文

# DOTA-v1.0 数据集资源

**DOTA-v1.0** 是面向旋转框目标检测的大规模航拍图像基准：2,806 张大尺寸航拍图，
标注 15 个类别的旋转边界框（最新数字与划分定义以官方页面为准）。本目录只随附
15 类类别表和三张示例图块；数据集本身**不随仓提供**，也**没有下载脚本**——
需按官方条款从官方来源手动获取。

<a id="files"></a>
## 随仓文件

```text
dotav1/
├── README.md             # 英文指南
├── README_cn.md          # 本指南
├── dota_classes.names    # 15 个类别名，每行一个
└── asset/
    ├── P0009.png         # 示例图块（原始 DOTA 图片 ID）
    ├── P0014.png
    └── P0035.png
```

### dota_classes.names

共 15 项（已按文件实测），每行一个类别名。DOTA 原生标注按**八个坐标、类别名和
难度标志**记录每个实例（官方标注规范）；**不存在官方的数字类别 ID 表**。因此本
文件只是仓库侧的固定列表：行序就是本仓库在需要编号 DOTA 类别列表时采用的顺序。
任何把 DOTA 转成数字 ID 格式的工具（例如 COCO 风格转换）都定义自己的索引映射——
评分时必须显式取得并声明该映射；不要假定它与本列表或下文的模型输出顺序一致。

```text
plane, baseball-diamond, bridge, ground-track-field, small-vehicle,
large-vehicle, ship, tennis-court, basketball-court, storage-tank,
soccer-ball-field, roundabout, harbor, swimming-pool, helicopter
```

<a id="ordering"></a>
## 本仓库两套不能混用的顺序

本仓库存在两套不同的 15 类顺序，**不可互换**：

| 文件 | 顺序 | 含义 |
| --- | --- | --- |
| `datasets/dotav1/dota_classes.names`（本目录） | 仓库侧固定列表（见上） | 本仓库需要编号 DOTA 类别列表时使用的行位置；不是官方 ID 表 |
| `samples/vision/ultralytics_yolo/test_data/ultralytics_dota_classes.names` | Ultralytics DOTA 模型输出顺序（如 `ship` = 1、`storage-tank` = 2，从 0 开始） | Ultralytics OBB 头的模型输出列命名 |

两套顺序只有索引 0（`plane`）相同，其余 14 个位置都不同：用其中一套去命名按另一套
排列的输出或转换，会把类别错位。对转换成数字 ID 的 DOTA 数据集评分时，ID 映射由该
转换定义；请在转换时记录对应关系。[Ultralytics YOLO OBB 评估器](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md)
导出旋转矩形/多边形，不计算 DOTA AP。它的 `--label-path` 参数只为旧命令兼容保留，
不参与评分。预测导出即评测接口：DOTA AP 由外部 DOTA 评分器基于导出的预测计算，导出文件本身不是精度结果。

<a id="usage"></a>
## 这些资源被谁使用

可使用 `asset/P0009.png`、`asset/P0014.png`、`asset/P0035.png` 作为 OBB 示例输入。[Ultralytics YOLO sample](../../samples/vision/ultralytics_yolo/README_cn.md) 使用 `test_data/ultralytics_dota_classes.names` 标识模型输出列。

Ultralytics OBB 使用 Sample 内的模型顺序标签文件；转换后的评估数据集须提供转换过程使用的类别映射。

三张 `asset/` 图块仅作直观参考和离线实验片段——不是可用的评估子集。

<a id="reference"></a>
## 官方网站与条款

- 官方页面：<https://captain-whu.github.io/DOTA/dataset.html>
  （下载申请、数据划分与条款均在此发布）

数据集与标注规范：[DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html)。转换标注进行评估时，请明确保存类别名称与索引的映射。
