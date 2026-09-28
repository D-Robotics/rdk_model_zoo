[English](./README.md) | 简体中文

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
排列的输出或转换，会把这些类别静默错位。对转换成数字 ID 的 DOTA 数据集评分时，
ID 映射由该转换定义——必须显式要求提供，不能假定与本仓库任一本地顺序一致。当前
OBB 路径也只输出预测：[Ultralytics YOLO OBB 评估器](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md)
导出旋转矩形/多边形，明确**不计算 DOTA AP**；其 `--label-path` 参数仅为兼容旧
命令保留，不参与评分。本仓库没有实现 DOTA 评分器；不得把 OBB 预测导出当作
精度结果。

<a id="usage"></a>
## 这些资源被谁使用

当前统一 `samples/` 代码没有任何脚本读取本目录。历史使用方保留为归档溯源：

- X5 交付分支的 `ultralytics_yolo26` sample 曾用 `asset/P0009.png` 作 OBB 测试图、
  `dota_classes.names` 作标签文件——见归档的
  [X5 YOLO26 运行指南](../../platforms/x5/samples/vision/ultralytics_yolo26/runtime/python/README_cn.md)
  与[评估指南](../../platforms/x5/samples/vision/ultralytics_yolo26/evaluator/README_cn.md)。
- 当前 [Ultralytics YOLO sample](../../samples/vision/ultralytics_yolo/README_cn.md)
  使用自带的 `test_data/ultralytics_dota_classes.names`（模型顺序）和随仓 OBB
  测试图；上述归档文件保持冻结参考。

三张 `asset/` 图块仅作直观参考和离线实验片段——不是可用的评估子集。

<a id="reference"></a>
## 官方网站与条款

- 官方页面：<https://captain-whu.github.io/DOTA/dataset.html>
  （下载申请、数据划分与条款均在此发布）

来源说明：X5（`ac11571`）与 S（`380e1a2`）交付分支仅有裸链接 README；本指南补充
实测文件清单和顺序警示。归档副本位于 `platforms/x5/datasets/dotav1/` 与
`platforms/s/datasets/dotav1/`。
