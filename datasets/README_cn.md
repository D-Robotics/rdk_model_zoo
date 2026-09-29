[English](./README.md) | 简体中文

# 数据集资源

本目录存放本仓库各 sample 共用的数据集侧资源：类别/标签文件、少量随仓示例图片
和一个获取脚本。这里**不包含**完整数据集；只有 ignore 规则覆盖的布局才会自动不进
Git——COCO 脚本默认的 `coco_full/` 输出**没有**被忽略，请在仓库外运行或先添加
本地排除（见 [COCO 指南](coco/README_cn.md)）。所有数据集均需按其自身条款手动
获取。

<a id="index"></a>
## 索引

| 目录 | 随仓资源 | 使用方 | 指南 |
| --- | --- | --- | --- |
| [coco/](coco/) | `coco_classes.names`（80 类）、示例图片 `bus.jpg` / `kite.jpg`、完整数据集下载脚本 | [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README_cn.md) 检测/分割/姿态评估器、[YOLOE](../samples/vision/yoloe/README_cn.md) 评估器（COCO 格式输入）、YOLOv5 sample（测试图） | [COCO 指南](coco/README_cn.md) |
| [imagenet/](imagenet/) | `imagenet_classes.names`（1000 类字典字面量）、示例图片 `zebra_cls.jpg` | 全部分类 sample（`--label-file`）、[Ultralytics YOLO](../samples/vision/ultralytics_yolo/README_cn.md) 分类评估器（Top-1/Top-5） | [ImageNet 指南](imagenet/README_cn.md) |
| [dotav1/](dotav1/) | `dota_classes.names`（15 类，仓库侧固定列表）、三张示例图块 | 当前统一代码没有脚本读取本目录；历史 X5 YOLO26 旋转框流程曾使用 | [DOTA 指南](dotav1/README_cn.md) |
| [PascalVOC/](PascalVOC/) | 仅有参考链接——无随仓文件 | [UNet](../samples/vision/unet/README_cn.md) 评估器（VOC 2012 分割，需另行获取） | [Pascal VOC 指南](PascalVOC/README_cn.md) |
| [yoloe/](yoloe/) | `yoloe_seg_pf_classes.names`——固定 4585 类 prompt-free 词表 | [YOLOE](../samples/vision/yoloe/README_cn.md) sample 及评估器 | [YOLOE 指南](yoloe/README_cn.md) |

<a id="boundaries"></a>
## 边界

- **除 COCO 外没有任何自动下载。** 本目录只有
  [coco/download_full_coco.sh](coco/download_full_coco.sh) 一个脚本；ImageNet、
  DOTA 和 Pascal VOC 没有下载器，必须从官方来源手动获取。本目录脚本不属于主机
  测试的一部分，撰写这些指南时也没有执行过。
- **许可与条款按数据集各自约定。** 请在各自条款下获取数据集；各指南给出官方网站。
  随仓标签文件和示例图片只是用于离线冒烟检查的小片段，不是数据集的再分发。
- **类别索引 ≠ 数据集类别 ID。** 模型输出列是连续索引。COCO 标注文件使用稀疏的
  数字 ID；DOTA 原生标注记录的是类别**名称**，任何数字 DOTA ID 空间都来自定义它的
  转换，而不是格式本身。各数据集指南写明精确映射；[YOLOE 指南](yoloe/README_cn.md)
  说明独立的 4585 类 prompt-free 词表，其索引既不是 COCO category ID，也不是
  Ultralytics COCO-80 输出索引。

<a id="provenance"></a>
## 来源

本目录中的数据集资源（类别表、示例图片、COCO 下载脚本）与 X5 交付分支快照
（`ac11571`）逐字节一致；README 指南在本次统一改造中重写，资源文件未改动。S 交付
分支（`380e1a2`）带有相同的 COCO、DOTA、ImageNet 资源（其 `coco_classes.names` 仅
换行符为 CRLF），但没有 `PascalVOC/` 与 `yoloe/` 目录。平台级冻结副本保留在
`platforms/x5/datasets/` 与 `platforms/s/datasets/` 作为历史参考；它们是归档快照，
不是活动入口。统一的 sample 文档在各 sample 目录下；活动清单为
`docs/release/x5/models.yaml` 与 `docs/release/s/models.yaml`。
