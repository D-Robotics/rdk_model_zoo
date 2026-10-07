[English](./README.md) | 简体中文

# 数据集资源

本目录存放共用的类别/标签文件、示例图片和 COCO 获取脚本。完整数据集需按各自条款
另行获取。COCO 脚本把 `coco_full/` 写入当前工作目录；下载时请将工作目录设在仓库外。

<a id="index"></a>
## 索引

| 目录 | 随仓资源 | 使用方 | 指南 |
| --- | --- | --- | --- |
| [coco/](coco/) | `coco_classes.names`（80 类）、示例图片 `bus.jpg` / `kite.jpg`、完整数据集下载脚本 | [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README_cn.md) 检测/分割/姿态评估器、[YOLOE](../samples/vision/yoloe/README_cn.md) 评估器（COCO 格式输入）、YOLOv5 sample（测试图） | [COCO 指南](coco/README_cn.md) |
| [imagenet/](imagenet/) | `imagenet_classes.names`（1000 类字典字面量）、示例图片 `zebra_cls.jpg` | 全部分类 sample（`--label-file`）、[Ultralytics YOLO](../samples/vision/ultralytics_yolo/README_cn.md) 分类评估器（Top-1/Top-5） | [ImageNet 指南](imagenet/README_cn.md) |
| [dotav1/](dotav1/) | `dota_classes.names`（15 类，仓库侧固定列表）、三张示例图块 | DOTA 类别映射与示例图块；Ultralytics YOLO OBB 使用 sample 自带的模型顺序标签文件 | [DOTA 指南](dotav1/README_cn.md) |
| [PascalVOC/](PascalVOC/) | 仅有参考链接——无随仓文件 | [UNet](../samples/vision/unet/README_cn.md) 评估器（VOC 2012 分割，需另行获取） | [Pascal VOC 指南](PascalVOC/README_cn.md) |
| [yoloe/](yoloe/) | `yoloe_seg_pf_classes.names`——固定 4585 类 prompt-free 词表 | [YOLOE](../samples/vision/yoloe/README_cn.md) sample 及评估器 | [YOLOE 指南](yoloe/README_cn.md) |

<a id="boundaries"></a>
## 适用范围

- **这里只有 COCO 提供下载脚本。**
  [coco/download_full_coco.sh](coco/download_full_coco.sh) 准备 COCO 2017；
  ImageNet、DOTA 和 Pascal VOC 请从官方来源获取。
- **许可与条款按数据集各自约定。** 请在各自条款下获取数据集；各指南给出官方网站。
  随仓标签文件和示例图片只是用于离线冒烟检查的小片段，不是数据集的再分发。
- **类别索引 ≠ 数据集类别 ID。** 模型输出列是连续索引。COCO 标注文件使用稀疏的
  数字 ID；DOTA 原生标注记录的是类别**名称**，任何数字 DOTA ID 空间都来自定义它的
  转换，而不是格式本身。各数据集指南写明精确映射；[YOLOE 指南](yoloe/README_cn.md)
  说明独立的 4585 类 prompt-free 词表，其索引既不是 COCO category ID，也不是
  Ultralytics COCO-80 输出索引。

<a id="provenance"></a>
## 来源

各 Sample 的数据准备与输入约定见 `samples/` 下的对应指南。模型制品清单为 `docs/release/x5/models.yaml` 与 `docs/release/s/models.yaml`。
