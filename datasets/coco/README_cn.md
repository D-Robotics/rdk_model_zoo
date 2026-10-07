[English](./README.md) | 简体中文

# COCO 数据集资源

**COCO（Common Objects in Context）** 是计算机视觉领域最常用的公开数据集之一，
覆盖目标检测、实例分割、关键点检测和图像理解等任务。它以复杂场景、丰富类别和
真实上下文关系著称，是模型基准评测的标准数据集。本目录提供小样本离线示例和
完整数据集下载脚本，用于开发、调试与验证。

数据集本身**不进 Git**；获取需手动完成，并遵守官方条款。

<a id="files"></a>
## 随仓文件

```text
coco/
├── README.md                  # 英文指南
├── README_cn.md               # 本指南
├── download_full_coco.sh      # 完整 COCO 2017 下载脚本
├── coco_classes.names         # 80 个类别名，每行一个
└── assets/
    ├── bus.jpg                # 示例图片（与 sample test_data 副本逐字节一致）
    └── kite.jpg               # 示例图片（与 YOLOv5 sample 副本逐字节一致）
```

### coco_classes.names

共 80 项（已按文件实测），每行一个类别名。行位置即模型输出类别索引（0–79）。
索引顺序与标准 Ultralytics COCO-80 输出顺序一致；其中 6 个显示名沿用 VOC 风格
同义词，与 COCO 规范名不同：

| 索引 | 本文件 | COCO 规范名 |
| --- | --- | --- |
| 3 | `motorbike` | motorcycle |
| 4 | `aeroplane` | airplane |
| 57 | `sofa` | couch |
| 58 | `pottedplant` | potted plant |
| 60 | `diningtable` | dining table |
| 62 | `tvmonitor` | tv |

### assets/

`bus.jpg`、`kite.jpg` 是随仓的两张示例图片，与下列默认测试图逐字节一致：
[Ultralytics YOLO](../../samples/vision/ultralytics_yolo/README_cn.md)
（`test_data/bus.jpg`）与
[YOLOv5](../../samples/vision/yolov5/README_cn.md)（X5 用 `test_data/bus.jpg`，
S 系列用 `test_data/kite.jpg`）。它们支持无数据集下载的离线运行冒烟检查。

### 类别索引与 COCO category ID 的区别

评估器正确工作依赖这一区别：

- **模型输出索引**：连续的 0–79，按 `coco_classes.names` 行序。
- **COCO 标注 `category_id`**：稀疏 ID 1–90，其中 10 个 ID（12、26、29、30、45、
  66、68、69、71、83）不存在。索引 0 → category 1（person），索引 79 →
  category 90（toothbrush）。

[Ultralytics YOLO 评估器](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md)
按标准索引→ID 映射（`eval_common.py` 的 `COCO_CATEGORY_IDS`）计分，不受标注文件
类别子集影响。绝不能把 `coco_classes.names` 行号当作 COCO JSON 的 `category_id`。

<a id="download"></a>
## 完整 COCO 下载脚本

[download_full_coco.sh](download_full_coco.sh) 下载并解压官方 COCO 2017 图像与
标注。以下按仓库中脚本现状描述；运行前自行确认目标路径与磁盘容量，下载的
数据不入库。

脚本行为（读自脚本本身）：

- 用 `wget -c --no-check-certificate` 从 `images.cocodataset.org` 下载三个压缩包：
  `zips/train2017.zip`、`zips/val2017.zip`、`annotations/annotations_trainval2017.zip`。
  `wget -c` 支持断点续传，重跑可继续。
- **输出目录是相对当前工作目录的 `coco_full/`**，不是固定位置。在希望存放数据的
  目录里执行，例如 `datasets/coco/`。脚本入仓时没有可执行位，请通过 `bash` 调用：

  ```bash
  # 工作目录：/data/coco
  bash /path/to/rdk_model_zoo/datasets/coco/download_full_coco.sh
  ```

- 在 `coco_full/` 内用 `unzip -q` 逐个解压，然后删除三个 `.zip`。依赖 `bash`、
  `wget`、`unzip`。
- 结果布局（train2017 共 118,287 张图，val2017 共 5,000 张；标注压缩包含两个
  划分的 instances/captions/person_keypoints JSON）：

  ```text
  coco_full/
  ├── train2017/            # 训练图像
  ├── val2017/              # 验证图像
  └── annotations/
      ├── instances_train2017.json
      ├── instances_val2017.json
      ├── captions_train2017.json
      ├── captions_val2017.json
      ├── person_keypoints_train2017.json
      └── person_keypoints_val2017.json
  ```

- 不下载 `test2017`。主要成本是带宽；官方源较慢时可修改脚本 URL 指向镜像，
  使用前请自行核验镜像完整性。

下载数据应存放在仓库外。脚本把 `coco_full/` 写入当前工作目录；上例从 `/data/coco`
运行。

<a id="usage"></a>
## 这些资源被谁使用

| 使用方 | 用途 |
| --- | --- |
| [Ultralytics YOLO 评估器](../../samples/vision/ultralytics_yolo/evaluator/README_cn.md) | 检测/分割使用 `instances_val2017.json`；姿态使用 `person_keypoints_val2017.json`；命令在仓库根目录执行，图像/标注路径由你准备 |
| [YOLOE 评估器](../../samples/vision/yoloe/evaluator/README_cn.md) | 任何 COCO 格式实例数据集（images + categories + annotations），不限于 COCO 本身 |
| [YOLOv5 sample](../../samples/vision/yolov5/README_cn.md) | `assets/bus.jpg` / `assets/kite.jpg` 作为默认测试图 |
| 分类 sample | 不使用本目录；它们使用 [ImageNet](../imagenet/README_cn.md) 标签 |

DOTA 旋转框评估见 [dotav1](../dotav1/README_cn.md)；当前 OBB 评估器只导出预测，
不对 DOTA 计算 COCO 风格 AP。

<a id="reference"></a>
## 官方网站与条款

- 官方网站：<https://cocodataset.org>
- 图像来自 Flickr，各自遵循其使用条款；标注以 Creative Commons Attribution 4.0
  发布。再分发或发布前请在官网确认最新条款。

数据集和标注来源：[COCO](https://cocodataset.org/)。下载脚本准备 COCO 2017 train/val 图像及标注；随仓显示标签采用模型连续的 80 类输出顺序。
