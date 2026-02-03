# COCO Dataset Resources

**COCO（Common Objects in Context）** 是目前计算机视觉领域最常用、最具代表性的公开数据集之一，主要用于：

- 目标检测（Object Detection）
- 实例分割（Instance Segmentation）
- 关键点检测（Keypoint Detection）
- 图像理解与多任务学习

COCO 数据集以**复杂场景、丰富类别和真实上下文关系**为特点，广泛用于学术研究、工程实践以及模型基准评测。
当前目录提供了 **小样本示例数据** 以及 **完整 COCO 数据集的自动化下载脚本**，用于开发、调试与验证。



## 1. 目录结构说明

```text
coco/
├── README.md                        # 本说明文档
├── download_full_coco.sh            # 下载完整 COCO 数据集脚本
└── coco_classes.names               # COCO 类别标签文件（80 类）
```

### 1.1 coco_classes.names

coco_classes.names 文件包含 COCO 数据集的 类别名称列表（通常为 80 类），顺序与模型输出的 class_id 对应，例如：

```text
person
bicycle
car
...
```

## 2. 完整 COCO 数据集下载

本目录提供 download_full_coco.sh 脚本，用于自动下载并整理 官方 COCO 数据集。

### 2.1 支持的数据内容

通常包含以下数据（以 COCO 2017 为例）：

- 训练集：train2017

- 验证集：val2017

- 标注文件：

    instances_train2017.json

    instances_val2017.json

### 2.2 下载步骤

在 coco/ 目录下执行：

```bash
./download_full_coco.sh
```

脚本将自动完成：

- 从 COCO 官方源下载数据压缩包

- 解压并整理目录结构


## 3. 数据集官方说明

官方网站：https://cocodataset.org

数据集版权与使用条款请参考 COCO 官方说明

## 4. 注意事项

若下载速度较慢，可自行修改脚本中的下载源
