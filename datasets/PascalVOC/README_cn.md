[English](./README.md) | 简体中文

> 下文的 `platforms/` 路径指统一前历史目录，已于 2026-10-01 移出活动分支。请从固定提交 `d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d` 读取（如 `git show d2d2a4e0:<path>`，或临时 `git worktree add <dir> d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d`）；见 `docs/migration/2026-09-30-model-examples.md`。


# Pascal VOC 数据集资源

**PASCAL VOC**（2007/2012）是经典的目标检测与分割基准：20 个前景类别加背景
（分割任务共 21 类），包含 JPEG 图像、XML 框标注和调色板索引的分割掩码。本目录
**只有参考链接——没有任何随仓文件**，也没有下载脚本。需按官方条款从官方主机
手动获取。

<a id="files"></a>
## 目录内容

本目录不包含数据集、类别表或示例图片。本页与[英文指南](./README.md)保留交付
分支随附的源参考链接。

<a id="usage"></a>
## 本仓库中 Pascal VOC 的使用方

| 使用方 | 用途 |
| --- | --- |
| [UNet 评估器](../../samples/vision/unet/evaluator/README_cn.md) | 评估 **VOC 2012 分割**：清单为 `JPEGImages/<id>.jpg` + `SegmentationClass/<id>.png` 的路径对（制表符分隔、绝对路径）；21 类 logits，调色板索引即类别索引，`255` 为忽略的空白标签。掩码必须保持调色板索引格式——不能转成灰度图 |
| [UNet test_data](../../samples/vision/unet/test_data/README_cn.md) | 随仓 `2007_000033.jpg`，一张 VOC 2012 验证图，用于离线冒烟检查 |

当前没有 sample 做 VOC **检测**；上表分割路径的类别命名规则为：调色板索引 =
类别索引（`0` 背景、`1`–`20` 前景、`255` 忽略）。

<a id="acquisition"></a>
## 获取数据集

从官方主机手动下载，并把 UNet 评估器清单指向准备好的绝对路径，例如：

```text
/data/VOC2012/JPEGImages/2007_000033.jpg
/data/VOC2012/SegmentationClass/2007_000033.png
```

- 官方网站：<http://host.robots.ox.ac.uk/pascal/VOC/>
- VOC 2012 工具包：<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
  （[UNet test_data 指南](../../samples/vision/unet/test_data/README_cn.md)
  引用同一链接）
- 入门文章（中文，继承自源 README）：
  <https://blog.csdn.net/generalsong/article/details/108471378>

请遵守数据集自身使用条款；图像与标注版权归原权利人所有。

<a id="provenance"></a>
## 来源

X5 交付分支（`ac11571`）随附的本目录只有同样两条链接（英文 README 近乎空白，
中文 README 很短）；S 交付分支（`380e1a2`）完全没有 `PascalVOC/` 目录。本指南
在该继承内容上补充了当前使用方映射。归档副本位于
`platforms/x5/datasets/PascalVOC/`。
