[English](./README.md) | 简体中文

# YOLOE 数据集资源

本目录存放 [YOLOE sample](../../samples/vision/yoloe/README_cn.md) 使用的固定
prompt-free（PF）类别词表。这里**没有图片，也没有数据集下载**——YOLOE 评估用的
数据集是你另行准备的 COCO 格式标注集（见
[YOLOE 评估器指南](../../samples/vision/yoloe/evaluator/README_cn.md)）。

<a id="files"></a>
## 文件说明

- `yoloe_seg_pf_classes.names` — YOLOE-11/26 Seg Prompt-Free 模型的类别表：
  **4585 行，每行一个类名，按字母排序**，覆盖开放词汇实例分割词表（模型输出索引
  从 0 开始；第 1 行即索引 0）。

它与规范副本 `samples/vision/yoloe/test_data/classes.names` **逐字节一致**
（SHA-256
`1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3`），后者是
sample 转换和运行时校验的哈希基准。

<a id="usage"></a>
## 使用方式

规范的 [YOLOE Python 运行时](../../samples/vision/yoloe/runtime/python/README_cn.md)
默认 `--label-file` 指向自己的 `test_data/classes.names`；本文件是同一词表，
也可以显式传入。命令以**仓库根目录**为工作目录：

```bash
# 工作目录：仓库根目录；先按运行时指南准备对应制品
python3 samples/vision/yoloe/runtime/python/main.py \
  --target x5 --model-path samples/vision/yoloe/model/<prepared-artifact>.bin \
  --label-file datasets/yoloe/yoloe_seg_pf_classes.names
```

请替换为你准备好的模型路径；全部参数见运行时指南。传入任何其他词表文件都会
无法通过 sample 的哈希校验——绝不能修改这份 4585 项列表。

<a id="pf-vs-coco"></a>
## PF 类别索引不是 COCO category ID

这 4585 个索引是模型的开放词汇输出顺序，**既不是** COCO category ID，**也不是**
Ultralytics 80 类 COCO 输出顺序。文件中实测的位置示例：索引 2163 = `person`，
索引 821 = `chair`。评分时把 PF 索引映射到数据集类别是
[YOLOE 评估器](../../samples/vision/yoloe/evaluator/README_cn.md)中显式、带名称
校验的步骤（`mapping.example.json` 演示 person → COCO 1、chair → COCO 62，
**不是**完整 COCO-80 映射）。

运行时命名模型输出应使用 [datasets/coco](../coco/README_cn.md) 或 sample 自带
`test_data/` 中的 80 类文件；不能用这份 4585 项词表替代。

<a id="provenance"></a>
## 来源与生成

PF 导出和转换准备流程生成并核对词表：

- [conversion/export.py](../../samples/vision/yoloe/conversion/README_cn.md)
  导出本地 PF checkpoint，生成 `yoloe_<variant>_seg_pf.onnx`，并在旁边写一份
  词表副本 `yoloe_<variant>_seg_pf.names`；
- 随后转换 `prepare.py` 要求 `--names` 与
  `samples/vision/yoloe/test_data/classes.names` **逐字节一致**，否则不生成
  任何转换配置。

导出器生成的 `.names` 应视为派生副本；本目录和 sample 的
`test_data/classes.names` 才是固定基准。
