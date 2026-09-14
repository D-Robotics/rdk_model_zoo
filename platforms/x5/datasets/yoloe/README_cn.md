简体中文 | [English](./README.md)

# YOLOE 数据集资源

本目录包含 YOLOE sample 使用的数据集相关资源。

## 文件说明

- `yoloe_seg_pf_classes.names` — YOLOE-11 Seg Prompt-Free 模型的类别名称文件，包含 4585 个类名，用于开放词汇实例分割。

## 使用方式

类别名称文件由 YOLOE Python 运行时引用，用于可视化检测到的实例：

```bash
python3 main.py --label-file ../../../../../datasets/yoloe/yoloe_seg_pf_classes.names
```

## 说明

- 4585 个类别覆盖了开放词汇检测与分割任务的广泛词汇表。
- 该文件由 ONNX 导出脚本（`conversion/onnx_export/export_yoloe11seg_bpu.py`）在模型转换过程中生成。
