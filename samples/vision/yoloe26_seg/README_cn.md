[English](./README.md) | 简体中文

# YOLOE-26 实例分割（Prompt-Free）

面向 **S100 和 S100P** 的开放词表目标检测与实例分割示例。
支持 n/s/m/l/x 五种规格，统一使用 batch=1、640×640 NV12 输入和完整的
4585 类 PF 词表。不支持文本提示、视觉提示、动态形状和 S600。

## 快速开始

在开发板上进入本示例目录后执行：

```bash
# Python：自动识别板型，下载匹配的 n 模型，并处理内置图片。
bash runtime/python/run.sh --size n --output result.jpg

# C++：下载、编译并运行。可将 n 替换为 s、m、l 或 x。
bash runtime/cpp/run.sh n
```

示例不会自动安装依赖或修改系统服务。请使用板端已安装的 UCP/DNN SDK
和 `hbm_runtime`。下载文件会按照发布清单进行 SHA256 校验。

## 模型支持

| 板型 | March | 已发布规格 | 编译配置 | 板端验证 |
|---|---|---|---|---|
| S100 | nash-e | n/s/m/l/x | OE 3.7.0，INT8 KL | 2026-09-08 已完成 Python/C++ 测试 |
| S100P | nash-m | n/s/m/l/x | OE 3.7.0，INT8 KL | n：2026-09-17 已完成 Python/C++ 测试；s/m/l/x 待验证 |

两平台模型均提供十个 NHWC 输出。分类、框和 mask 系数输出为 INT32，proto 为 INT8；
CPU 后处理根据实际量化参数反量化，并正确处理张量 padding。
框回归使用 reg_max=1，不执行 DFL；端到端候选筛选采用 top-k，不执行 NMS。

目前尚未通过数据集精度验收。这些模型是 PTQ 基线，并非经过精度调优的生产模型；
Python/C++ 结果一致不代表量化前后 mAP 不变。

## 目录结构

```text
conversion/       ONNX 导出及 S100/S100P 校准、编译
model/            已发布 HBM 的下载入口
runtime/python/   Python 图片推理
runtime/cpp/      C++ UCP 图片推理
evaluator/        Runtime 性能结果与评测说明
test_data/        示例图片、词表及效果图
```

- [模型下载](model/README_cn.md)
- [模型转换](conversion/README_cn.md)
- [Python 推理](runtime/python/README_cn.md)
- [C++ 推理](runtime/cpp/README_cn.md)
- [Runtime 性能](evaluator/README_cn.md)

![S100 YOLOE-26n PF 示例](test_data/result.jpg)

上图使用已发布 S100 n 模型的实测输出生成，并非数据集精度评测。
标签按照 PF 检查点中的类别 ID 顺序导出。
