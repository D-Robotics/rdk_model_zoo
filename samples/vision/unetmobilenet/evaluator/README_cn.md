[English](README.md) | 简体中文

# UNetMobileNet 验证

<a id="dataset"></a>
## 数据集

Cityscapes 定义 19 类任务，但源未提供带标签验证 split 或数据集执行器。segmentation.png 为冒烟输入，result.jpg 为历史示意。数据集评估需要有许可的图片／标签、明确 train-ID 映射、忽略标签策略及记录的 split。

<a id="environment"></a>
## 环境

主机套件需要 Python 3.10+、NumPy/OpenCV/PyYAML，以及执行原生纯数值／假接口测试的 C++17 编译器。真实 runtime 检查需要运行时指南所列匹配 S100/S600 镜像、模型和 SDK。原生测试目录中的假头文件不能代替 SDK 安装，也不是实际 SDK 编译证据。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; host regression (no model/SDK required)
python3 -m unittest discover -s samples/vision/unetmobilenet/tests
# On S100, after explicit preparation; single-image result, not dataset accuracy
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 --mask-save-path outputs/unetmobilenet/python.npy --report-path outputs/unetmobilenet/python.json
bash samples/vision/unetmobilenet/runtime/cpp/run.sh --target s100 --build --mask-save-path outputs/unetmobilenet/native.png --report-path outputs/unetmobilenet/native.json
```

两侧 runtime 必须使用相同目标、制品和输入。检查 S600 时，各对应步骤均使用 --target s600。这些板端命令供后续验证，本轮迁移未执行。

<a id="metrics"></a>
## 指标

主机检查验证 fixture 的源预处理／绘图一致性、类别解码、逐通道 SCALE 反例、直接最近邻恢复、目标拒绝及注入失败时的资源释放，不证明 mIoU 或延迟。把 Python NPY 与 C++ PNG 读取为整数数组比较；逐像素完全一致可作为同制品冒烟判据。不要对类别 ID 计算 logits 余弦相似度。

<a id="outputs"></a>
## 输出

Python 输出 int32 NPY；C++ 输出无损 uint8 PNG ID，API mask 仍为 int32。两侧均保留原图尺寸，输出元数据报告和叠加图。应在可视化前比较数值 mask；JPEG 压缩不适合叠加图的逐像素精确比较。

<a id="reference-results"></a>
## 参考结果

本示例源中没有 mIoU/FPS/延迟表。[历史效果图](../test_data/result.jpg) 保留，但不声称新测量结果。[源审计](../../../../docs/releases/unified-migration/evidence/2026-09-26-b8-unetmobilenet-audit.json) 记录了对整数直接 argmax 和原生中间 resize 的明确修正，不从主机测试推定板端验收通过。

<a id="boundaries"></a>
## 边界

这里没有数据集评估循环、训练源模型、真实 SDK 编译、板端推理或性能运行证据；独立评审仍待执行。实现明确拒绝缺失／不支持的整数量化元数据，不假定通道排序。原生任务测试中的假接口仅验证失败分支，实际 SDK 兼容性仍需板端／工具链环境。
