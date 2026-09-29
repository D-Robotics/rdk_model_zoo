[English](./README.md) | [简体中文](./README_cn.md)

# UNet 精度评测

`eval_unet.py` 是五个 UNet ResNet 变体的统一精度入口，可以使用同一份
Pascal VOC 图像与标签清单评测 PyTorch checkpoint、浮点 ONNX 或 RDK X5
`bayes-e` `.bin` 模型。

评测脚本会把 RGB 图像和类别索引 mask 缩放到 `512x512`，保留 VOC 忽略标签
`255`，对 21 类 logits 执行 `argmax`，并输出 mIoU、Pixel Accuracy 和每类
IoU。每次运行都会把模型和 manifest 哈希写入新的 JSON 报告。

<a id="dataset"></a>
## Manifest 格式

每个非空行包含一个绝对图像路径和一个 mask 路径，中间使用单个 Tab 分隔：

```text
/data/VOC2012/JPEGImages/2007_000033.jpg\t/data/VOC2012/SegmentationClass/2007_000033.png
```

VOC 调色板 mask 必须按类别索引读取，评测前不能转换为灰度图。

<a id="environment"></a>
## 环境

公共依赖为 Python 3.10+、NumPy、Pillow；PyTorch/ONNX 后端分别另需 torch/onnxruntime。X5 后端另需 OpenCV、PyYAML、配套 SDK，并校验实际 X5 身份与 OS 版本；它复用统一 runtime 三阶段。训练/ONNX 后端保留原始浮点输入规则。

| Parameter | Default | Meaning |
| --- | --- | --- |
| --model | required | .pth / .onnx / .bin |
| --manifest | required | image TAB mask absolute path pairs |
| --report | required | new JSON path; existing file rejected |
| --backend | auto | suffix detection or pytorch/onnx/x5 |
| --backbone | None | required for .pth or custom-named .bin |
| --limit | None | first N manifest entries; positive |
| --progress-every | 50 | progress interval |
| --min-miou | 0.0 | write report and return 2 if below threshold |

标准发布文件名可推导骨干；自编译自定义文件名须传 --backbone，和文件名冲突时拒绝。--model 是显式评估输入，不验证发布哈希；报告保存实际模型 digest，不能把这条路径当作已认证下载。

<a id="command"></a>
## PyTorch checkpoint

在装有 PyTorch 和 Pillow 的开发机上执行：

```bash
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
  --model /models/unet_resnet18_voc_best.pth \
  --backbone resnet18 \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_pytorch.json \
  --min-miou 0.50
```

## ONNX

在主机评测环境中安装 `onnxruntime` 后执行：

```bash
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
  --model /models/unet_resnet18_voc_512x512.onnx \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_onnx.json \
  --min-miou 0.50
```

## RDK X5 `.bin`

`.bin` 后端必须在 aarch64 RDK X5 本机运行，并且 RDK OS 不低于 3.5.0。
必须使用板卡系统随附的 X5 `hbm_runtime`，不要从 PyPI 安装其他平台的同名包。
编译模型必须提供一个 packed NV12 输入和一个 21 类 logits 输出。

```bash
# cwd: samples/vision/unet/evaluator
python3 eval_unet.py \
  --model /models/unet_resnet18_voc_512x512_nv12.bin \
  --manifest /data/unet/val.tsv \
  --report /reports/unet_resnet18_x5.json \
  --min-miou 0.50
```

可以用 `--limit` 执行小样本冒烟测试。如果结果低于 `--min-miou`，JSON 报告
仍会正常生成，但程序退出码为 2。

<a id="metrics"></a>
## 指标

mIoU 对 union>0 的类别平均；像素准确率为非 ignore 像素中预测正确比例；class_iou 给出 21 类 IoU。标签 255 忽略，图像与掩码均缩放至 512×512（RGB 双线性、类别 mask 最近邻）。与原图分辨率评测不可直接比较；limit 子集也不能替代完整 1449 张记录。

<a id="outputs"></a>
## 输出

新 JSON 保存后端、模型/manifest 哈希、样本数量和 input_contract；runtime 保存环境/模型 metadata，metrics 保存 miou、pixel_accuracy 和 class_iou；另有 min_miou、passed 和 elapsed_seconds。不序列化内部混淆矩阵。低于门槛仍写报告后退出 2，不能只看文件存在就认定成功。

<a id="reference-results"></a>
## 参考结果

保留的五骨干训练/PTQ 表及 ResNet18 三后端 1449 张记录见[样例根文档](../README_cn.md)。它们绑定原分支的 checkpoint/制品条件；本轮没有重新测量 mIoU、BPU 延迟或 FPS。主机合成数据测试不替代这些证据。

<a id="boundaries"></a>
## 边界

完整数据集、checkpoint、ONNX、BIN 需事先准备。当前未执行完整数据集评估或板测；运行时耗时不是本评估器提供的纯 BPU benchmark。源代码保留 PyTorch 严格权重加载和 ONNX 单输入/输出限制，不宣称任意新架构或 S 系列资产兼容。
