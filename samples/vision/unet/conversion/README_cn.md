[English](./README.md) | [简体中文](./README_cn.md)

# RDK X5 UNet 模型转换

本目录只保留五个 UNet ResNet 变体从浮点 checkpoint 到 X5 的转换链路；训练
代码继续维护在 Model Zoo 仓库之外。

## 目录

```text
conversion/
├── mapper.py
├── onnx_export/
│   ├── export_unet.py
│   └── model/
└── ptq_yamls/
```

`onnx_export/model` 是训练、评测和导出共同使用的唯一 PyTorch 模型源码。
`ptq_yamls` 为每个 backbone 保存一份经过审阅的 `bayes-e` 模板；`mapper.py`
把模板绑定到本次 ONNX、校准集和全新输出目录，再执行 checker、makertbin 与
`hb_model_info`。

<a id="source-model"></a>
## 源模型

五个骨干共用本地 onnx_export/model 架构；来源、固定参考 commit 和 MIT 许可见[根 README](../README_cn.md#license)。需要与所选骨干匹配的训练 checkpoint，本仓不下载或捏造它。严格载入失败不能用 strict=False 绕过。

<a id="toolchain-targets"></a>
## 工具链

目标为 X5 / bayes-e。源未固定 OE Docker 版本；需 x86 Linux 的 hb_mapper、hb_model_info，mapper 会记录实际版本。导出环境另需 PyTorch、ONNX、ONNX Runtime、NumPy；校准示例需 Pillow；mapper 需 PyYAML。本轮未安装/执行 OE，也未声明任一未实测组合可复现。

<a id="export"></a>
## 1. 导出 ONNX

checkpoint 必须与所选 backbone 一致。导出器会严格加载权重，生成固定 shape
的 opset 11 ONNX，执行 ONNX checker，并使用同一份确定性输入比较 PyTorch 与
ONNX Runtime。已有 ONNX 和报告不会被覆盖。

```bash
# cwd: samples/vision/unet/conversion
python3 onnx_export/export_unet.py \
  --backbone resnet18 \
  --checkpoint /models/unet_resnet18_voc_best.pth \
  --output /models/unet_resnet18_voc_512x512.onnx
```

只有数值比较通过的导出报告才能进入 `mapper.py`。`--skip-runtime-check` 仅用于
结构预检，生成的报告会明确标记为不能进入 X5 PTQ。

<a id="calibration"></a>
## 2. 准备校准张量

建议选取约 100 张有代表性的 Pascal VOC 训练图像。每个校准文件必须是无文件
头的小端 float32 `.bin`，保存一份 shape 为 `[3, 512, 512]` 的 RGB CHW 张量，
数值范围为 `[0, 255]`。数据脚本不要除以 255；PTQ YAML 通过
`data_scale=1/255` 统一承担归一化，板端 NV12 输入也使用同一规则。

`mapper.py` 会读取全部张量，拒绝文件大小错误、NaN/Inf 和越界值，并在本次
运行的 reports 目录生成带哈希的 `calibration-manifest.json`。

```python
# cwd: samples/vision/unet/conversion; prepare a representative VOC image folder first
from pathlib import Path
import numpy as np
from PIL import Image

images = sorted(Path("/data/VOC2012/JPEGImages").glob("*.jpg"))[:100]
if not images:
    raise ValueError("No calibration images found")
out = Path("/data/unet/calibration_data_rgb_f32_512")
out.mkdir(parents=True, exist_ok=False)
for source in images:
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype="<f4")
    with (out / (source.stem + ".bin")).open("xb") as stream:
        stream.write(chw.tobytes())
print(len(images), "calibration tensors", out)
```

<a id="compile"></a>
## 3. 编译 X5 模型

在同时提供 `hb_mapper` 与 `hb_model_info` 的 OpenExplorer Mapper 环境执行；
`--output` 指定的目录必须不存在。

```bash
# cwd: samples/vision/unet/conversion
python3 mapper.py \
  --backbone resnet18 \
  --onnx /models/unet_resnet18_voc_512x512.onnx \
  --calibration /data/unet/calibration_data_rgb_f32_512 \
  --output /output/unet_resnet18_x5_run_001
```

受门禁保护的执行顺序如下：

```text
导出报告 → 校准审计 → hb_mapper checker → hb_mapper makertbin
         → 唯一 .bin → hb_model_info 确认 BPU march: bayes-e
```

运行目录会保留解析后的 YAML、checker/build/model-info 日志、校准 manifest、
复制出的制品、哈希、工具版本与 `run-receipt.json`。编译成功后仍需使用
`../evaluator/eval_unet.py` 完成精度评测和板端 Runtime 验证。

<a id="validation"></a>
## 验证

导出器默认执行确定性输入的 PyTorch/ORT 数值比较，skip-runtime-check 只允许结构预检，不能通过后续 PTQ gate。编译后用 evaluator 对同一 VOC manifest 测量精度；自编译模型通过其显式 --model 运行，不冒充发布哈希制品。

```bash
# cwd: repository root; on X5 after a real successful compile, with a prepared VOC manifest
python3 samples/vision/unet/evaluator/eval_unet.py --backend x5 --backbone resnet18 --model /output/unet_resnet18_x5_run_001/artifacts/unet_resnet18_voc_512x512_nv12.bin --manifest /data/unet/val.tsv --report /reports/unet_custom_x5.json --min-miou 0.50
```

编译输出文件位置以实际 run-receipt.json 为准，上方 --model 需替换为 receipt 记录的 BIN。当前统一入口板测、完整导出、编译和数据集评估均 not-run。

<a id="artifacts"></a>
## 制品

输出保留 ONNX/report、已解析 YAML、校准 manifest、checker/build/model-info 日志、制品与哈希、run-receipt.json。五个发布 BIN 名称和 SHA-256 见[模型表](../model/README_cn.md#artifacts)，不能把发布哈希赋给重新编译的文件。

<a id="known-gaps"></a>
## 缺失前提

训练 checkpoint、完整 VOC 及代表性校准子集需自行准备；源没有固定 OE 镜像/框架版本。本轮只校验主机逻辑、入口与文档，未补造转换或精度证据。上面的取前 100 张是格式准备示例，不保证校准代表性，正式量化需选择适合数据分布的子集。
