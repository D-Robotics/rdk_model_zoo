[English](./README.md) | [简体中文](./README_cn.md)

# UNet Python Runtime

本 RDK X5 示例读取一张 OpenCV BGR 图片，将其转换为 packed NV12，在 BPU 上
执行 UNet，并保存类别索引 mask、彩色叠加图和机器可读 JSON 报告。

## 环境依赖

- RDK X5，RDK OS 3.5.0 或更新版本
- Python 3.10 或更新版本
- 板卡系统随附的 `hbm_runtime`
- OpenCV 和 NumPy
- 由 [`conversion/mapper.py`](../../conversion/mapper.py) 生成的 `bayes-e`
  UNet BIN

在板端检查 Runtime，并仅安装缺失的通用依赖：

```bash
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
sudo apt update
sudo apt install -y python3-opencv python3-numpy
```

不要安装为其他平台构建的同名 `hbm_runtime` wheel。

## 目录结构

```text
python/
├── unet.py       # UNetConfig 和可复用 UNet 推理封装
├── main.py       # CLI、可视化与 JSON 报告入口
├── run.sh        # 一键运行脚本
├── README.md
└── README_cn.md
```

## 默认资源

代码已经定义符合仓库规范的默认路径：

- 模型：`../../model/unet_resnet18_voc_512x512_nv12.bin`
- 图片：`../../test_data/2007_000033.jpg`

示例图片已包含在仓库中。未指定 `--model-path` 且默认模型不存在时，`run.sh` 会先
自动调用 `../../model/download_model.sh resnet18`，然后开始推理。

## 命令行参数

执行 `python3 main.py --help` 可以查看当前接口。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | X5 UNet BIN 模型路径 | `../../model/unet_resnet18_voc_512x512_nv12.bin` |
| `--test-img` | OpenCV 可读取的输入图片 | `../../test_data/2007_000033.jpg` |
| `--mask-save-path` | 原始 uint8 类别索引 PNG | `unet_mask.png` |
| `--img-save-path` | 彩色语义分割叠加图 | `unet_result.png` |
| `--report-path` | Runtime 元数据与结果 JSON | `unet_runtime_report.json` |
| `--priority` | 可选 BPU 调度优先级 | 未设置 |
| `--bpu-core` | 可选 BPU 核编号 | 未设置 |
| `--alpha` | 叠加图中分割颜色的权重 | `0.55` |

## 快速运行

### 使用默认路径

默认命令会在需要时自动下载 ResNet18，然后执行推理：

```bash
cd samples/vision/unet/runtime/python
./run.sh
```

### 显式指定路径

```bash
cd samples/vision/unet/runtime/python
./run.sh \
  --model-path /path/to/unet_resnet18_voc_512x512_nv12.bin \
  --test-img /path/to/image.jpg \
  --mask-save-path unet_mask.png \
  --img-save-path unet_result.png \
  --report-path unet_runtime_report.json
```

调度参数是可选的。不传入 `--priority` 和 `--bpu-core` 时，不修改板端 Runtime
的默认调度行为。

显式传入 `--model-path` 会关闭默认模型自动下载；指定的模型文件必须已经存在。

## 输出文件

| 输出 | 内容 |
| --- | --- |
| `unet_mask.png` | `[512, 512]` 的 uint8 Pascal VOC 类别 ID `0..20` |
| `unet_result.png` | VOC 彩色 mask 与缩放后原图的叠加结果 |
| `unet_runtime_report.json` | Runtime 版本、模型 I/O、出现类别、耗时与输出路径 |

Runtime 要求 BIN 中只有一个模型、一个 NV12 输入和一个语义 logits 输出。不支持
的图片类型或不兼容的模型合同会直接抛出异常，不会静默生成结果。

## Python 接口

`unet.py` 遵循仓库规定的 Config/Model 接口：

| 接口 | 职责 |
| --- | --- |
| `UNetConfig` | 保存模型路径、输入尺寸和类别数 |
| `UNet.__init__` | 加载 BIN，提取并校验模型元数据 |
| `set_scheduling_params` | 设置可选优先级或 BPU 核；全部为空时无副作用 |
| `pre_process` | 缩放 BGR uint8 输入，返回可直接传给 Runtime 的 NV12 字典 |
| `forward` | 返回 `HB_HBMRuntime.run` 的直接输出 |
| `post_process` | 必要时反量化，并返回 uint8 类别索引 mask |
| `predict` / `__call__` | 串联预处理、推理和后处理 |

生成接口文档的方法见[源码文档说明](../../../../../docs/source_reference/README.md)。

## 注意事项

- 输入必须是 OpenCV BGR uint8；其他 layout 或 dtype 会被拒绝。
- 输出 mask 保持模型分辨率，不会自动缩放回原图尺寸。
- 纯 BPU 性能应使用 `hrt_model_exec` 测量。JSON 中的耗时还包含 Python
  预处理、推理输出回读和后处理。
