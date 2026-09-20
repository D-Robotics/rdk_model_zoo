# RDK Model Zoo

[English](README.md) | 简体中文

RDK Model Zoo 提供在地瓜机器人 BPU 上部署模型的示例。每个 Sample 包含模型准备、板端推理、源码说明，以及已有的模型转换和评估流程。运行模型使用 Python 与板卡随系统提供的 Runtime，不依赖 Agent 或 Node.js。

## 从一个样例开始

本轮整合三类代表样例。点击任务进入完整使用说明；表中的验证只针对列出的模型和固定输入，不代表同系列所有规格都已测试。

| 任务 | Sample | 代表模型与板卡 | 开发入口 |
| --- | --- | --- | --- |
| 目标检测 | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README_cn.md) | YOLOv8n、YOLO26n；X5 8GB/4GB、S100、S100P、S600 | [转换](samples/vision/ultralytics_yolo/conversion/README_cn.md) · [评估](samples/vision/ultralytics_yolo/evaluator/README_cn.md) |
| 图像分类 | [ResNet](samples/vision/resnet/README_cn.md) | ResNet18；X5 8GB/4GB、S100、S600 | [转换](samples/vision/resnet/conversion/README_cn.md) · [评估](samples/vision/resnet/evaluator/README_cn.md) |
| 文字检测与识别 | [PaddleOCR](samples/vision/paddle_ocr/README_cn.md) | X5 PP-OCRv3、S100 PP-OCRv6；检测与识别串联 | [转换](samples/vision/paddle_ocr/conversion/README_cn.md) · [评估](samples/vision/paddle_ocr/evaluator/README_cn.md) |

2026-09-17 的源码修订已复测 X5 两块板、S100 和 S100P 的适用模型。S600 的对照结果来自 2026-09-16，当前连接尚未恢复，新修订复测未执行。

在板卡上保留完整仓库。下面的命令从仓库根目录执行，先查看模型与参数，不加载 BPU 或下载模型：

```bash
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --help
python3 samples/vision/resnet/runtime/python/main.py --list-models --target x5
python3 samples/vision/paddle_ocr/runtime/python/main.py --list-models --target x5
```

选择样例后，按其 README 安装用户态依赖、准备对应板卡的模型，再执行推理。X5 的 `.bin` 与 S 系列的 `.hbm` 不能互换；S100、S100P、S600 也必须选择各自制品。不要用通用 PyPI 包替换板卡 Runtime。

## 阅读和修改代码

```text
samples/
├── _shared/                  # 平台识别、资产读取、共用图像字节转换
└── vision/
    ├── ultralytics_yolo/      # 检测流程与 DFL / LTRB 解码
    ├── resnet/               # 分类流程与 Top-K 结果
    └── paddle_ocr/           # 检测 → 裁剪 → 识别 → CTC 解码
```

每个 Sample 的 `runtime/python/main.py` 是命令入口；任务模块维护推理流程，`model_runner.py` 封装模型调用，`model_binding.py` 检查制品及输入输出约定。`conversion/` 在开发主机上生成模型，`evaluator/` 评估结果，`test_data/` 保存最小输入。具体调用示例和文件职责见各 Sample README。

同一算法在一处维护，必要的模型差异留在局部适配中。例如 OCR 两代模型使用不同词典，YOLO DFL 和直接 LTRB 使用不同解码，不因目录合并而混用。

## 平台源码、数据与历史版本

未纳入本轮的模型仍从平台目录使用。X3 保留历史内容，不参与此次 X5/S 整合。

| 平台 | 源码与使用说明 | 发布事实 |
| --- | --- | --- |
| X5 | [平台 README](platforms/x5/README_cn.md) | [模型与 Benchmark 清单](platforms/x5/docs/release) |
| S100 / S100P / S600 | [平台 README](platforms/s/README_cn.md) | [模型与 Benchmark 清单](platforms/s/docs/release) |
| X3（历史） | [平台 README](platforms/x3/README_cn.md) | [历史清单](platforms/x3/release) |

数据集准备仍位于 [X5 datasets](platforms/x5/datasets) 和 [S datasets](platforms/s/datasets)。现有清单继续保存模型 URL、身份和历史测量；合并源码不修改已发布资产的含义。

历史 Tag 保留当时的仓库布局，旧版本文档应与对应 Tag 一起阅读。[平台登记](platforms/README.md)说明历史分支；本地整合不代表已切换默认分支或发布新版本。

静态 Model Zoo 网页维护在 [`model_zoo_web`](model_zoo_web)，经审核的 Sample 模型数据可通过其中的 Catalog 构建流程导入。

## 许可证 (License)

各平台发行版各自携带许可证文件——见 [`platforms/x5/LICENSE`](./platforms/x5/LICENSE) 与 [`platforms/s/LICENSE`](./platforms/s/LICENSE)。上游 X3 未发布许可证文件，本次迁移也未新增。
