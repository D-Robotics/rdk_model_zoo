[English](./README.md) | 简体中文

# MODNet

<a id="overview"></a>
## 算法与来源

MODNet 是单阶段人像抠图网络：输入一张 RGB 图像即可输出 alpha matte，不需要 trimap。源工程为 [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)，论文为 [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)。本 sample 保留源的几何处理、RGB 归一化、uint8 matte 输出和可选背景合成。

<a id="support-matrix"></a>
## 支持与验证矩阵

| target | variant | Python | C++ | 状态 |
|---|---|---|---|---|
| X5 | `modnet_512x512_rgb.bin` | supported-not-run | not-supported | 主机 fixture 通过；板测未运行——2026-09-24 X5 板端批次没有拿到该手工制品，因此没有可记录的下载或推理 |
| S100/S100P/S600 | — | not-supported | not-supported | 没有源制品 |

本 sample 没有 C++ 实现。运行模型是外部手工制品；主机测试不等同于板端验证。

<a id="prerequisites"></a>
## 环境前提

板端运行需要带 `hbm_runtime` 的 RDK X5、NumPy 和 OpenCV。必须精确准备外部模型 `x5:modnet:modnet_512x512_rgb.bin`；active manifest 没有 URL，且 `sha256: null (unknown)`。模型契约为 float32 RGB NCHW `(1,3,512,512)` 输入和 float32 `(1,1,512,512)` matte 输出。

<a id="quickstart"></a>
## 快速体验

将外部模型放到 `samples/vision/modnet/model/modnet_512x512_rgb.bin`，然后在仓库根目录运行：

```bash
python3 -m samples.vision.modnet.runtime.python.main --target x5 \
  --asset-id x5:modnet:modnet_512x512_rgb.bin
```

命令读取 `test_data/person.jpg`，写出 `test_data/matte.png`；当 `test_data/bg.jpg` 存在时同时写出 `test_data/result.png`。成功判断为退出码 `0` 且 JSON 列出输出路径。该 manual 制品没有可用下载器；`model/download.py` 只打印精确准备要求并返回 `2`。

<a id="expected-results"></a>
## 预期结果

matte 是与原图高宽相同的 8-bit 灰度 PNG；可选合成图是保持原图几何的 BGR PNG。具体 alpha 值和质量取决于外部模型，本说明不新增板端结果；源历史性能在下方评估文档中标记为未复测。

<a id="directory"></a>
## 目录职责

```text
.
├── model/                 # 手工制品身份和准备说明
├── runtime/python/        # binding、懒加载 runner、task、CLI、run.sh
├── conversion/            # 源事实与缺失导出/PTQ 材料
├── evaluator/             # 自包含 matte 对照工具
├── test_data/             # 源 person.jpg 和 bg.jpg
└── tests/                 # 主机 metadata、几何、raw、CLI fixture
```

<a id="entry-points"></a>
## 入口索引

- [`model/README_cn.md`](./model/README_cn.md)：手工模型身份、路径和未知校验值。
- [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)：CLI 与 `MODNetTask` API。
- [`conversion/README_cn.md`](./conversion/README_cn.md)：源转换能力和真实缺口。
- [`evaluator/README_cn.md`](./evaluator/README_cn.md)：完整保存 matte 的对照步骤。

<a id="historical-performance"></a>
## 源历史性能

下表和测试条件完整保留源数据；这些是源历史测量，本迁移没有复测。

| 模型 | 尺寸 | 输入格式 | 延迟 (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet（2 threads） | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

条件：RDK X5、CPU 8xA55@1.8G、BPU 1xBayes-e@1G（10TOPS INT8）。单线程延迟使用单帧、单线程、单 BPU core；多线程 FPS 使用 2 个并发线程。

<a id="license"></a>
## 许可

迁移 wrapper 和仓库文件遵循 Apache-2.0。MODNet 源许可声明、论文和上游工程仍归其作者所有。外部 manual 模型没有随源提供模型许可或发布者 checksum。
