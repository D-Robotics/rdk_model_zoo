[English](README.md) | 简体中文

# OSNet 行人重识别

<a id="overview"></a>
## 模型与来源

OSNet 使用多尺度卷积块和通道门控，从已裁剪的人体图像提取 512 维外观特征。本 Sample 使用官方 x1.0 Market1501 权重，提供 X5 量化版本及 S100/S100P/S600 独立编译并实测的 HBM；X5 默认 spatial-rgb-late8。来源：[官方仓库](https://github.com/KaiyangZhou/deep-person-reid)、[ICCV 论文](https://arxiv.org/abs/1905.00953)、[官方权重](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html)。

<a id="directory"></a>
## 目录

```text
osnet/
├── model/          # 制品准备与身份记录
├── runtime/        # Python 特征提取与 C++ 验证
├── conversion/     # 导出、校准及编译配置
├── evaluator/      # Market1501 评测与原始证据
├── test_data/      # 私有准备的固定输入
├── tests/          # 主机行为测试
└── third_party/    # 固定上游源码与 MIT 许可证
```

<a id="support-matrix"></a>
## 支持矩阵

| Target | Variant | Language | Artifact |
| --- | --- | --- | --- |
| x5 | spatial-rgb-late8 | Python / C++ | Archive（仅供研究） |
| x5 | spatial-rgb-int16 | Python / C++ | available |
| x5 | spatial-rgb | Python / C++ | available |
| s100 | s100-f32-int16 | Python / C++ | Archive（仅供研究） |
| s100p | s100p-f32-int16 | Python / C++ | Archive（仅供研究） |
| s600 | s600-f32-int16 | Python / C++ | Archive（仅供研究） |
S100/S100P/S600 分别使用 nash-e/nash-m/nash-p HBM，各自完成全量精度和性能实测；不能混用各平台制品。上游有 x0.25/x0.5/x0.75 权重（256×128 下约 0.2/0.6/1.3 M 参数，0.08/0.27/0.57 GFLOPs）。x1.0 约 2.2 M / 0.98 GFLOPs。本 Sample 未提供其他宽度或 IBN/AIN 的 RDK 制品。


本轮精确统计为 2,169,508 个推理参数；Conv/Linear 的每 MAC 按两次运算计，得到 1.957750784 GFLOPs（0.978875392 GMAC）。与上游约 0.98 的口径差异来自 MAC 计数方式，模型仍为 x1.0。

<a id="prerequisites"></a>
## 环境前提

使用提供匹配 hbm_runtime、libdnn 1.24.5、HBRT 3.15.55.0 的 X5 镜像，这是制品记录中的环境；原实验未记录系统镜像版本。Python ≥3.10、NumPy、Pillow ≥9.1 为必需依赖，转换另需 PyYAML。回读包、BIN 和固定输入预留约 40 MB。使用已有环境，脚本不安装依赖。

以下命令均从仓库根目录执行。


S 系列板端已验证 UCP/DNN 3.13.6、HBRT 4.7.5；使用板卡自带 hbm_runtime、NumPy 和 Pillow（S100 的 Pillow 9.0.1 亦可运行，主机导出依赖见 requirements-host.txt）。S100/S100P 系统为 4.0.5-Beta，S600 为 5.1.0。

<a id="quickstart"></a>
## 快速运行

在仓库根目录执行。模型已上传 Archive，可按固定 SHA-256 下载；另准备一张有权使用的行人裁剪图。下载仅包含模型。以下以 X5 为例：

```bash
python3 samples/vision/osnet/model/download.py --target x5 --download
python3 samples/vision/osnet/runtime/python/main.py --target x5 \
  --variant spatial-rgb-late8 --test-img /absolute/path/person_crop.jpg --output feature.npy
python3 -c 'import numpy as np; f=np.load("feature.npy"); print(f.shape, f.dtype, np.linalg.norm(f))'
```

S100/S100P/S600 的模型准备使用对应 `--target`；运行时同时指定 `--variant <target>-f32-int16`。例如：

```bash
python3 samples/vision/osnet/model/download.py --target s100 --download
python3 samples/vision/osnet/runtime/python/main.py --target s100 \
  --variant s100-f32-int16 --test-img /absolute/path/person_crop.jpg --output feature.npy
```

<a id="expected-results"></a>
## 预期结果

已有交付包可用 `--candidate-dir /absolute/path/accepted-delivery` 代替 `--download` 离线准备。资产上传回执见[此处](evaluator/evidence/2026-11-02-archive-upload.json)；本地源码整合和研究用途目录登记已获用户放行，远端推送与站点上线待执行。

feature.npy 为独立拥有内存的 float32 (512,) 数组，默认 L2 范数约 1；--raw 保留原始值。人体检测与跟踪关联由应用负责。


2026-10-30 本地验收：默认 late8 的 Market1501 全量 Rank-1 为 94.0618%，mAP 为 83.3643%；新测 FP32 ONNX 为 94.2399% / 83.5696%。Runtime 1/2 并发平均延迟 8.189 / 15.048 ms，吞吐量 121.936 / 132.422 FPS；C++ 1/2 流端到端平均延迟 8.768 / 15.071 ms，吞吐量 114.040 / 132.530 FPS。每种条件 3 轮 × 200 帧（C++ 按每流计），预热 20 帧；原生 CPU 工作线程，schedutil，BPU 996 MHz。沿用原 BIN，未重新执行量化编译。下表其他规格仍为历史实验。[本轮收据](evaluator/evidence/2026-10-30-acceptance.json) · [Workbench 流程](evaluator/README_cn.md)。

| x1.0 variant | Rank-1 | mAP | C++ E2E P50 / P95 |
| --- | --- | --- | --- |
| FP32 reference | 94.2399% | 83.5696% | — |
| spatial-rgb-late8 | 94.0618% | 83.3643% | 8.614 / 9.641 ms |
| spatial-rgb-int16 | 94.2102% | 83.5551% | 9.018 / 10.040 ms |
| spatial-rgb | 93.3195% | 81.7215% | 1.670 / 3.488 ms |

数值来自 2026-10-29 原实验：3368 query + 15913 gallery，L2 归一化平方欧氏距离，无 reranking。C++ 预热 50 次、计时 3000 次，batch=1、单路，BPU 996 MHz、CPU schedutil。E2E 为已解码 64×128 RGB 人体块 → Pillow 等价 NEON 缩放至 128×256 → 同步 Runtime → L2，不含解码、检测和匹配；其他尺寸不能套用这些延迟。

默认版 mAP 损失 0.2053 个百分点，最快 INT8 损失 1.8481 个百分点。[证据与验证范围](evaluator/evidence/README.md)。


### S 系列本地验收（2026-10-30）

| Target | Rank-1 | mAP | Runtime FPS 1 / 2 | C++ E2E ms 1 / 2 | C++ FPS 1 / 2 |
| --- | --- | --- | --- | --- | --- |
| S100 | 94.0914% | 83.5073% | 534.01 / 622.06 | 2.263 / 3.202 | 441.87 / 623.81 |
| S100P | 94.0914% | 83.5073% | 655.52 / 779.28 | 1.832 / 2.560 | 545.71 / 780.36 |
| S600 | 94.0914% | 83.5121% | 816.89 / 1008.27 | 1.515 / 1.973 | 660.05 / 1012.55 |

INT16 激活配置；同一完整 Market1501、同一 FP32 基线和 1 个百分点门槛。每块板单独测量，Runtime 使用 BPU core 0（core_id=1），1/2 表示提交并发数；C++ 使用 1/2 个原生 worker。每种条件 3 轮，Runtime 每轮总计 200 帧，C++ 每流每轮 200 帧，预热 20 帧。未修改时钟策略。已验收模型于 2026-11-02 上传 Archive，仅供研究；默认 INT8 失败实验仍保留在回执中，不作推荐下载。

[完整实测回执](evaluator/evidence/2026-10-30-s-series-acceptance.json) · [S 系列命令和验收流程](evaluator/S_SERIES_cn.md)。

<a id="entry-points"></a>
## 入口

[模型准备](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [C++](runtime/cpp/README_cn.md) · [转换](conversion/README_cn.md) · [评测](evaluator/README_cn.md)。matching.py 实现 OSNet.preprocess → infer → postprocess，由 predict 串联；main.py 直接构造模型。帮助、列表和 dry-run 不加载 SDK。

<a id="license"></a>
## 许可

接入代码遵循仓库 [Apache-2.0](../../../LICENSE)。上游 OSNet 源码为 [MIT](third_party/LICENSE.deep-person-reid)，保留版权及许可声明。当前 Market1501 权重按[原作者说明](https://github.com/KaiyangZhou/deep-person-reid/issues/520)仅供研究；它与四个平台的转换模型不能作为已获商用许可的模型分发。Market1501 原始包禁止分发和商用，因此图片、解码后的 fixture 和原始检索封面不随公开源码分发。新版 Web 封面为 AI 生成的任务示意图。详见[许可核验及来源](third_party/LICENSE_REVIEW.md)。
