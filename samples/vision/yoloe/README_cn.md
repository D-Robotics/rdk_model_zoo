# YOLOE PF 实例分割

<a id="overview"></a>
## 概述

YOLOE 在本 sample 中提供固定 4585 类词表的 Prompt-Free（PF）实例分割。11 使用 DFL16 框回归和类别内 NMS；26 使用直接 LTRB 与 Top-K，不做 NMS。模型词表固定，不提供任意文字或视觉提示输入。

统一代码位于 `samples/vision/yoloe`。

算法出处和版本背景保留在 [X5 源说明](../../../platforms/x5/samples/vision/yoloe/README_cn.md)及 [S26 源说明](../../../platforms/s/samples/vision/yoloe26_seg/README_cn.md).

<a id="support-matrix"></a>
## 支持矩阵

| Variant | x5 | s100 | s100p | s600 | Canonical Python | Canonical C++ |
| --- | --- | --- | --- | --- | --- | --- |
| 11s | supported-not-run | not-supported* | not-supported | not-supported | X5; local float S route* | no, pending |
| 11m / 11l | supported-not-run | not-supported | not-supported | not-supported | X5 | no source X5 implementation |
| 26n/s/m/l/x | not-supported | not-supported* | not-supported* | not-supported | local float S route* | no, pending |

* 表中 S 的 not-supported 指“当前发布制品不能直接用于本浮点入口”，不是删除该能力。S11 最终 HBM 为混合精度，S26 公开 HBM 声明量化输出。统一代码允许明确指定、哈希固定的本地浮点转换制品，但兼容的 S 浮点 HBM 尚未编译验证，不能标为 supported-verified。S600 无源支持，不回退 S100。

本轮板测全部 `not-run`；统一权重导出和校准/配置准备已提供，真实 OE 验收与 C++ 收编仍 pending。源分支的历史板测不转算为统一代码验证。

<a id="prerequisites"></a>
## 环境与前提

Python 3.10+，NumPy、OpenCV、SciPy、PyYAML；实际推理由匹配板卡系统提供 `hbm_runtime`。主机可运行帮助、列表、dry-run 和合成张量测试。主机测试环境为 Python 3.14.7，不代表板端版本验收。

本轮未核定 X5 最低系统镜像版本。S26 源记录使用 RDK OS 4.0.5-Beta、UCP 3.13.6、HBRT 4.7.5、OE 3.7.0，不能推导为新的兼容性承诺。每张图分类头共约 154 MB float32，另有中间张量和掩码；峰值内存和性能尚未实测。

<a id="quickstart"></a>
## 快速开始

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11s
```

在 X5 上执行。下载命令将原制品保存到 `model/x5/` 并校验发布 SHA-256；推理使用随附 `office_desk.jpg`，成功退出为 0，输出 JSON 计数/分数/类别并写入 `test_data/result.jpg`。默认运行不下载。可用 `runtime/python/run.sh` 转发相同参数。

主机检查：

```bash
# cwd: repository root
python3 samples/vision/yoloe/runtime/python/main.py --list-models
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n --dry-run
```

S dry-run 成功仅表示选择解析成功；输出会明确提示需要独立浮点转换制品。

<a id="expected-results"></a>
## 结果与验证边界

输出为原图 xyxy float32 框、sigmoid float32 概率和 int64 PF 类别 ID。X5 11 保留 `[N,H,W]` bool 整图掩码；S11/26 为 uint8 0/1 ROI 列表，`mask_layout` 明确区分。空输出框形状为 `[0,4]`，分数和 ID 为 `[0]`。

本轮无真实模型结果，不承诺测试图固定检测数。历史 Runtime 数据如下，仅供与源说明对齐，不能当作统一 Python 端到端性能：

| Historical model | Target | Runtime latency / FPS |
| --- | --- | --- |
| YOLOE-11s PF | X5 | 146.16 ms / 6.84 |
| YOLOE-11m PF | X5 | 177.14 ms / 5.65 |
| YOLOE-11l PF | X5 | 189.97 ms / 5.26 |
| YOLOE-26n PF | S100 | 4.943 ms / 200.74 |
| YOLOE-26s PF | S100 | 9.944 ms / 100.08 |
| YOLOE-26m PF | S100 | 11.765 ms / 84.55 |
| YOLOE-26l PF | S100 | 13.417 ms / 74.18 |
| YOLOE-26x PF | S100 | 22.013 ms / 45.31 |

X5 为源单线程 libdnn Runtime 记录。S100 为源 2026-09-08、200 帧、warmup、thread_num=1/core_id=0 记录，不含前后处理。S100P 未测 Runtime 性能。完整条件和精度边界见 [S26 evaluation](../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README_cn.md).

<a id="directory"></a>
## 目录职责

```text
yoloe/
├── model/             # explicit published-artifact preparation
├── conversion/        # ONNX checks, calibration, target YAML and optional compile
├── evaluator/         # explicit category mapping, COCO metrics and prediction export
├── runtime/python/    # CLI, binding, raw runner and three-stage task
├── test_data/         # source image and fixed vocabulary
├── tests/             # host fixtures, source comparisons and README execution
└── README.md
```

统一权重导出、转换准备和评估流程已提供，C++ 尚待收编。浮点导出检查不代表真实编译器或板端验收完成。

<a id="entry-points"></a>
## 入口

- [Model](model/README_cn.md) — 制品身份、下载、哈希。
- [转换](conversion/README_cn.md) — 权重导出、浮点输出检查、各平台校准、编译命令与制品记录。
- [评估](evaluator/README_cn.md) — 显式 PF 映射、CPU/板端后端、COCO 指标、来源与历史性能。
- [Python](runtime/python/README_cn.md) — 参数、协议、库接口与排障。
- [Test data](test_data/README_cn.md) — 图片和词表来源。
- [X5 conversion](../../../platforms/x5/samples/vision/yoloe/conversion/README_cn.md) / [S11 conversion](../../../platforms/s/samples/vision/yoloe11_seg/conversion/README.md) / [S26 conversion](../../../platforms/s/samples/vision/yoloe26_seg/conversion/README_cn.md) — 原配方；S 配方会产生量化输出。统一准备入口保留浮点输出节点，实际编译精度仍待验证。
- [统一 C++ 进度](runtime/cpp/README_cn.md) — 可复用三阶段 C++ 库已提供，包含独立持有的 NV12 输入、浮点绑定和 E11/E26 解码与掩码；SDK 适配器已实现，本机/模型/词表核验已实现，发布制品选择与板端入口仍待集成。
- [S11 C++](../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) / [S26 C++](../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README_cn.md) — 历史实现，统一移植 pending。
- [X5 evaluation](../../../platforms/x5/samples/vision/yoloe/evaluator/README_cn.md) / [S26 evaluation](../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README_cn.md) — 历史记录，不是本轮验收。

<a id="license"></a>
## 许可

样例代码沿用源文件 Apache-2.0 声明和仓库 [LICENSE](../../../LICENSE)。权重与上游 YOLOE/Ultralytics 软件遵循其各自许可，仓库代码许可不自动覆盖模型权重；版本来源见概述中的源说明。
