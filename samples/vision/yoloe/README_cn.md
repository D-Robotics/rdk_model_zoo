[English](README.md) | 简体中文

# YOLOE PF 实例分割

<a id="overview"></a>
## 概述

YOLOE 在本 sample 中提供固定 4585 类词表的 Prompt-Free（PF）实例分割。11（E11）使用 DFL16 框回归——64 个框通道在 stride 8/16/32 上按 LTRB 四边各解码为 16-bin 分布——随后做类别内 NMS。26（E26）不使用 DFL（reg_max=1）：框头对每边直接输出一个 LTRB 距离，候选筛选取全局 Top-K 分数，不做 NMS。模型词表固定，不提供任意文字或视觉提示输入。

两版的实例掩码组合方式相同：每个保留候选携带 32 个掩码系数，与模型 32 通道 160×160 原型特征线性组合，再按 sigmoid 中点二值化。

本 sample 位于 `samples/vision/yoloe`。

来源出处：

- 论文：[YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1)；官方仓库：[um-assn/yoloe](https://github.com/um-assn/yoloe)（X5 源）
- 基础检测器谱系：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)（S11 源）

<a id="directory"></a>
## 目录结构

```text
yoloe/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

权重导出、转换准备和评估流程见下方文档。浮点导出检查覆盖图结构；编译与板端运行见[转换指南](conversion/README_cn.md)。

<a id="support-matrix"></a>
## 支持矩阵

| 变体 | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| 11s | 支持 | 仅本地浮点路线* | 不支持 | 不支持 | X5 发布制品；S 本地浮点路线* | 支持 |
| 11m / 11l | 支持 | 不支持 | 不支持 | 不支持 | X5 | 支持（X5 扩展） |
| 26n/s/m/l/x | 不支持 | 仅本地浮点路线* | 仅本地浮点路线* | 不支持 | S 本地浮点路线* | 支持 |

* S 上存在两套输出契约并分开维护：发布的 S11 HBM 为混合输出，S26 公开 HBM 声明量化输出，而本 Python 入口消费浮点输出。要通过本浮点输出口在 S 目标上运行，请按[转换说明](conversion/README_cn.md)导出并编译本地浮点模型，再以其 SHA-256 显式选择；当前没有发布浮点 S HBM，已发布 S 制品为量化输出契约。S600 无源支持，不回退 S100。

权重导出与校准/配置准备工作见[转换说明](conversion/README_cn.md)。下方板测数据使用已发布制品测得；本地浮点路线在自行编译后另行测量。

原生依赖、精确模型选择、构建运行命令及 ROI 结果见 [C++ 流程](runtime/cpp/README_cn.md)；下面的快速启动使用 Python。

<a id="prerequisites"></a>
## 环境与前提

Python 3.10+，NumPy、OpenCV、SciPy、PyYAML；实际推理由匹配板卡系统提供 `hbm_runtime`。主机可运行帮助、列表、dry-run 和合成张量测试。

X5 使用提供 `hbm_runtime` 的系统镜像即可，本 sample 不固定最低镜像版本。S26 源记录使用 RDK OS 4.0.5-Beta、UCP 3.13.6、HBRT 4.7.5、OE 3.7.0。每张图分类头共约 154 MB float32，另有中间张量和掩码；板端内存预算以目标板实际运行为准。

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

dry-run 解析选择并以 JSON 预览显示，不加载模型、不连接板卡。S 目标会提示浮点准备要求：按[转换说明](conversion/README_cn.md)导出并编译本地浮点模型，再以其 SHA-256 显式选择。

<a id="expected-results"></a>
## 预期结果

输出为原图 xyxy float32 框、sigmoid float32 概率和 int64 PF 类别 ID。X5 11 保留 `[N,H,W]` bool 整图掩码；S11/26 为 uint8 0/1 ROI 列表，`mask_layout` 明确区分。空输出框形状为 `[0,4]`，分数和 ID 为 `[0]`。

不承诺测试图固定检测数。下表为源记录的 Runtime 数据（不含前后处理）；Python 端到端性能另含主机侧各阶段开销：

| 模型（源记录） | 目标 | Runtime 延迟 / FPS |
| --- | --- | --- |
| YOLOE-11s PF | X5 | 146.16 ms / 6.84 |
| YOLOE-11m PF | X5 | 177.14 ms / 5.65 |
| YOLOE-11l PF | X5 | 189.97 ms / 5.26 |
| YOLOE-26n PF | S100 | 4.943 ms / 200.74 |
| YOLOE-26s PF | S100 | 9.944 ms / 100.08 |
| YOLOE-26m PF | S100 | 11.765 ms / 84.55 |
| YOLOE-26l PF | S100 | 13.417 ms / 74.18 |
| YOLOE-26x PF | S100 | 22.013 ms / 45.31 |

X5 为源单线程 libdnn Runtime 记录。S100 为源 2026-09-08、200 帧、warmup、thread_num=1/core_id=0 记录，不含前后处理。S100P 无已发布的 Runtime 测量记录。完整条件与精度边界见[评估说明](evaluator/README_cn.md)。

下面两幅插图由 S 源交付发布，均基于随附的同一张 `office_desk.jpg`；此处以独立文件名逐字节保留，与运行时生成的 `test_data/result.jpg` 分开：

![S11 源结果插图](test_data/source_s11_result_figure.jpg)

随 S100 量化 YOLOE-11s PF HBM 发布的结果插图，基于随附 `office_desk.jpg`。该源交付仅支持 S100，其可运行发布制品即该量化 11s PF HBM；源未记录产生此图的具体运行。

![S26 源结果插图](test_data/source_s26_result_figure.jpg)

S26 源交付发布的示例：已发布 S100 量化 YOLOE-26n PF 模型在随附 `office_desk.jpg` 上的实测输出，标签按 PF 检查点类别 ID 顺序导出，与源图注一致。

两图均为源发布的量化 S 结果（基于随附图片）；浮点路线的预期输出来自实际运行本 sample，精度指标来自[评估器](evaluator/README_cn.md)。

<a id="entry-points"></a>
## 入口

- [Model](model/README_cn.md) — 制品身份、下载、哈希。
- [转换](conversion/README_cn.md) — 权重导出、浮点输出检查、各平台校准、编译命令与制品记录；包含原始 X5 E11、S E11、S E26 配方——S 源配方产生量化输出，本准备入口保留浮点输出节点，实际编译精度以编译产物 metadata 为准。
- [评估](evaluator/README_cn.md) — 显式 PF 映射、CPU/板端后端、COCO 指标、来源与历史性能。
- [Python](runtime/python/README_cn.md) — 参数、协议、库接口与排障。
- [C++ 运行时](runtime/cpp/README_cn.md) — 可复用三阶段 C++ 库，包含独立持有的 NV12 输入、浮点绑定和 E11/E26 解码与掩码；SDK 适配器、板卡/模型/词表核验、发布制品选择、CLI 及结果记录一并提供；SDK 构建与板端运行按 C++ 流程指南执行。
- [Test data](test_data/README_cn.md) — 图片、词表与源记录结果插图来源。

<a id="license"></a>
## 许可

样例代码沿用源文件 Apache-2.0 声明和仓库 [LICENSE](../../../LICENSE)。权重与上游 YOLOE/Ultralytics 软件遵循其各自许可，仓库代码许可不自动覆盖模型权重；版本来源见概述中的源说明。
