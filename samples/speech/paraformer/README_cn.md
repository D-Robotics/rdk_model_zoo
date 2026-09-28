# Paraformer 语音识别

[English](README.md)

<a id="overview"></a>
## 概览

Paraformer 将 16 kHz 语音转换为文本，流程为 FunASR 前端、encoder、predictor、
CPU CIF（连续积分触发）和 decoder。部署使用三份独立发布的 S100 HBM 和包含
8,404 项的有序词表。CIF 明确位于 predictor 与 decoder 之间；文本解码保留重复
字符并移除源特殊 token／BPE 标记。本 Sample 不提供 VAD、流式、标点恢复、时间戳
或自定义热词能力。

上游工具包：[FunASR](https://github.com/modelscope/FunASR)。迁移依据 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`，不直接采用当前上游 main 的其他能力。
本目录位于 `samples/speech/paraformer`。Python CLI 与真实 CPU 前端已提供，
原生 C++ 入口已实现并完成主机流程检查；真实权重三阶段 FP32 导出也已实现。
真实音频校准和显式 OE 编译编排已实现；专用主机评测已实现，实际 OE/HMCT 验证仍待完成，尚未通过完整 Sample 验收。

<a id="support-matrix"></a>
## 支持矩阵

| 发布部署 | x5 | s100 | s100p | s600 | Python | 统一 C++ |
| --- | --- | --- | --- | --- | --- | --- |
| large：encoder 400×560／predictor 400×512／decoder 100×8404 | not-supported | supported-not-run | not-supported | not-supported | 已实现并做主机检查 | 已实现并做主机流程检查；SDK／板测 not-run |

`supported-not-run` 表示声明的 S100 部署与现有 Python 实现，不代表新增板测通过。
本轮未执行真实 SDK／模型元数据核验或板端推理。[原生说明](runtime/cpp/README_cn.md#quickstart)
提供完整准备特征、启动、构建、运行与结果报告流程。主机传输替身测试不能证明
真实双语言 SDK／模型等价。

<a id="prerequisites"></a>
## 前置条件

主机前端已验证 Python 3.12、Torch／torchaudio 2.6.0、FunASR 1.3.14、NumPy 1.26.4、
SoundFile 0.14.0、protobuf 4.23.0。按照[运行说明](runtime/python/README_cn.md#environment)
在独立环境显式安装[前端依赖](runtime/python/requirements-frontend.txt)，以下命令中的
`python` 使用该解释器。帮助／列表／dry-run 仅需基本 Python、NumPy、PyYAML，
不需要 Torch、FunASR 或板端 SDK。

实际推理需要 S100、匹配的板载 `hbm_runtime`、三份 HBM 和固定词表。
归档 Sample 与本轮主机工作未核定准确系统镜像／SDK 版本，不编造最低版本要求。
主机前端环境本身不提供该 SDK。使用已发布制品不需要转换工具链。

磁盘需容纳完整模型包与输出。默认每份特征 `.npy` 为 896,128 字节（400×560 float32
及文件头），另需报告空间。前端会先读取整段音频再截断特征，推理同时加载三模型；
本轮尚未测定最大内存需求或板端容量上限。

<a id="quickstart"></a>
## 快速开始

所有命令从仓库根目录执行。先查看模型包与选择结果，预览不下载、不加载 SDK、
不写文件：

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100 --dry-run
python samples/speech/paraformer/runtime/python/main.py --target s100 --dry-run
```

无板环境可以对两条内置 WAV 执行真实前端。CMVN 和音频已在仓库中，
此模式不需要下载 HBM 或词表：

```bash
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

成功时退出码为 0，`outputs/paraformer-prepared/result.json` 的 status 为 `completed`，
`feats/` 下有两份特征，独立 `prepared-manifest.json` 的 feat_length 为 71、78。
原清单不变。每次使用新输出目录，已有目录会被拒绝。

在 S100 上显式准备六文件模型包，再进行推理。下列命令用于真实板端环境，本轮未做板测：

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
python samples/speech/paraformer/runtime/python/main.py --target s100 --output-dir outputs/paraformer-inference
```

下载器打印路径与实测摘要，不覆盖已有文件。清单没有 HBM 发布方哈希，
本地摘要只能记录字节身份，不能独立认证来源。推理核对本机板型、声明制品和物理
张量契约，不自动下载。可选 `runtime/python/run.sh` 转发相同参数，接受 `PYTHON`
环境变量指定解释器，不安装依赖。

<a id="expected-results"></a>
## 预期结果与限制

内置输入均产生 float32 `[1,400,560]`，分别有 71、78 个有效帧，其余补零。
7 组真实前端对照与固定源逐字节一致，证明的是特征而非新增语音识别精度。
30 秒用例产生 500 个 LFR 帧，只使用前 400 帧并显式记录 `truncated`；
这不是分块转写完整长录音。

推理 `result.json` 包含模型／输入摘要、绑定元数据、逐条文本／token ID、计数和
耗时，参考标注单独保存。未执行的 HBM 推理不编造期望转写、CER 或模型延迟。
CIF 无 token 时返回空文本并标明跳过 decoder。输出目录创建后失败，在可写时留下
包含部分结果的 `failed.json`；缺输入、不兼容文件和板型不符均报错，不跳过。

[主机证据](../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cli-review.md)
覆盖真实预处理 CLI，并明确区分 SDK 替身测试与真实推理。[评测说明](evaluator/README_cn.md) 分开记录历史数据集指标和本次两条语音的 FP32 实测结果。

<a id="directory"></a>
## 目录结构

```text
paraformer/
├── model/           # 显式六文件准备、CMVN/配置与模型说明
├── runtime/python/  # CLI/I/O、真实前端、三个原始 runner、CPU CIF 与文本
├── runtime/cpp/     # 原生应用、SDK 适配与准备后特征输入
├── conversion/      # FP32 导出、真实校准和显式 OE 编排
├── evaluator/       # CPU FP32/HMCT 适配、严格特征校验及 CER 报告
├── test_data/       # 原样保留的源 WAV 与参考清单
├── tests/           # 主机行为和 SDK 边界测试
└── README.md        # 概览、完整操作路径及验证边界
```

<a id="entry-points"></a>
## 各入口

- [模型包](model/README_cn.md)：六文件、身份、下载与重跑行为。
- [Python 运行](runtime/python/README_cn.md#usage)：默认／自定义命令、全部参数、结果、
  阶段接口、完整 CPU 示例和失败处理。
- [测试数据](test_data/README_cn.md)：输入来源与参考文本。
- [C++ 运行](runtime/cpp/README_cn.md)：完整原生应用与启动器已通过明确传输替身做主机检查，真实 SDK／板端未验证。
- [模型转换](conversion/README_cn.md)：严格本地权重加载、真实三阶段 FP32 导出、数值检查、真实音频校准及显式 OE 编排；实际 OE 仍未执行。
- [评测](evaluator/README_cn.md)：特征准备、FP32/HMCT 命令、CER 定义、失败记录和历史指标；两条 smoke 输入不代表数据集精度。

<a id="license"></a>
## 许可证

Sample 代码遵循仓库 [Apache-2.0 许可证](../../../LICENSE)，FunASR 派生导出编排另遵循其
[MIT 许可](conversion/LICENSE-FunASR)。上游模型和数据有各自条款，
代码许可证不能代替权重／数据许可证。活动二进制清单没有逐制品许可字段，
本次迁移不新增权利声明。
