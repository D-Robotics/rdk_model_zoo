[English](README.md) | [简体中文](README_cn.md)

<a id="overview"></a>
# 在 RDK S100 / S100P / S600 上运行 MiniCPM5-2B

本示例使用 S600 BPU 和 OELLM Runtime 运行 OpenBMB MiniCPM5-2B 文本生成，提供 C++ 命令行程序、带 SHA256 校验的模型下载、转换说明和完整 WikiText2 测评记录。

> S600 使用 OELLM 2.0 SDK；S100/S100P 使用下方 OELLM 1.0.0 SDK 流程。请准备与所选模型匹配的 SDK。

<a id="directory"></a>
## 目录结构

```text
minicpm5-2b/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | SDK / 原生入口 | 源精度结论 |
| --- | --- | --- |
| S100 / S100P | 1.0.0 / legacy | PPL +27.83%，未达到 ≤3% |
| S600 | 2.0 beta SDK / cpp | PPL +1.60%，源记录达标 |
| X5 | 无对应实现/资产 | 不适用 |

## S100 / S100P 支持

S100（Nash-e）与 S100P（Nash-m）使用独立的 OELLM 1.0.0 W8 模型包和 [OELLM 1.0.0 C++ 入口](runtime/legacy/README_cn.md)。两板均完成 140 × 2048-token 全量 WikiText2 TEST：PPL 17.91995，相对浮点上升 27.83%，未达到 ≤3% 精度目标。可复现入口见 [全量评估](evaluator/legacy/README_cn.md)。两板均有中英文单轮生成与正常 EOS 的记录。短请求 decode 约 12.1 / 13.0 token/s。内存配置、下载和命令见该入口；转换见 [OELLM 1.0.0 转换](conversion/legacy/README_cn.md)。下文原有的 PPL、多轮和稳定性数据仅属于 S600。

## S600 模型与支持范围

MiniCPM5-2B 使用 Llama 架构，包含 42 层、2048 隐藏维度、16 个 query head 和 2 个 KV head。本产物采用 256-token prefill chunk、4096-token KV cache 和四个 Nash-p 核，使用关闭 thinking 的贪心生成，支持中英文文本及同一会话中的后续一轮提问。

运行平台为 **S600**，记录环境为 RDK OS V5.1.0 与 OELLM 2.0.4 runtime。模型不能直接用于 S100/S100P。原始模型的更长上下文能力不适用于本次 4096-token 编译配置。图片输入、工具执行和服务端 API 不属于本示例范围。

<a id="prerequisites"></a>
## 前置条件

启动器需要 Python 3 标准库，原生推理需要 C++17/CMake、对应 OELLM SDK 与已准备的板型模型。S600 另需 gflags/nlohmann-json；legacy 的内存配置与依赖见其运行指南。模型下载/解压预留约 6 GB。仅运行预编译模型不需要量化工具链。

<a id="quickstart"></a>
## S600 快速开始

获取并解压 OpenExplorer LLM 2.0.0-beta1，包括其中的 `oellm_runtime` 目录。在 S600 上安装构建依赖：

```bash
sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
cd samples/llm/minicpm5-2b/runtime/cpp
BOARD=s600 bash ../../model/download_model.sh
bash run.sh --build
bash run.sh
bash run.sh -- --prompt="What is the capital of France? Answer with the city name only." \
  --follow_up="Translate the previous answer into Chinese. Answer with the city name only."
```

下载和解压需要至少 6 GB 可用空间。SDK 单独获取，模型下载包不包含 SDK 库和头文件。详见[模型下载](model/README_cn.md)与[运行参数](runtime/cpp/README_cn.md)。

<a id="expected-results"></a>
## 转换与评估

[转换说明](conversion/README_cn.md)介绍外部 MiniCPM5 适配代码及固定版本的 SDK 环境。不要修改 SDK 的 `deps_version.conf`。[评估说明](evaluator/README_cn.md)记录数据准备、完整 PPL 统计及测量条件。

| 完整 WikiText2 TEST，140 × 2048 token | PPL |
| --- | ---: |
| 浮点参考 | 14.0184 |
| 假量化模型 | 14.2687 |
| 最终 S600 HBM | 14.2428 |

最终 HBM 相对浮点 PPL 上升 1.60%，覆盖全部 286580 个下一 token 预测目标。板端 NumPy 评估与 SDK/PyTorch RPC 路径在 5 个样本上的 PPL 相差 0.0055%，并非逐位一致。

生成验证包括新增 6 条提示词的文本和 token 与浮点参考一致、中英文两轮对话、填充后 2048/3840 token 输入的信息检索，以及 50 次重复请求。该重复请求场景的平均 decode 速度为 53.25 token/s，平均首 token 延迟为 147.86 ms。这些是特定工作负载下的数据，不含并发或长时间稳定性保证；runtime 的 prefill token 数包含 chunk 填充。

原始浮点模型对一条中文 `1+1` 提示词回答错误，并在要求裸 JSON 时添加 Markdown 围栏；量化模型保留了这些表现。格式敏感的应用请自行校验输出。

<a id="entry-points"></a>
## 入口与下一步

| 任务 | 指南 |
| --- | --- |
| 模型准备与哈希 | [model](model/README_cn.md) |
| 启动器参数与无板预览 | [runtime](runtime/README_cn.md) |
| S600 C++ | [cpp](runtime/cpp/README_cn.md) |
| S100 / S100P C++ | [legacy](runtime/legacy/README_cn.md) |
| 转换配方 | [conversion](conversion/README_cn.md) |
| 评估与参考结果 | [evaluator](evaluator/README_cn.md) |
| 提示词与参考输出 | [test_data](test_data/README_cn.md) |

<a id="license"></a>
## 许可与来源

原始模型：[OpenBMB/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)，版本 `0e9c66dce9fedde5ba8663bbcdd54b6810bb929a`，采用 Apache-2.0。模型包包含 LICENSE 和量化、元数据修改说明。示例代码遵循仓库许可，OpenExplorer/OELLM 使用其独立 SDK 条款。
