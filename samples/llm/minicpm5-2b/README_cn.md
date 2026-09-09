[English](README.md) | [简体中文](README_cn.md)

# 在 RDK S100 / S100P / S600 上运行 MiniCPM5-2B

本示例使用 S600 BPU 和 OELLM Runtime 运行 OpenBMB MiniCPM5-2B 文本生成，提供 C++ 命令行程序、带 SHA256 校验的模型下载、转换说明和完整 WikiText2 测评记录。

> S600 当前对应 OELLM 2.0 内测 SDK，尚未公开，正式版计划于 2026 年 10 月中旬发布。公开 S600 1.0.5 与当前 HBM 的直接运行未通过验证，不能替代 2.0；S100/S100P 使用下方独立的公开 1.0.0 SDK 流程。

## S100 / S100P 支持

S100（Nash-e）与 S100P（Nash-m）使用独立的 OELLM 1.0.0 W8 模型包和 [legacy C++ 入口](runtime/legacy/README_cn.md)。两板已通过中英文单轮生成和正常 EOS 验证；未进行全量 PPL。短请求 decode 约 12.1 / 13.0 token/s。内存配置、下载和命令见该入口；转换见 [legacy 转换](conversion/legacy/README_cn.md)。下文原有的 PPL、多轮和稳定性数据仅属于 S600。


## S600 模型与支持范围

MiniCPM5-2B 使用 Llama 架构，包含 42 层、2048 隐藏维度、16 个 query head 和 2 个 KV head。本产物采用 256-token prefill chunk、4096-token KV cache 和四个 Nash-p 核，使用关闭 thinking 的贪心生成，支持中英文文本及同一会话中的后续一轮提问。

本次交付在 **S600、RDK OS V5.1.0** 上验证。模型不能直接用于 S100/S100P。原始模型的更长上下文能力不适用于本次 4096-token 编译配置。图片输入、工具执行和服务端 API 不属于本示例范围。

## 目录结构

```text
conversion/     主机适配代码与量化、编译说明
evaluator/      完整 PPL 评估与测评证据
model/          模型下载及校验
runtime/cpp/    S600 的 CMake 工程和 run.sh
runtime/legacy/ S100/S100P 的 CMake 工程和 run.sh
test_data/      生成提示词与实际参考结果
```

## S600 快速开始

获取并解压 OpenExplorer LLM 2.0.0-beta1，包括其中的 `oellm_runtime` 目录。在 S600 上安装构建依赖：

```bash
sudo apt-get install build-essential cmake libgflags-dev nlohmann-json3-dev curl
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
cd samples/llm/minicpm5-2b/runtime/cpp
bash run.sh
bash run.sh --prompt="What is the capital of France? Answer with the city name only." \
  --follow_up="Translate the previous answer into Chinese. Answer with the city name only."
```

下载和解压需要至少 6 GB 可用空间。SDK 单独获取，模型下载包不包含 SDK 库和头文件。详见[模型下载](model/README_cn.md)与[运行参数](runtime/cpp/README_cn.md)。

## 转换与评估

[转换说明](conversion/README_cn.md)介绍外部 MiniCPM5 适配代码及固定版本的 SDK 环境。不要修改 SDK 的 `deps_version.conf`。[评估说明](evaluator/README_cn.md)记录数据准备、完整 PPL 统计及测量条件。

| 完整 WikiText2 TEST，140 × 2048 token | PPL |
| --- | ---: |
| 浮点参考 | 14.0184 |
| 假量化模型 | 14.2687 |
| 最终 S600 HBM | 14.2428 |

最终 HBM 相对浮点 PPL 上升 1.60%，覆盖全部 286580 个下一 token 预测目标。板端 NumPy 评估与 SDK/PyTorch RPC 路径在 5 个样本上的 PPL 相差 0.0055%，并非逐位一致。

生成验证包括新增 6 条提示词的文本和 token 与浮点参考一致、中英文两轮对话、填充后 2048/3840 token 输入的信息检索，以及 50 次重复请求。该重复请求场景的平均 decode 速度为 53.25 token/s，平均首 token 延迟为 147.86 ms。这些是特定工作负载的数据，不代表并发或长时间稳定性保证；runtime 的 prefill token 数包含 chunk 填充。

原始浮点模型对一条中文 `1+1` 提示词回答错误，并在要求裸 JSON 时添加 Markdown 围栏。量化模型保留了这些表现，本示例不承诺严格 JSON 格式或普遍的事实正确性。

## 许可与来源

原始模型：[OpenBMB/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)，版本 `0e9c66dce9fedde5ba8663bbcdd54b6810bb929a`，采用 Apache-2.0。模型包包含 LICENSE 和量化、元数据修改说明。示例代码遵循仓库许可，OpenExplorer/OELLM 使用其独立 SDK 条款。
