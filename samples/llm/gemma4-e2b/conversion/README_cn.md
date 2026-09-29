# 模型转换（PC 端）

**简体中文** | [English](./README.md)

PTQ 量化与 HBM 编译在**开发 PC**上完成，不在板端执行。

<a id="source-model"></a>
## 源模型与配方

本目录保留固定 S 源提交 `380e1a2` 的 Gemma4-E2B 转换方案：原始权重、Gemma4 的 leap_llm 适配、
Vision/Text 校准、PTQ 编译和验证工具。模型来源为 `google/gemma-4-e2b`；获取权重与访问条件按完整教程 §3.4 执行。
本文命令从 `samples/llm/gemma4-e2b`（sample 根目录）运行，除非代码块显式切换目录。
本次迁移仅优化教程组织，不重新执行量化方案。

<a id="toolchain-targets"></a>
## 环境要求

| 项 | 最低 | 推荐 |
| --- | --- | --- |
| 内存 | 64 GB | 128 GB+（Text 编译峰值 ~100 GB） |
| 系统 | Ubuntu 22.04 | 同 |
| SDK | OE-LLM 1.0.0 | 地瓜官方渠道 |
| GPU | 可选 | CUDA 加速 Vision 校准 |

## 目标 SoC

同一套转换入口支持三种 RDK S 平台。为了兼容原 S100P 样例，`TARGET_SOC`
默认值为 `s100p`。

| `TARGET_SOC` | HBDK march | Vision 核数 | Text prefill / decode 核数 |
| --- | --- | ---: | ---: |
| `s100` | `nash-e` | 1 | 1 / 1 |
| `s100p` | `nash-m` | 1 | 1 / 1 |
| `s600` | `nash-p` | 4 | 2 / 2 |

动态量化、`opt=1`、HPC 和 decode no-padding 只在 `nash-p` 启用，S100/S100P
继续保持原来的单核行为。实测 S600 板端可用内存为 23 GiB（标称 24 GB），
不是 64 GB。

## 目录内容

```bash
conversion/
├── leap_llm_gemma4/          leap_llm 用的 Gemma4 模型定义
│   ├── models/gemma4/
│   └── apis/model/
└── scripts/
    ├── calibration/          COCO 图像 + 文本校准数据准备
    ├── compile/              Vision/Text HBM 编译脚本
    └── verify/               BC/HBM 精度验证
```

<a id="export"></a>
## 模型适配与导出入口

在已准备的 OE-LLM 环境中安装本目录的 Gemma4 适配：

```bash
bash conversion/leap_llm_gemma4/install.sh
```

此流程由 OE-LLM 的模型适配与编译入口承接图导出，不提供独立 ONNX 导出步骤。
完整架构和 Vision/Text 接口分别见完整教程 §2、§5、§6；不要将其他 Sample 的 ONNX 命令套入本流程。

<a id="calibration"></a>
## 校准数据

```bash
# 准备固定的 50 张真实 COCO val2017 图像
python3 conversion/scripts/calibration/download_coco_images.py
```

<a id="compile"></a>
## 编译

```bash
# 选择目标平台；S100/S100P 分别改为 s100/s100p
TARGET_SOC=s600 bash conversion/scripts/compile/run_vision_compile.sh
TARGET_SOC=s600 bash conversion/scripts/compile/run_text_compile.sh
```

Vision 脚本会校验 `images_coco_manifest.json`，拒绝合成图或未登记图片；Text
编译沿用已有文本校准语料，不会自行生成替代 prompt。

当前发布的 Text HBM 使用 `CHUNK_SIZE=256`、`CACHE_LEN=4096` 编译，本次
交付不包含 8K/16K HBM。交互入口 `main` 的 `--max_tokens=0`（默认值）会
自动使用 prompt 之后剩余的全部 KV 容量，因此无需重新编译 HBM，即可用满
现有 4096-token 预算。

后续若重新编译不同规格的 HBM，必须同步修改
`runtime/cpp/inc/gemma4_config.hpp` 中的 `kChunkSize` / `kCacheLen`，再
重新编译板端 runtime。

## 完整教程

完整教程保留源流程；其中板端自动构建/启动描述以当前 [C++ README](../runtime/cpp/README_cn.md#build) 的显式准备、构建、运行步骤为准。量化配方本身不变。

含踩坑记录的逐步指南：

- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md)（中文）
- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md)（English）

<a id="validation"></a>
## 验证流程导航

[评测说明](../evaluator/README_cn.md) 保留 PC 的 BC/浮点对照和板端 golden 输入对齐命令。
完整教程说明精度观察和排查流程；这些是供使用者执行的方案，不是本次迁移的实测声明。

<a id="artifacts"></a>
## 输出制品

Vision 和 Text 分别输出到 `conversion/output/gemma4_e2b_vision_<target>/` 与
`conversion/output/gemma4_e2b_text_<target>/`。板端使用 `gemma4-e2b_vit_ptq.hbm` 与
`gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm`，另外需要 embedding 和 tokenizer；目录及源校验记录见
[模型说明](../model/README_cn.md)。`.bc` 为开发 PC 验证产物，不能作为板端 HBM 传入。

<a id="known-gaps"></a>
## 使用边界

配方原有前提不变：权重、SDK 与文本校准语料需提前准备；Vision 使用带清单的真实图像。
转换支持某个目标不等于该目标存在公开 HBM（S100 仍需自备）。本次文档重构不以重新量化作为验收条件，
也没有重新发布模型、扩大上下文规格或新增板测结论。源教程、脚本与历史示意图继续保留。
