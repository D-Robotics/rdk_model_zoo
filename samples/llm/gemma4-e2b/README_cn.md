# Gemma4-E2B VLM

**简体中文** | [English](./README.md)

<p align="center">
  <img src="./docs/image.jpg" alt="Gemma4-E2B on RDK S100P" width="960">
</p>

Google **Gemma4-E2B** 视觉语言模型在 **地瓜 RDK S100P**（`march=nash-m`）上的实时 VLM 推理示例。完全在 BPU 上运行，支持纯文本对话和图文多模态对话。

![纯文本对话演示](./docs/test3.jpg)

*S100P 板端纯文本对话：中文提问，BPU 流式输出（约 6.9 tok/s）。*

![VLM 演示](./docs/test1.jpg)

*VLM 对话：加载图片、中文提问、BPU 流式输出（BPU 利用率 86%）。*

> 支持平台：**RDK S100P**（社区示例；上游项目：[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)）

---

## 算法介绍

Gemma4-E2B 是轻量多模态模型（Vision ViT + 2B Text LLM）：

- **Vision**：16 层 ViT → 每张图 280 个 soft token
- **Text**：35 层 Decoder + PLE + KV cache（4096 上下文）
- **部署**：两个 HBM（Vision + Text）+ 外挂 `tok_embeddings.bin`
- **板端 runtime**：原生 C++（`tokenizers-cpp`），推理时不依赖 Python

---

## 平台兼容性

| 平台 | 支持 | 说明 |
| --- | --- | --- |
| RDK S100P | ✅ | 主要目标平台（`nash-m`，`core_num=1`） |
| RDK S100 / S600 | ❌ | 本示例未验证 |

---

## 目录结构

```bash
samples/llm/gemma4-e2b/
├── README.md / README_cn.md     示例总览（本文件）
├── model/                       预编译 HBM 下载
│   ├── download_model.sh
│   └── README.md
├── conversion/                  PC 端 PTQ 量化编译（128GB RAM + OE-LLM SDK）
│   ├── leap_llm_gemma4/
│   ├── scripts/
│   └── README.md
├── runtime/
│   └── cpp/                     ★ 板端 C++ 推理（gemma4_chat）
│       ├── run.sh
│       └── README.md
├── evaluator/                   精度 / golden 验证
│   └── README.md
├── docs/                        完整量化部署教程
│   ├── QUANTIZATION_TUTORIAL.md
│   └── QUANTIZATION_TUTORIAL_zh.md
├── test_data/                   VLM 测试图片（红熊猫等）
└── third_party/
    └── tokenizers-cpp/          自带的 HF tokenizers C++ 绑定
```

---

## 快速体验

板端一键运行：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh
```

脚本会自动：

1. 安装编译依赖（`cmake`、`g++`、`libopencv-dev`、`cargo`）
2. 若本地无模型，从 HuggingFace 下载预编译 HBM（默认 `~/gemma4_e2b`）
3. 编译 `gemma4_chat`（首次编译 `tokenizers-cpp`，耗时数分钟）
4. 启动交互式 VLM 对话

交互示例：

```
gemma4> /image ../../test_data/image1.jpg
gemma4> 描述这张图片
gemma4> /reset
gemma4> /quit
```

详细步骤见 [runtime/cpp/README_zh.md](./runtime/cpp/README_zh.md)。

---

## 从源码重新量化

需要 **128 GB 内存 PC** + **OE-LLM SDK**。完整教程：

- [QUANTIZATION_TUTORIAL_zh.md](./docs/QUANTIZATION_TUTORIAL_zh.md)（中文）
- [QUANTIZATION_TUTORIAL.md](./docs/QUANTIZATION_TUTORIAL.md)（English）

---

## 许可证说明

本示例中的 C++ runtime 代码为 MIT 许可（见上游 [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)）。预编译模型单独发布在 HuggingFace。
