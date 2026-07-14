# Gemma4-E2B VLM 模型说明

**简体中文** | [English](./README.md)

<p align="center">
  <img src="./test_data/results/image.jpg" alt="Gemma4-E2B on RDK S100P" width="960">
</p>

Google **Gemma4-E2B** 视觉语言模型在 **地瓜 RDK S100P**（`march=nash-m`）上的实时 VLM 推理示例。完全在 BPU 上运行，支持纯文本对话和图文多模态对话。

![纯文本对话演示](./test_data/results/test3.jpg)

*S100P 板端纯文本对话：中文提问，BPU 流式输出（约 6.9 tok/s）。*

![VLM 演示](./test_data/results/test1.jpg)

*VLM 对话：加载图片、中文提问、BPU 流式输出（BPU 利用率 86%）。*

> 支持平台：**RDK S100P**（社区示例；上游项目：[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)）

---

## 算法介绍

Gemma4-E2B 是 Google 推出的轻量多模态模型，由 Vision ViT 编码器与 2B 参数 Text LLM decoder 组成。官方资料：

- 模型卡：https://huggingface.co/google/gemma-4-e2b
- 上游部署项目：https://github.com/shockley6668/gemma4-e2b-rdk-s100p

### 算法功能

- 多模态理解：图片 + 文本 → 文本
- 多轮文本对话，复用 KV cache
- BPU 上流式 token 输出

### 算法特性

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
├── conversion/                  PC 端 PTQ 量化编译和完整量化教程
│   ├── QUANTIZATION_TUTORIAL.md
│   ├── QUANTIZATION_TUTORIAL_zh.md
│   ├── leap_llm_gemma4/
│   ├── scripts/
│   └── README.md
├── runtime/
│   └── cpp/                     ★ 板端 C++ 推理（main）
│       ├── run.sh
│       └── README.md
├── evaluator/                   精度 / golden 验证
│   └── README.md
├── test_data/                   VLM 测试图片和结果截图
│   └── results/
└── third_party/                 tokenizers-cpp（构建时下载）
    ├── install_tokenizers_cpp.sh
    └── README.md
```

---

## 快速体验（QuickStart）

板端一键运行：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh
```

脚本会自动：

1. 安装编译依赖（`cmake`、`g++`、`libopencv-dev`、`libgflags-dev`、`nlohmann-json3-dev`、`cargo`、`wget`）
2. 若本地无模型，从地瓜机器人模型服务器下载预编译运行模型文件（默认 `~/gemma4_e2b`）
3. 下载 `tokenizers-cpp` 源码（固定 commit）
4. 编译 `main`（首次编译 `tokenizers-cpp`，耗时数分钟）
5. 启动交互式 VLM 对话

交互示例：

```
gemma4> /image ../../test_data/image1.jpg
gemma4> 描述这张图片
gemma4> /reset
gemma4> /quit
```

详细步骤见 [runtime/cpp/README_cn.md](./runtime/cpp/README_cn.md)。

---

## 模型转换（Model Conversion）

ModelZoo 已提供适配完成的 HBM 模型，用户可直接运行 `model/download_model.sh` 下载使用，如不关心模型转换流程，**可跳过本小节**。

如需自定义重新量化（需要 128 GB 内存的 PC + OE-LLM SDK），请参考 [conversion/README.md](./conversion/README.md) 及完整教程：

- [QUANTIZATION_TUTORIAL_zh.md](./conversion/QUANTIZATION_TUTORIAL_zh.md)（中文）
- [QUANTIZATION_TUTORIAL.md](./conversion/QUANTIZATION_TUTORIAL.md)（English）

---

## 模型推理（Runtime）

本示例仅提供 **C++** 板端推理（LLM 推理为 C++ 原生，不提供 Python 路径）。编译、参数及详细用法请参考 [runtime/cpp/README_cn.md](./runtime/cpp/README_cn.md)。

---

## 模型评估（Evaluator）

`evaluator/` 目录记录精度 / golden 张量校验，详见 [evaluator/README.md](./evaluator/README.md)。

---

## 推理结果

![VLM 演示](./test_data/results/test1.jpg)

*S100P 板端 VLM 对话：图片 + 中文提问 → BPU 流式回复。*

---

## License

本示例中的 C++ runtime 代码为 MIT 许可（见上游 [gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)）。预编译模型单独发布在地瓜机器人模型服务器。示例本身遵循 Model Zoo 顶层 License。
