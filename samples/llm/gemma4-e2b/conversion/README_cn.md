# 模型转换（PC 端）

**简体中文** | [English](./README.md)

PTQ 量化与 HBM 编译在**开发 PC**上完成，不在板端执行。

## 环境要求

| 项 | 最低 | 推荐 |
| --- | --- | --- |
| 内存 | 64 GB | 128 GB+（Text 编译峰值 ~100 GB） |
| 系统 | Ubuntu 22.04 | 同 |
| SDK | OE-LLM 1.0.0 | 地瓜官方渠道 |
| GPU | 可选 | CUDA 加速 Vision 校准 |

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

## 常用命令

```bash
# Vision 编译（建议 GPU）
bash conversion/scripts/compile/run_vision_compile.sh

# Text 编译（CPU，内存峰值 ~100 GB）
bash conversion/scripts/compile/run_text_compile.sh

# 调整上下文长度
CHUNK_SIZE=512 CACHE_LEN=8192 bash conversion/scripts/compile/run_text_compile.sh
```

编译后需同步修改 `runtime/cpp/gemma4_config.h` 中的 `kChunkSize` / `kCacheLen`，并重新编译板端 runtime。

## 完整教程

含踩坑记录的逐步指南：

- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md)（中文）
- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md)（English）

将 Gemma4 适配代码安装到 OE-LLM 环境：

```bash
bash conversion/leap_llm_gemma4/install.sh
```
