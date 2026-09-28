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

## 常用命令

```bash
# 准备固定的 50 张真实 COCO val2017 图像
python3 conversion/scripts/calibration/download_coco_images.py

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

含踩坑记录的逐步指南：

- [QUANTIZATION_TUTORIAL_zh.md](./QUANTIZATION_TUTORIAL_zh.md)（中文）
- [QUANTIZATION_TUTORIAL.md](./QUANTIZATION_TUTORIAL.md)（English）

将 Gemma4 适配代码安装到 OE-LLM 环境：

```bash
bash conversion/leap_llm_gemma4/install.sh
```
