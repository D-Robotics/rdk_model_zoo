# C++ Runtime

**中文** | [English](./README.md)

S100P 板端 Gemma4-E2B VLM 推理 C++ runtime，加载预编译 HBM 模型在 BPU 上跑实时视觉语言推理。

> 属于 [Gemma4-E2B 示例](../../README_cn.md)。完整上游项目：[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)。

## 前置条件

板子需安装 OE-LLM runtime：

```bash
# 检查 Horizon BPU SDK
ls /usr/hobot/lib/libdnn.so    # BPU 推理库
ls /usr/hobot/lib/libhbucp.so  # 内存管理库
ls /usr/include/hobot/dnn/hb_dnn.h
```

系统依赖：

```bash
sudo apt install cmake g++ libopencv-dev libgflags-dev nlohmann-json3-dev cargo wget
```

> **无需 Python**。分词走原生 C++ `tokenizers-cpp`（构建时下载，见 `third_party/README_cn.md`），与 OpenExplorer_LLM-s600 参考实现一致。

## 目录结构

```
runtime/cpp/                            C++ 源码（本目录）
├── CMakeLists.txt                      构建入口（引入 tokenizers-cpp + gflags）
├── run.sh                              一键编译 + 交互对话
├── inc/                                公共头文件
│   ├── gemma4_config.hpp               模型常量（图像 token ID、维度等）
│   ├── gemma4_text_engine.hpp          Text LLM 引擎（prefill + decode + KV cache）
│   ├── gemma4_vision_engine.hpp        Vision ViT 引擎
│   ├── gemma4_embeddings.hpp           Token embedding 查表 + vision 注入
│   ├── gemma4_kv_cache.hpp             零拷贝 KV cache 管理
│   ├── gemma4_vision_preprocess.hpp    图像缩放 + 分块
│   ├── gemma4_native_tokenizer.hpp     原生 C++ tokenizer（来自 OE-LLM-s600）
│   ├── gemma4_tokenizer.hpp            TokenizerBridge：chat template + 图片展开
│   └── hb_utils.hpp                    Horizon BPU 辅助函数（tensor、flush、infer）
└── src/                                实现 + 入口
    ├── main.cpp                        ★ 交互式 VLM 对话（主入口）
    ├── gemma4_server.cpp               HTTP API 服务
    ├── gemma4_demo.cpp                 单次 VLM 演示
    ├── gemma4_text_bench.cpp           纯文本基准测试
    ├── gemma4_golden_verify.cpp        Golden mask/KV 对齐校验
    └── gemma4_*.cpp                    引擎实现

../../third_party/
└── tokenizers-cpp/                     构建时下载（见 third_party/README_cn.md）
```

## 编译

一键体验（推荐）：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh
```

手动编译：

```bash
cd runtime/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

首次编译会下载并构建 `tokenizers-cpp`（HF tokenizers Rust 绑定 + sentencepiece + abseil），耗时数分钟；之后增量编译很快。

产出 5 个可执行文件：

| 可执行文件 | 说明 |
|------------|------|
| `main` | 交互式 VLM 对话，流式输出（主入口） |
| `gemma4_server` | HTTP API 服务，供程序化调用 |
| `gemma4_demo` | 单次：图片 + prompt → 文本 |
| `gemma4_text_bench` | 纯文本推理基准 |
| `gemma4_golden_verify` | 校验 prefill 张量与 golden 数据对齐 |

## 下载预编译模型

```bash
export GEMMA4_HOME=~/gemma4_e2b
bash ../../model/download_model.sh
```

脚本从地瓜机器人模型服务器下载 3 个运行模型文件和 2 个必需的 tokenizer 文件。

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329 MB  Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## 运行

设 `GEMMA4_HOME` 指向模型目录，然后运行：

```bash
export GEMMA4_HOME=~/gemma4_e2b

# 交互式 VLM 对话（零参数即可，默认从 $GEMMA4_HOME 解析路径）
./main

# 对话内命令：
#   /image /path/to/photo.jpg        为下一条消息加载图片
#   你看到了什么？                    提问
#   /reset                            清空对话
#   /quit                             退出
```

示例输出：

```
gemma4> /image test.jpg
Processing image: test.jpg...
Image loaded (430080 features).
gemma4> 描述这张图片
This is a photograph of a Red Panda resting on a wooden structure...
```

## 命令行参数

5 个可执行文件统一使用 [gflags](https://github.com/gflags/gflags) 解析命令行，参数名采用 `snake_case`（与 Model Zoo 规范一致）。每个参数都有合理默认值，导出 `GEMMA4_HOME` 后零参数即可运行。

### `main` — 交互式 VLM 对话

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--text_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | Text LLM HBM 路径 |
| `--vision_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_vit_ptq.hbm` | Vision ViT HBM 路径 |
| `--tok_embeddings` | string | `$GEMMA4_HOME/model/tok_embeddings.bin` | 外挂 token embedding 表 |
| `--tokenizer_path` | string | `$GEMMA4_HOME/tokenizer/tokenizer.json` | HF tokenizer JSON |
| `--max_tokens` | int | `4096`（`kCacheLen`） | 每轮最多生成 token 数 |

### `gemma4_demo` — 单次文本或 VLM 推理

```
./gemma4_demo {text|vlm} --prompt "..." [--image_path PATH] [其他参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--text_hbm` | string | 同 `main` | Text LLM HBM |
| `--vision_hbm` | string | 同 `main` | Vision ViT HBM（vlm 模式必需） |
| `--tok_embeddings` | string | 同 `main` | Token embedding 表 |
| `--prompt` | string | `""`（必填） | 用户提示文本 |
| `--image_path` | string | `""` | 图片路径（vlm 模式必填） |
| `--max_tokens` | int | `32` | 最多生成 token 数 |

### `gemma4_server` — 长驻对话服务

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--text_hbm` | string | 同 `main` | Text LLM HBM |
| `--vision_hbm` | string | 同 `main` | Vision ViT HBM |
| `--tok_embeddings` | string | 同 `main` | Token embedding 表 |
| `--max_tokens` | int | `128` | 每次请求最多生成 token 数 |

### `gemma4_text_bench` — 纯文本吞吐 / 烟雾测试

```
./gemma4_text_bench {bench|generate} [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--text_hbm` | string | 同 `main` | Text LLM HBM |
| `--tok_embeddings` | string | 同 `main` | Token embedding 表 |
| `--token_ids` | string | `9259`（= `Hello`） | prompt token id，逗号分隔 |
| `--max_tokens` | int | `8` | 生成 token 数 |
| `--warmup` | int | `2` | 计时前的 decode 预热步数 |

### `gemma4_golden_verify` — prefill 与 golden 数据对齐校验

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--golden_root` | string | `$GEMMA4_HOME/golden_mask_kv` | golden 张量根目录 |
| `--prompt_id` | string | `prompt_0` | 子目录名 |
| `--text_hbm` | string | 同 `main` | Text LLM HBM |
| `--tok_embeddings` | string | 同 `main` | Token embedding 表 |

任意 binary 加 `--help` 可查看 gflags 自动生成的完整帮助。

## 核心设计

1. **Vision 原样注入** — ViT 输出 `[280, 1536]` 直接替换 image soft-token 位置（token ID 249560）的 `inputs_embeds`，不做 L2-norm 缩放，不乘 √1536。

2. **PLE 用 pad embedding** — image 位置的 Per-Layer Embedding token-identity 路径用 `pad_token_id=0`（不是 249560），与 HuggingFace `masked_scatter` 行为一致。

3. **Chat template** — C++ 内拼成 Gemma turn 格式（`<bos><|turn>user\n...<turn|>\n<|turn>model\n`），与 `chat_template.jinja` 一致。分词用原生 `tokenizers-cpp`（HF tokenizers），不依赖 python。

4. **零拷贝 KV cache** — KV cache 内存只分配一次，prefill 和 decode 通过指针赋值共享，避免每步 memcpy。

5. **分块 prefill** — 超过 `chunk_size=256` token 的 prompt 自动拆成多个 prefill chunk。

## 验证

验证板端推理与 PC golden 数据是否一致：

```bash
# 可选内部校验数据：将 golden_mask_kv/ 放到
# $GEMMA4_HOME/golden_mask_kv/。该数据不包含在公开模型服务器中。
./gemma4_golden_verify --prompt_id prompt_0
# 预期：ALL PASSED（全部 5 个张量 cosine=1.0）
```
