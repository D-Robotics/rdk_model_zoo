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
sudo apt install cmake g++ libopencv-dev cargo
# nlohmann-json 系统自带；若缺失：sudo apt install nlohmann-json3-dev
```

> **无需 Python**。分词走原生 C++ `tokenizers-cpp`（vendored 在 `third_party/`），与 OpenExplorer_LLM-s600 参考实现一致。

## 目录结构

```
runtime/cpp/                      C++ 源码（本目录）
    ├── CMakeLists.txt            构建入口（引入 tokenizers-cpp）
    ├── run.sh                    一键编译 + 交互对话
    ├── gemma4_config.h           模型常量（图片 token ID、维度等）
    ├── gemma4_text_engine.*      Text LLM 引擎（prefill + decode + KV cache）
    ├── gemma4_vision_engine.*    Vision ViT 引擎
    ├── gemma4_embeddings.*       Token embedding 查表 + vision 注入
    ├── gemma4_kv_cache.*         零拷贝 KV cache 管理
    ├── gemma4_vision_preprocess.* 图像缩放 + 分块
    ├── gemma4_native_tokenizer.* 原生 C++ tokenizer（来自 OE-LLM-s600）
    ├── gemma4_tokenizer.*        TokenizerBridge：chat template + 图片展开
    ├── hb_utils.h                Horizon BPU 辅助函数（tensor、flush、infer）
    ├── gemma4_chat.cpp           ★ 交互式 VLM 对话（主入口）
    ├── gemma4_server.cpp         HTTP API 服务
    ├── gemma4_demo.cpp           单次 VLM 演示
    ├── gemma4_text_bench.cpp     纯文本基准测试
    └── gemma4_golden_verify.cpp  Golden mask/KV 对齐校验

../../third_party/
└── tokenizers-cpp/               自带的 HF tokenizers C++ 绑定 + sentencepiece
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

首次编译会构建 vendored 的 `tokenizers-cpp`（HF tokenizers Rust 绑定 + sentencepiece + abseil），耗时数分钟；之后增量编译很快。

产出 5 个可执行文件：

| 可执行文件 | 说明 |
|------------|------|
| `gemma4_chat` | 交互式 VLM 对话，流式输出 |
| `gemma4_server` | HTTP API 服务，供程序化调用 |
| `gemma4_demo` | 单次：图片 + prompt → 文本 |
| `gemma4_text_bench` | 纯文本推理基准 |
| `gemma4_golden_verify` | 校验 prefill 张量与 golden 数据对齐 |

## 下载预编译模型

```bash
pip install huggingface_hub
hf download ShockleyWong/gemma4-e2b-rdk-s100p --local-dir ~/gemma4_e2b
```

下载 3 个模型文件 + tokenizer：

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329 MB  Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json                                   # 32 MB
    ├── tokenizer_config.json
    ├── chat_template.jinja
    └── config.json
```

## 运行

设 `GEMMA4_HOME` 指向模型目录，然后运行：

```bash
export GEMMA4_HOME=~/gemma4_e2b

# 交互式 VLM 对话
./gemma4_chat

# 对话内命令：
#   /image /path/to/photo.jpg   为下一条消息加载图片
#   你看到了什么？              提问
#   /reset                      清空对话
#   /quit                       退出
```

示例输出：

```
gemma4> /image test.jpg
Processing image: test.jpg...
Image loaded (430080 features).
gemma4> 描述这张图片
This is a photograph of a Red Panda resting on a wooden structure...
```

## 核心设计

1. **Vision 原样注入** — ViT 输出 `[280, 1536]` 直接替换 image soft-token 位置（token ID 249560）的 `inputs_embeds`，不做 L2-norm 缩放，不乘 √1536。

2. **PLE 用 pad embedding** — image 位置的 Per-Layer Embedding token-identity 路径用 `pad_token_id=0`（不是 249560），与 HuggingFace `masked_scatter` 行为一致。

3. **Chat template** — C++ 内拼成 Gemma turn 格式（`<bos><|turn>user\n...<turn|>\n<|turn>model\n`），与 `chat_template.jinja` 一致。分词用原生 `tokenizers-cpp`（HF tokenizers），不依赖 python。

4. **零拷贝 KV cache** — KV cache 内存只分配一次，prefill 和 decode 通过指针赋值共享，避免每步 memcpy。

5. **分块 prefill** — 超过 `chunk_size=256` token 的 prompt 自动拆成多个 prefill chunk。

## 验证

验证板端推理与 PC golden 数据是否一致：

```bash
# 将 golden_mask_kv/ 放到 $GEMMA4_HOME/golden_mask_kv/
./gemma4_golden_verify --prompt prompt_0
# 预期：ALL PASSED（全部 5 个张量 cosine=1.0）
```
