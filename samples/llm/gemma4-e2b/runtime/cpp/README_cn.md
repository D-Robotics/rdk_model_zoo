# C++ Runtime

**中文** | [English](./README.md)

RDK S100P / S600 板端 Gemma4-E2B VLM 推理 C++ runtime，加载与对应 SoC 匹配的预编译 HBM 模型，在 BPU 上运行实时视觉语言推理。

> 属于 [Gemma4-E2B 示例](../../README_cn.md)。完整上游项目：[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)。

<a id="supported-boards"></a>
## 板卡与运行范围

S100P 使用 `nash-m` HBM，S600 使用 `nash-p` HBM；S100 保留 `nash-e` 运行分支，但需自备匹配 HBM。
本次迁移没有新增板测。`--target` 是目标选择，不会把现有模型转换为该目标。
主机只能执行启动器帮助和预览；原生可执行文件需要板端 SDK。

<a id="dependencies"></a>
## 前置条件

板端需安装 OE-LLM runtime：

```bash
# 检查 Horizon BPU SDK
ls /usr/hobot/lib/libdnn.so    # BPU 推理库
ls /usr/hobot/lib/libhbucp.so  # 内存管理库
ls /usr/include/hobot/dnn/hb_dnn.h
```

系统依赖：

```bash
sudo apt install cmake g++ libopencv-dev libgflags-dev nlohmann-json3-dev cargo wget git curl
```

> 启动器使用 Python 3 标准库；推理与分词仍为原生 C++，第三方 `tokenizers-cpp` 需显式准备。直接执行原生二进制不依赖 Python。

## 目录结构

```
runtime/cpp/                            C++ 源码（本目录）
├── CMakeLists.txt                      构建入口（引入 tokenizers-cpp + gflags）
├── run.sh                              显式编译或启动
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
└── tokenizers-cpp/                     显式准备（见 third_party/README_cn.md）
```

<a id="build"></a>
## 编译

从仓库根目录准备并显式构建（系统包按上节安装）：

```bash
cd samples/llm/gemma4-e2b
export GEMMA4_HOME=~/gemma4_e2b
# S100P: s100p; S600: s600. Keep different targets in separate model directories.
GEMMA4_SOC=s600 bash model/download_model.sh
bash third_party/install_tokenizers_cpp.sh
cd runtime/cpp
./run.sh --target s600 --build
./run.sh --target s600
./run.sh --target s600 server --port=8000
./run.sh --target s600 main --max_tokens=512
```

启动器参数放在入口名之前，原生参数放在入口名之后。例如 `./run.sh --target s600 demo text --prompt "Hello"`。
零参数仍选择 `main`，但要求已构建；`--build` 只构建，不启动推理。
`--home`（默认 `GEMMA4_HOME` 或 `~/gemma4_e2b`）选择数据目录，`--build-dir` 默认本目录 `build/`。
`--target auto` 默认从共享平台注册表识别实际板卡，包含 S100P 别名；显式 target 与板身份不符时拒绝运行。
S100 保留手动 HBM 分支，没有默认公共 HBM。不同板目标使用不同数据目录，避免同名 HBM 混用。

主机离线入口：
```bash
./run.sh --help
./run.sh --target s600 --dry-run demo text --prompt "Hello"
./run.sh --target s100p --build --dry-run
```
预览输出 JSON，`executed=false`；不加载 SDK，不证明模型哈希或板端兼容性。
实际命令保留原生退出码；启动器预检失败返回 2，并提示缺失可执行文件或目标不符。

第三方准备脚本需要网络，可能安装 Rust 1.80+ 的 rustup 工具链。它不再由启动器或 CMake 自动调用。
编译 Rust 依赖仍可能访问包仓库；离线构建需要提前准备全部依赖缓存。
独立 Abseil 可通过 `GEMMA4_ABSL_PREFIX=/opt/abseil ./run.sh --target s600 --build` 指定。

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
GEMMA4_SOC=s600 bash ../../model/download_model.sh
```

S100P 与 S600 会下载各自已验证的公共 HBM，以及共享 embedding 和 tokenizer。S100 需预置匹配的 HBM，或设置 `GEMMA4_MODEL_BASE_URL`；缺失的共享文件仍会自动下载。

```
~/gemma4_e2b/
├── model/
│   ├── gemma4-e2b_vit_ptq.hbm                          # 329-377 MB Vision
│   ├── gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm      # 4.5 GB  Text
│   └── tok_embeddings.bin                               # 1.5 GB  Embedding
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

<a id="run"></a>
## 运行

下面的直接原生命令从 `samples/llm/gemma4-e2b/runtime/cpp/build` 目录执行（先完成构建）。

设 `GEMMA4_HOME` 指向模型目录，然后运行：

```bash
export GEMMA4_HOME=~/gemma4_e2b

# S600 手动启动时使用系统 DNN runtime；run.sh 会自动设置这些环境变量
unset LD_LIBRARY_PATH GEMMA4_USE_DNN_V3
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6

# 交互式 VLM 对话（零参数即可，默认从 $GEMMA4_HOME 解析路径）
./main

# 对话内命令：
#   /image /path/to/photo.jpg        为下一条消息加载图片
#   你看到了什么？                    提问
#   /context                          查看 KV cache 使用量
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

### 4096-token 上下文

- Text HBM 的总容量固定为 4096 tokens，约束是 `prompt_tokens + output_tokens <= 4096`。
- `main` 和 `gemma4_server` 默认都使用 `--max_tokens=0`，表示每轮自动使用当前 prompt 之后的全部剩余容量；短 prompt 最多可获得接近 4096 tokens 的输出预算。
- 新一轮至少为回复保留 `--min_response_tokens`（默认 256）个 token；空间不足时按完整 user/assistant 对裁掉最旧历史并重建 KV cache。
- `/context` 显示当前使用量、剩余容量和轮数。停止 token 不会显示或写入 assistant 正文。
- `main` 在进入交互循环前统一加载 Text 和 Vision，两者在整个会话期间常驻；S100/S100P/S600 共用同一生命周期，`/image` 只执行图片预处理和 Vision 推理，不会重新加载模型。
- 图文追问会保留原始图片轮，并在当前用户问题旁再次显式注入同一组 Vision 特征；prompt 中最多包含两个 280-token 图片块。
- 运行时默认不打印内部诊断；仅当设置 `GEMMA4_DEBUG=1` 时输出 `[DEBUG]` / `[VLM-FIX]` 信息。

<a id="parameters"></a>
## 命令行参数

5 个可执行文件统一使用 [gflags](https://github.com/gflags/gflags) 解析命令行，参数名采用 `snake_case`（与 Model Zoo 规范一致）。每个参数都有合理默认值，导出 `GEMMA4_HOME` 后零参数即可运行。

### `main` — 交互式 VLM 对话

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--text_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm` | Text LLM HBM 路径 |
| `--vision_hbm` | string | `$GEMMA4_HOME/model/gemma4-e2b_vit_ptq.hbm` | Vision ViT HBM 路径 |
| `--tok_embeddings` | string | `$GEMMA4_HOME/model/tok_embeddings.bin` | 外挂 token embedding 表 |
| `--tokenizer_path` | string | `$GEMMA4_HOME/tokenizer/tokenizer.json` | HF tokenizer JSON |
| `--max_tokens` | int | `0` | 每轮最多生成 token 数；`0` 表示使用 prompt 后全部剩余 KV 容量 |
| `--min_response_tokens` | int | `256` | 自动裁剪旧历史时为新回复保留的最小容量 |

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

### `gemma4_server` — OpenAI 兼容文本服务

`gemma4_server` 常驻加载 Text HBM，并提供串行的 OpenAI 兼容 HTTP API。连续请求的 token 前缀一致时会复用 KV cache。该接口仅支持文本；如果请求中包含图片会返回 HTTP 400，图文对话继续使用交互式 `main`。

~~~bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh server --host=0.0.0.0 --port=8000
~~~

| 方法 | 接口 | 说明 |
|---|---|---|
| `GET` | `/health` | 就绪状态、模型名、上下文长度和已缓存 token 数 |
| `GET` | `/v1/models` | OpenAI 兼容模型列表 |
| `POST` | `/v1/chat/completions` | 普通 JSON 或 SSE 流式对话 |

普通请求示例：

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"请详细介绍 RDK S600。"}],"max_tokens":0}'
~~~

`max_tokens: 0` 是本样例扩展，表示使用固定 4096-token KV cache 中 prompt 之后的全部剩余容量。需要 SSE 时加入 `"stream": true`，同时支持 `stream_options.include_usage`。

ChatBox 中选择 OpenAI 兼容接口，Base URL 填 `http://板端IP:8000/v1`，模型名填 `gemma4-e2b`。如果客户端强制要求 API Key，填任意非空占位值即可，服务端不会校验 `Authorization` 请求头。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--host` | string | `0.0.0.0` | HTTP 监听地址 |
| `--port` | int | `8000` | HTTP 监听端口 |
| `--model` | string | `gemma4-e2b` | `/v1/models` 返回的模型名 |
| `--text_hbm` | string | 同 `main` | Text LLM HBM |
| `--tok_embeddings` | string | 同 `main` | Token embedding 表 |
| `--tokenizer_path` | string | 同 `main` | HF tokenizer JSON |
| `--max_tokens` | int | `0` | 默认输出上限；`0` 表示使用 prompt 后全部剩余容量 |
| `--min_response_tokens` | int | `256` | 裁剪旧完整轮次时为回复保留的容量 |
| `--request_limit_mb` | int | `4` | HTTP 请求体大小上限 |

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

<a id="interface-lifecycle"></a>
## 核心设计

1. **Vision 原样注入** — ViT 输出 `[280, 1536]` 直接替换 image soft-token 位置（token ID 249560）的 `inputs_embeds`，不做 L2-norm 缩放，不乘 √1536。

2. **PLE 用 pad embedding** — image 位置的 Per-Layer Embedding token-identity 路径用 `pad_token_id=0`（不是 249560），与 HuggingFace `masked_scatter` 行为一致。

3. **Chat template** — C++ 内拼成 Gemma turn 格式（`<bos><|turn>user\n...<turn|>\n<|turn>model\n`），与 `chat_template.jinja` 一致。分词用原生 `tokenizers-cpp`（HF tokenizers），不依赖 python。

4. **零拷贝 KV cache** — KV cache 内存只分配一次，prefill 和 decode 通过指针赋值共享，避免每步 memcpy。

5. **分块 prefill** — 超过 `chunk_size=256` token 的 prompt 自动拆成多个 prefill chunk。

6. **完整 KV 预算** — 交互入口按当前 prompt 动态计算输出上限，可精确使用到 `4096/4096`，下一轮再按完整对话轮次裁剪旧历史。

7. **统一双模型生命周期** — `main` 启动时统一按 Vision→Text 顺序加载两个模型，并在整个进程内常驻。该顺序避免 S600 跨 core IOVA 映射冲突，同时 S100/S100P/S600 共用完全相同的聊天主流程；板型差异只体现在匹配的 HBM、CMake SoC 宏和 `run.sh` 环境设置。

### Vision 库接口与职责

`gemma4_image_io` 负责读图，`gemma4_vision_preprocess` 只处理内存像素，`gemma4_vision_task`
组合三个阶段；`VisionEngine` 负责 SDK 模型及张量传输。交互与单次 VLM 入口均使用同一组合。

| 接口 | 输入 → 输出 | 契约 |
| --- | --- | --- |
| `LoadImage(path)` | 文件路径 → `cv::Mat` | 应用 IO，OpenCV 解码为 BGR；读取失败抛异常 |
| `PreprocessImage(bgr)` | 非空二维 `CV_8UC3` → float `[2520,768]` | BGR→RGB、bicubic 到 960×672、除 255、16×16 分块；不改输入、不读文件 |
| `ForwardVision(patches, runner)` | 上述 float patches → runner 原始结果 | 固定元素数、有限 `[0,1]`，显式 runner 调用一次 |
| `PostprocessVision(raw)` | float `[280,1536]` → 自有 feature 向量 | 元素数及有限值检查，无缩放、L2 或额外归一化 |
| `PredictVision(bgr, runner)` | BGR 图像与 runner → features | 三阶段组合，不创建 SDK、不读写文件、不打印 |

前处理按 patch 行、patch 列排序，patch 内按像素行、像素列、RGB 通道交错排列。
输出以值持有，后续调用不会覆写先前结果；SDK engine 本身仍需串行访问。
`VisionEngine::Infer` 接受已准备 patches，不再接受图片路径。原路径式调用迁移到下面的显式 IO 组合。

以下完整示例在已有 SDK 工程中链接 `gemma4_runtime` 使用，不是无板主机示例：

```cpp
#include <iostream>
#include <stdexcept>
#include <vector>
#include "gemma4_image_io.hpp"
#include "gemma4_vision_engine.hpp"
#include "gemma4_vision_task.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: vision_example VISION_HBM IMAGE\n";
    return 2;
  }
  try {
    const cv::Mat image = gemma4::LoadImage(argv[2]);
    gemma4::VisionEngine engine(argv[1]);
    const auto features = gemma4::PredictVision(
        image, [&engine](const std::vector<float>& patches) {
          return engine.Infer(patches);
        });
    std::cout << features.size() << " vision feature values\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
```

模型需与板卡及编译目标匹配；成功时输出 `430080 vision feature values`。例子只输出 Vision 特征数量，不生成文本。
在原生 CMake 工程中将例子保存为 `vision_example.cpp`，增加：

```cmake
add_executable(vision_example vision_example.cpp)
target_link_libraries(vision_example PRIVATE gemma4_runtime)
```

<a id="results-interpretation"></a>
## 验证

验证板端推理与 PC golden 数据是否一致：

```bash
# 可选内部校验数据：将 golden_mask_kv/ 放到
# $GEMMA4_HOME/golden_mask_kv/。该数据不包含在公开模型服务器中。
./gemma4_golden_verify --prompt_id prompt_0
# 预期：ALL PASSED（全部 5 个张量 cosine=1.0）
```

`main`、`demo` 输出生成文本；`server` 返回 JSON 或 SSE，`text_bench` 输出生成/吞吐记录。
这些输出不等于数据集精度结果。Golden 校验器比较五个 prefill 输入：整数完全一致，embedding 最大绝对误差 ≤1e-3，两个 mask 误差为 0；cosine 仅打印参考。
`ALL PASSED` 对应退出码 0；不匹配或异常为 1。完整数据前提见 [评测说明](../../evaluator/README_cn.md)。
每个 TextEngine 持有一个会话及其 KV 状态，调用方应串行访问；不要把交互会话当成无状态、可并发共享的推理函数。

### 无 SDK 的主机回归

仅检查算法边界与源前处理一致性，使用已安装的 OpenCV C++ 库；不安装依赖、不准备模型：

```bash
# From repository root; set OpenCV_DIR if OpenCV is not on CMake's search path.
cmake -S samples/llm/gemma4-e2b/tests/native -B /tmp/gemma-vision-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gemma-vision-tests --parallel
ctest --test-dir /tmp/gemma-vision-tests --output-on-failure
```

两项 CTest 分别覆盖三阶段/所有权/异常输入，以及四张源图片前处理逐字节比较。Release 构建仍启用断言。
这些测试使用显式测试 runner，不加载 BPU 或证明真实 SDK 的描述符、资源生命周期及板端数值；相关审查继续进行。
