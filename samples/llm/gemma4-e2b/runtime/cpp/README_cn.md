# C++ Runtime

**中文** | [English](./README.md)

RDK S100P / S600 板端 Gemma4-E2B VLM 推理 C++ runtime，加载与对应 SoC 匹配的预编译 HBM 模型，在 BPU 上运行实时视觉语言推理。

> 属于 [Gemma4-E2B 示例](../../README_cn.md)。完整上游项目：[gemma4-e2b-rdk-s100p](https://github.com/shockley6668/gemma4-e2b-rdk-s100p)。

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/  # inc 相关文件
├── src/  # src 相关文件
├── CMakeLists.txt  # 源码或数据文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── build.sh  # Shell 脚本
├── launcher.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 板卡与运行范围

S100P 使用 `nash-m` HBM，S600 使用 `nash-p` HBM；S100 保留 `nash-e` 运行分支，但需自备匹配 HBM。
`--target` 只选择目标，不会把现有模型转换为该目标。
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
│   ├── gemma4_chat_app.hpp             交互式对话应用会话（应用 facade）
│   ├── gemma4_text_engine.hpp          Text 编排器（prefill + decode + KV 会话）
│   ├── gemma4_text_inputs.hpp          Text 阶段 1：CPU 输入准备（ids/嵌入/位置/mask）
│   ├── gemma4_text_transport.hpp       Text 阶段 2：原始 SDK 写入/推理/KV 收集
│   ├── gemma4_text_session.hpp         Text 会话状态与续写策略
│   ├── gemma4_text_tensor.hpp          固定 Text 导出的描述符契约与带 stride 读写
│   ├── gemma4_vision_engine.hpp        Vision ViT 引擎
│   ├── gemma4_embeddings.hpp           Token embedding 查表 + vision 注入
│   ├── gemma4_kv_cache.hpp             零拷贝 KV cache 管理
│   ├── gemma4_vision_preprocess.hpp    图像缩放 + 分块
│   ├── gemma4_vision_task.hpp          Vision 三阶段及显式 runner 组合
│   ├── gemma4_image_io.hpp             应用层图片读取
│   ├── gemma4_vision_tensor.hpp        SDK 描述符、带 stride 的打包/解包
│   ├── gemma4_vision_debug.hpp         可选诊断日志
│   ├── gemma4_native_tokenizer.hpp     原生 C++ tokenizer（来自 OE-LLM-s600）
│   ├── gemma4_tokenizer.hpp            TokenizerBridge：chat template + 图片展开
│   └── hb_utils.hpp                    Horizon BPU 辅助函数（tensor、flush、infer）
└── src/                                实现 + 入口
    ├── main.cpp                        ★ 薄入口：flags → 路径 → 构造并运行应用
    ├── gemma4_chat_app.cpp             ★ 交互式对话会话（REPL、历史、控制台 IO）
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
export GEMMA4_HOME=~/gemma4_e2b_s600
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
预览以 JSON 输出所选目标、计划执行的命令和设置，`executed=false`。
实际命令保留原生退出码；启动器预检失败返回 2，并提示缺失可执行文件或目标不符。

第三方准备脚本需要 Git/网络及显式安装的稳定版 Rust 1.80+ 工具链；不会安装或升级 Rust。启动器和 CMake 都不会自动调用该脚本。
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
export GEMMA4_HOME=~/gemma4_e2b_s600
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
export GEMMA4_HOME=~/gemma4_e2b_s600

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
- 运行时默认不打印内部诊断。`[VLM-FIX]` 输出跟随 `GEMMA4_DEBUG=1`；Text 引擎诊断使用安装的 `SetDebugSink` 接收器。

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

### 交互式对话入口：应用 facade 与模型类

`main` 在应用边界上拆分。`main.cpp` 是薄入口：解析 gflags、解析 `$GEMMA4_HOME`
默认路径、校验生成参数后构造 `gemma4::chat::InteractiveChatApp`
（`gemma4_chat_app.hpp/.cpp`）并调用 `Run`。应用 facade 持有全部控制台与会话
职责——banner/help、REPL 提示符、UTF-8/GB18030 终端编码归一、对话历史 JSON、
按 4096-token 预算裁剪最旧轮次、前缀失配重置与流式回显。它不承担任何模型
计算：文本生成委托给真正的运行时类 `TextEngine::ContinueGenerateStream`
（图文轮次配合 `BuildPromptHidden`），图像编码委托给 `PredictVision` +
`VisionEngine::Infer`。该 facade 是应用类而非模型
类——不存在执行控制台 IO 的 `predict` 型引擎 API，引擎自身也从不会隐式打印。

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
`VisionEngine::Infer` 接受已准备 patches（图片路径经下面的显式 IO 组合准备）。

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
# 预期：ALL PASSED（5 个输入比较分别满足各自判据）
```

`main`、`demo` 输出生成文本；`server` 返回 JSON 或 SSE，`text_bench` 输出生成/吞吐记录。
数据集级精度请使用评估指南的 PC BC 对比。Golden 校验器比较五个 prefill 输入：整数完全一致，embedding 最大绝对误差 ≤1e-3，两个 mask 误差为 0；cosine 仅打印参考。
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

测试套件覆盖 Vision 三阶段/源图前处理/张量存储、KV 分配/Reset/追加/别名/前缀保留、
Text 所有权（含注入分配失败下的张量采用）、张量契约、生成流程与阶段/会话行为，
外加可运行的 README 示例。Release 构建仍启用断言。测试运行于显式离线 runner；
真实 SDK 描述符与板端数值以本指南的板端命令为准。

交互式对话入口另有专属主机检查：编译生产 `src/gemma4_chat_app.cpp` 与真实
`src/main.cpp`，链接 `tests/native/chat_app_doubles.cpp` 引擎替身、
`tests/native/sdk_fixtures` 的 SDK 头替身，以及 `tests/native/app_stubs/`
中明确标注的 tokenizers-cpp / OpenCV 第三方头编译桩，再通过重定向 stdin
驱动 REPL：

```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_chat_app.py -v
```

该检查覆盖会话逻辑全链路：引擎构造信息、流式回显、`/reset`
`/context` 命令、图文轮次接线（一次 `LoadImage`/`PredictVision`/`Infer`
链路并注入 prompt hidden）、超长 prompt 拒绝、最旧轮次裁剪、每轮重建模式、
跨轮上下文增长、GB18030 终端编码转换，以及薄入口的参数校验。它运行于
显式标注的替身之上；真实 SDK/OpenCV/tokenizers-cpp 栈与生成在板端执行。
`GEMMA_CXX` 选择编译器。nlohmann-json 头文件依次从
`GEMMA_JSON_INCLUDE` 覆盖、`pkg-config nlohmann_json` 或标准系统包含根
（`/usr/include`、`/usr/local/include`、`/opt/homebrew/include`）获取；入口
检查同样链接真实主机 gflags，来源为 `GFLAGS_INCLUDE_DIR` +
`GFLAGS_LIB_DIR`（须成对设置）、`pkg-config gflags` 或标准系统路径。
依赖缺失时相关主机检查以原因显式跳过，无效覆盖则直接失败。iconv 仅在
macOS 链接 `-liconv`；Linux 使用 libc 中的 iconv。

### SDK 失败处理

Vision 构造失败时会释放已经取得的输入/输出 buffer 和 packed model；成功但返回空 handle/buffer 会显式报错。
`MakeTensor` 在分配失败但仍返回内存时也会释放该内存。Vision 要求恰好一个输入和一个输出；具体张量类型、形状和 stride 检查见下节。

全部刷新与 Text/KV 选择性刷新入口共用同一个 task 生命周期：输入刷新 → infer → 按编译核数调度 → submit/wait → 输出刷新与属性更新 → release。
取得 task 之后的失败（包括 infer 返回错误但已给出 task）都会触发释放；正常路径的 release 错误继续上抛，不重复释放同一 handle。
选择性刷新的索引语义、S600 的编译核数选择和可选 V3 入口均按源实现执行。

主机资源测试使用 SDK 接口替身，在 S100/S600 两个编译分支下检查错误处理中的内存/handle 所有权。可从仓库根目录运行：

```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_resources.py -v
```

### Vision 张量传输契约

`gemma4_vision_tensor` 集中负责物理描述符校验、F16 存储打包与带 stride 的输出读取；`VisionEngine` 只组合这些操作和 SDK 调用。
描述符先校验再分配，输出属性在推理后重新校验，不把 SDK 返回的新分配长度当成原 buffer 的真实容量。

| 项 | 接受范围 |
| --- | --- |
| 输入 | `F16`、无量化 metadata、逻辑矩阵 `[2520,768]` |
| 输出 | `F16` 或 `F32`、无量化 metadata、逻辑矩阵 `[280,1536]` |
| 形状 | 可有前导单例轴，如 `[1,2520,768]`；不接受额外 batch、转置或仅元素总数相等的其他形状 |
| stride | 单位为字节，按元素大小对齐、元素与行不重叠，所有访问地址均落在声明分配及原 buffer 容量内 |
| 数据 | 输入 float patches 必须有限且在 `[0,1]`；输出 NaN/Inf 显式拒绝 |

输入保持源实现的 F32→F16 截断方式，写入前清零 padding；输出分别按行和列 stride 提取，自有 float 向量不含 padding。
未知或整数输出会被拒绝，不会强制解释为 float。F16/F32 是运行时存储转换，不涉及修改或重跑量化方案。

主机集成测试实际调用生产 `VisionEngine::Infer`，由 SDK 替身检查输入存储并填充带行/元素间隙的 F16/F32 输出，
同时注入非法类型、量化标记、形状、stride、推理后容量变更和非有限输出。真实发布 HBM 的描述符以板端运行验证。

### KV 缓存状态与所有权

一个 `KvCache` 对应一个串行会话，拥有 15 层 K/V 的 30 个 UCP buffer。每层使用连续 S8 `[4096, head_dim]`
矩阵，`head_dim` 由 `kHeadDims` 固定为 256 或 512；可有矩阵尾部 padding，但这里不支持矩阵内部行 padding。
分配长度不是有效 token 行数，尾部对齐空间不会参与滚动和前缀保留。

| 操作 | 状态与别名 |
| --- | --- |
| `Allocate(k_bytes, v_bytes)` | 各 15 个大小，不能小于对应逻辑矩阵；全部成功后替换旧内存并清空位置；失败保留旧内存和状态 |
| `Reset` | 按 K/V 各自长度清零、清空位置；不释放或改变地址，因此已绑定的输入别名仍有效 |
| `AppendPrefillChunk(...)` | 1–256 个连续位置的 token，从 `OccupiedLen` 追加；先检查所有层指针、stride 和位置，再修改 cache |
| `AppendDecodeStep(...)` | 同一追加逻辑，固定一行 |
| `CompactShift(n_keep, discard)` | 保留当前驻留窗口最前面的 `n_keep` 行、丢弃其余行，重新编号为 0…`n_keep-1`；`discard` 必须等于旧逻辑长度减 `n_keep` |
| `PhysicalIndex(pos)` | 返回右对齐物理行号，未驻留位置返回 -1；`OccupiedLen` 是逻辑结束位置，`CacheStart` 是最旧驻留位置 |

成功重新 Allocate 会使旧别名失效，应只在绑定模型输入之前执行；分配失败不会使旧别名失效。
Append 的源输出内存必须与缓存分离，并至少包含 `(rows-1)*row_stride + head_dim` 个可读字节；
追加会直接拒绝指向任何已驻留 K/V 分配内部的源指针。同一层 K/V 输出行 stride 必须相同，
这一前提由 Text 描述符层在每次追加前强制检查。
共享输入 buffer 仍会由 CPU 搬运数据：追加和前缀保留都执行 CPU 移动/复制，随后推理入口刷新 CPU 改动的 KV 输入。

通过 Reset、Append 和 CompactShift 维护位置与数据一致。
TextEngine 的 `ContextShift` 会调用前缀保留，后续由调用方重新 prefill 需要保留的后缀。

独立主机测试入口：
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_kv.py -v
```

### Text 初始化与张量所有权

`ModelIo` 只能移动，不能复制，负责一个子图的输入/输出内存。子图句柄借用自
packed HBM；KV 输入槽显式借用 `KvCache` 内存，移动时连同借用标记一起转移。
部分构造失败会释放已完成的分配；TextEngine 无论正常销毁还是初始化失败，
都先清理两个子图，再释放 packed model。构造异常传给应用，不返回半初始化对象。
embedding 加载发生在模型获取之前。

固定 Text 导出要求 35 个输入（5 个普通输入、15 对 K/V）和 31 个输出
（logits、15 对 K/V）。空模型句柄、数量不符、缺失或非正的序列维度会在索引使用前拒绝。

### Text 张量传输契约

`gemma4_text_tensor` 集中负责物理描述符校验、带 stride 的输入写入、KV 输出行寻址和贪心 logits argmax；
`TextEngine` 只组合这些操作和 SDK 调用。所有描述符先按固定导出契约校验、再分配；每次推理之后按引擎
最初分配的容量（而非 SDK 刷新后的声明）重新校验输出描述符。不匹配的导出（例如 512 序列长度或
不支持的 dtype）会在构造时、读取任何张量之前拒绝。

| 绑定 | 接受范围 |
| --- | --- |
| `inputs_embeds` | `F32`、无量化 metadata，矩阵 `[chunk,1536]`（prefill）/ `[1,1536]`（decode） |
| `token_ids` / `position_ids` | `S64` / `S32`，一行 `seq` 个元素（源图声明为 `[1,seq]`） |
| `full_mask` / `sliding_mask` | `S16`，矩阵 `[seq,4096]`；允许行 padding，拒绝元素间隙 |
| `logits` | `S16`，矩阵 `[seq,262144]`；行、列 stride 均按描述符读取，其他存储宽度不会重新解释 |
| K/V 输入（5..34） | 每层 `S8` 稠密 `[4096,head_dim]`（忽略单例轴）；拒绝矩阵内部行 padding |
| K/V 输出（1..30） | 每层 `S8` `[seq,head_dim]`；允许行 padding，同层 K/V 行 stride 必须一致 |

单例轴在比较前折叠，因此同一物理布局的 `[4096,1,head_dim]`、`[1,seq]`、`[seq,1536]` 声明都会接受。
byte stride 必须按元素大小对齐、互不重叠，且所有访问地址落在声明分配和原 buffer 容量内。
序列维度固定：prefill 256、decode 1——`kChunkSize`/`kCacheLen`/`kHiddenSize`/`kVocabSize`/`kHeadDims`
这些常量就是导出契约；不同导出的模型需要显式适配层，不得静默进入本引擎。

mask 的 int16 量化、logits 的 `kLogitScale` 反量化与 int8 KV 存储仍保留为 CPU 侧源算法：
携带量化 metadata 的描述符会被拒绝；写入路径先清零 padding，再按描述符 stride 复制。
贪心 argmax 将 int16 存储乘以 `kLogitScale` 并保留首个最大值；平票与全零行的行为与源实现一致。

主机覆盖包括：一项 helper 契约测试（padding、单例轴、dtype/形状/stride/容量/量化拒绝、argmax 行寻址与平票语义），
以及一项生成流程测试——用说这套契约的 SDK 替身驱动生产引擎，校验生成 token、经借用 KV 输入的缓存搬运、
推理失败清理、推理后描述符漂移和构造期拒绝矩阵。

主机测试入口：
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_text_tensors.py -v
```

主机测试仅替换 SDK 调用与 embedding 加载：通过注入失败点、检查 6 类非法描述符及正常销毁，
确认没有残留的张量/模型分配。独立所有权测试覆盖部分构造、移动、重复清理及借用缓存不被释放。
所有权测试不加载权重，也不调用推理。

### Text 流水线阶段

Text 流水线拆分为三个显式阶段加一个会话策略模块；`TextEngine` 只负责编排。行为遵循源实现——
贪心 `kLogitScale` 解码、首个最大值平票、EOS/turn-end 集合、完整返回向量、前缀续写对齐和基准计时范围——
每项职责均有独立可寻的单元：

| 阶段 | 头文件 | 职责 |
| --- | --- | --- |
| 1. 输入准备 | `gemma4_text_inputs.hpp` | `PrepareBatchInputs` / `PrepareDecodeInputs` 生成 `TextBatchInputs`（PLE 替换后的 ids、嵌入行、位置、量化 mask），纯 vector、无 SDK 类型；同时拥有源 mask 构建算法。 |
| 2. SDK 传输 | `gemma4_text_transport.hpp` | `InitTextSubgraph`（描述符契约 + 分配）、`BindKvCache`（零拷贝借用）、`WriteBatchInputs`（带 stride 写入）、`RunSubgraphInference`（刷新 → 推理 → 刷新输出）、`CollectKvOutputs`（重校验后的 KV 行）。不做解码、不做 IO。 |
| 3. 解码 + KV 更新 | `gemma4_text_engine.cpp` | 显式步骤：`ArgmaxTextLogits` 解码、`KvCache::Append*` 更新缓存，随后才推进会话计数。 |
| 会话策略 | `gemma4_text_session.hpp` | 对 `TextSessionState` 的纯决策：上下文平移、自动截断、续写对齐、logits 行选择。 |

容量与窗口契约（在任何分配、查表或写入之前用带溢出保护的符号检查校验；违约抛出
`std::invalid_argument`/`std::runtime_error`，绝不静默钳制注意力）：

- mask 几何（阶段 1 与公开 mask 助手）：`0 <= chunk_valid <= seq_len`、
  `1 <= seq_len <= 4096` 且 `chunk_start + seq_len <= 4096`。超出固定
  4096-token 上下文的 chunk 或解码位置都是错误——`AutoTruncate`/`ContextShift`
  才是留在窗口内的工具。
- prepared token 数：每个 chunk 恰好 `chunk_valid` 个 id。
- prebuilt hidden：两个续写入口与 `GenerateWithPromptEmbeddings` 要求恰好
  `full_ids.size * kHiddenSize` 个 float，从 prompt 起始处索引（不是后缀）；
  尺寸不符在会话状态变更前抛出。`PrepareBatchInputs` 自身也会拒绝小于其所索引
  行数的 hidden。
- 续写入口在对齐平移之前校验 hidden 容量，因此被拒绝的调用之后会话仍可复用。

引擎自身不做任何隐式打印。`SetDebugSink` 为 Text 引擎诊断安装显式接收器；
`[VLM-FIX]` 诊断跟随 `GEMMA4_DEBUG=1`。

下面的示例由主机检查实际编译并运行（`tests/native/readme_text_stages_example.cpp`），
对宿主替身会产生与文档完全一致的输出。板端只需把夹具模型句柄换成对已准备 HBM 的
`hbDNNInitializeFromFiles`，其余代码不变。

```cpp
#include "text_fixture.hpp"   // 离线检查用的 SDK 宿主替身
#include <iostream>
#include <vector>

int main() {
  // 阶段准备：子图 owner 加一个借用的 KV cache。
  hbDNNPackedHandle_t packed = text_fixture::Packed();
  gemma4::TokenEmbeddings embeddings("tok_embeddings.bin");
  gemma4::ModelIo prefill = gemma4::InitTextSubgraph(packed, "prefill", gemma4::kChunkSize);
  gemma4::ModelIo decode  = gemma4::InitTextSubgraph(packed, "decode", 1);
  gemma4::KvCache cache;
  gemma4::BindKvCache(prefill, decode, cache);  // KV 槽位借用 cache

  const std::vector<int64_t> prompt = {11, 22, 33, 44, 55};
  // 阶段 1：按调用准备的上下文。
  const auto batch = gemma4::PrepareBatchInputs(
      embeddings, prompt, 0, static_cast<int>(prompt.size()), nullptr,
      gemma4::kChunkSize);
  // 阶段 2：带 stride 写入 + 一次选择性刷新推理。
  gemma4::WriteBatchInputs(prefill, batch);
  gemma4::RunSubgraphInference(prefill);
  // 阶段 3：追加校验后的 KV 行，再贪心解码。
  const auto rows = gemma4::CollectKvOutputs(prefill, 5);
  //（经 cache.AppendPrefillChunk 追加；完整代码见可运行示例）
  const int64_t first = gemma4::ArgmaxTextLogits(
      prefill.outputs[0], 4, prefill.seq_len, prefill.OutputCapacity(0));
  std::cout << "stage first token: " << first << std::endl;
  prefill.Clear();
  decode.Clear();

  // 高层会话：同一组阶段的多轮编排。
  gemma4::TextEngine engine("text.hbm", "tok_embeddings.bin");
  engine.SetDebugSink([](const std::string &m) { std::cerr << m << "\n"; });
  const auto out = engine.Generate(prompt, 2);
  const auto next = engine.ContinueGenerate(out, 1);
  std::cout << "session processed: " << engine.ProcessedTokens() << std::endl;
  engine.ResetSession();
  return 0;
}
```

主机检查的精确输出（板端 token id 随真实模型不同）：

```
stage first token: 104
session out: 11 22 33 44 55 104 100
session processed: 7
```

示例的生命周期规则：子图句柄借用自 packed model，因此必须在释放 packed model 之前
`Clear` `prefill`/`decode`；KV 输入槽借用 `KvCache` 内存，其有效性持续到 cache 重新分配或销毁；
`CollectKvOutputs` 返回的行借用输出 tensor，仅在下一次推理之前有效。`TextEngine`
（以及对 `ModelIo` 操作的各阶段）都不是线程安全——需串行化访问；阶段函数本身不持有全局状态。
推理失败以异常上抛，task 已释放、所有 buffer 仍被持有，因此会话可以 `ResetSession` 后继续。

主机测试入口：
```bash
python3 -m unittest discover -s samples/llm/gemma4-e2b/tests -p test_cpp_text_stages.py -v
```
