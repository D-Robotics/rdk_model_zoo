# ASR C++ 音频与契约核心

[English](README.md) | 简体中文

<a id="supported-boards"></a>
## 支持范围
统一原生实现仍在迁移中。S100/S600 有已发布模型，但此目录目前已有可移植解码、音频读取、前处理与主机测试，还没有板端推理可执行程序。X5/S100P 没有 ASR 制品。所有板测均未执行，主机编译不能证明 SDK 兼容。迁移期间保留[源分支 C++ 实现](../../../../../platforms/s/samples/speech/asr/runtime/cpp/)，不会取消原生能力。

<a id="dependencies"></a>
## 依赖
核心仅需 C++17 标准库。测试需 CMake >=3.18；Clang/GCC 默认启用 AddressSanitizer 和 UndefinedBehaviorSanitizer。核心测试不使用 OpenCV、板端 SDK 或音频库；音频测试另需 libsndfile 和 libsamplerate 头文件与库。主机证据使用隔离构建的 libsndfile 1.2.2、libsamplerate 0.2.2，不代表板端版本认证。UCP 适配与预检代码已用主机 API 替身验证；真实 SDK 构建/ABI 和部署 CLI 尚待验证或完成。测试用 libsndfile 关闭外部编解码依赖，已验证 WAV PCM/float；FLAC 等格式是否可用取决于实际构建。

<a id="build"></a>
## 构建
```bash
# cwd: repository root; CMake and a C++17 compiler must be on PATH
cmake -S samples/speech/asr/runtime/cpp/tests -B /tmp/rdk-asr-contract -DASR_SANITIZERS=ON
cmake --build /tmp/rdk-asr-contract
ctest --test-dir /tmp/rdk-asr-contract --output-on-failure
```
产物是 `test_contract`，不是部署用 ASR 程序。不支持 sanitizer 的编译器可以设置 `-DASR_SANITIZERS=OFF`，但该结果不能作为 sanitizer 验证证据。

<a id="run"></a>
## 运行
CTest 运行 `asr_contract`，覆盖 CTC/legacy、负 logits、平局、非有限值、非法词表/ID、音频块几何和归一化，不加载模型或转录音频。已实现的推理入口见 [Python 文档](../python/README_cn.md)；部署 CLI 与真实 SDK 验证待完成。

<a id="parameters"></a>
## 参数
尚无推理 CLI，测试不接收运行参数。`ASR_SANITIZERS` 为 CMake 选项，默认 `ON`。`ASR_AUDIO_TESTS` 默认 `OFF`，开启后构建真实音频库测试。`normalize_probe.cc` 和 `audio_probe.cc` 仅用于对照测试的输出适配，不是部署启动器。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期
`inc/contract.h` 在 `asr` 命名空间提供 `DecodeMode`、`decode_ids`、`decode_logits`、`source_chunk_size`、`PreparedChunk`、`normalize_and_pad`。函数读取调用方输入、返回自有结果，无全局解码状态、SDK 句柄或任务分配。词表必须唯一且首项为 `<pad>`。CTC 先折叠连续 ID 再移除 blank 0；legacy 仅移除 blank。每次调用独立解码，相同 logits 选择最小索引；非有限值和非法形状会报错。

`normalize_and_pad` 只接收已经混为单声道并重采样的有限值。以方差加 1e-5 归一化，再截断/补零到 30000 点，同时返回有效长度。双精度累加与 Python float32 累加可能存在微小差异。该函数不读文件、不重采样，局部输出一致不代表完整原生音频流程等价。

<a id="results-interpretation"></a>
## 结果解释
CTest 退出 0 仅表示主机契约断言通过。解码原样拼接词表字符串，保留 `|` 与非 blank 特殊 token；不做文本清理、置信度、语言模型或跨块拼接。历史原生归一化和解码不同，迁移须遵循这里明确的契约，不能悄悄保留原先 CTC 未折叠的问题。

## 音频读取与前处理

`AudioReader(path)` 独占 libsndfile 句柄，不可复制；打开时验证文件非空及有效元数据，
构造失败或析构均释放句柄。`next(AudioChunk&)` 返回自有交错浮点数据及源采样率、
声道数、帧偏移、块索引，每块最多读取 `ceil(30000 × 源采样率 / 16000)` 帧。
正常 EOF 返回 false 并清空目标；读取错误抛出异常。该类不做数值前处理。

`prepare_chunk(AudioChunk)` 不依赖文件读取。它检查有限交错数据，均值混声，
需要时使用 `SRC_SINC_BEST_QUALITY`，再归一化和补零，输出 30000 个 float
及 `valid_samples`。空输入、错误几何、超长块和不足一个目标点的时长均拒绝。
重采样有效长度读取实际 `output_frames_gen`；混声使用双精度累加以避免多声道浮点溢出。

源实现采用独立窗口和 libsamplerate simple API，本轮保留该窗口契约，不跨块携带
重采样历史或重叠。这不是连续流式重采样：[上游 API 说明](https://libsndfile.github.io/libsamplerate/api_simple.html)
要求连续分块音频使用有状态 API。Python 保留 Fourier 重采样，边界结果可能不同。
两侧均先按方差加 1e-5 归一化再补零，修正了源 C++ 仅按标准差归一化的差异。

先显式安装音频开发库，再从仓库根目录执行。库在自定义前缀时可设置
`ASR_AUDIO_PREFIX`；未设置则使用系统搜索路径。CMake 不自动下载依赖。

```bash
cmake -S samples/speech/asr/runtime/cpp/tests -B /tmp/rdk-asr-audio -DASR_AUDIO_TESTS=ON -DASR_SANITIZERS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}"
cmake --build /tmp/rdk-asr-audio
ctest --test-dir /tmp/rdk-asr-audio --output-on-failure
```

预期 `asr_contract`、`asr_audio`、`asr_task`、`asr_sdk_fixture`、`asr_preflight` 均通过。音频测试创建 8/16/44.1 kHz 临时 WAV，
检查独立窗口几何、末块补零、常量输入、资源归属及错误输入；Release 构建仍保留断言。
[额外七块源实现对照](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-native-audio/)
使用真实音频库、随附录音与生成输入，保留源重采样，只替换旧归一化/补零以比较。
这些结果不验证真实模型或 SDK。

## 三阶段任务接口

`asr.h` 只包含构造/配置与 `pre_process`、`forward`、`post_process`、`predict`。
构造参数为 `Runner`、实测的正数输出步数、有序 3503 项词表及解码模式（默认 CTC）。
调用方须在构造前验证模型身份、张量元数据和词表哈希；任务本身不打开文件。
`SdkRunner::metadata()` 在绑定检查后提供实测步数、步长与分配大小；SDK 调用前使用下述预检工厂。

前处理返回自有定长波形和有效长度。forward 检查输入后只调用传输一次，返回自有
原始 FLOAT32 logits。后处理校验 `[1,steps,3503]`、拒绝非有限值并解码，不做
激活或文件 I/O；predict 组合三阶段。原生沿用源 FLOAT32 输出契约，不能将 Python
支持整数 SCALE 外推为原生也支持。传输依赖的对象需在任务使用期间存活；并发调用
要求传输层线程安全，本接口不保证未来 SDK 实例可并发复用。

以下完整主机示例使用合成音频、词表和传输；`AA` 只是固定替身结果，不是识别文本。
编译时提供 `runtime/cpp/inc` 头文件路径、`runtime/cpp/src/frontend.cc` 与 libsamplerate：

```cpp
#include "asr.h"
#include <iostream>
int main() {
  std::vector<std::string> vocabulary{"<pad>"};
  for (size_t i = 1; i < 3503; ++i)
    vocabulary.push_back("token" + std::to_string(i));
  vocabulary[5] = "A";
  asr::Runner fixture = [](const std::vector<float>&) {
    std::vector<float> logits(4 * 3503, 0.f);
    logits[5] = logits[3503 + 5] = logits[3 * 3503 + 5] = 1.f;
    return logits;
  };
  asr::ASR task(fixture, 4, vocabulary);
  asr::AudioChunk audio{{0.1f, 0.2f, 0.3f}, 16000, 1, 0, 0};
  auto prepared = task.pre_process(audio);
  auto raw = task.forward(prepared);
  std::cout << task.post_process(raw) << '\n';
}
```

## SDK 适配与身份门禁

`SdkRunner(SdkModel{path, target}, preflight)` 只接受 S100/S600，必须提供显式预检，
并在任何 SDK 调用前执行。实际集成使用 `make_preflight(expected_model_sha256,
vocabulary_path)`：它按共享平台注册规则读取本机身份，拒绝目标不符（包括
SoC 为 S100、板型别名为 S100P），核对模型摘要并锁定 3503 项词表摘要。
本地记录的模型哈希只能固定文件字节；发布清单没有摘要时，它不证明发布方来源。

预检后要求一个有名称的模型、一个无量化 FLOAT32 `[1,30000]` 输入，以及一个
无量化 FLOAT32 `[1,T,3503]` 输出，T 必须为正数。分配内存前检查正数对齐容量、
互不重叠且按 float 对齐的字节步长；动态/缺失步长和整数输出均拒绝，不把整数
内存强转为 float。`metadata()` 提供观察到的步数、字节步长、分配容量和模型名，
供调用方写入报告。

`infer(prepared)` 验证有限的 30000 点输入、清零物理补齐区域，按实际步长复制，
清理输入缓存，调用共享同步 UCP 推理，再使输出缓存失效。返回值是按输出步长
拷贝的自有紧凑 logits，后续调用不会覆盖旧输出。调度沿用共享 UCP 默认
`HB_UCP_BPU_CORE_ANY`，此适配器不提供优先级/核心覆盖。传输内部不做激活、
CTC 或音频 I/O。任务 Runner 可捕获存活的 SdkRunner 并委托 `infer`，须保证对象
在全部调用期间存活，不并发复用一个 SDK 实例。

模型和张量所有者在初始化中途失败时清理已取得资源。共用张量所有者现可接管
“返回错误但给出非空地址”的分配，并拒绝“返回成功但地址为空”。任务创建、
提交、等待、释放和缓存操作错误向调用方传播。析构尝试释放但不抛异常；若 SDK
释放本身失败，不能据此认定底层资源已经释放。

## 库构建与验证边界

显式准备音频开发库后，从仓库根目录执行：

```bash
cmake -S samples/speech/asr/runtime/cpp -B /tmp/rdk-asr-library -DASR_BUILD_TESTS=ON -DASR_AUDIO_TESTS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}"
cmake --build /tmp/rdk-asr-library
ctest --test-dir /tmp/rdk-asr-library --output-on-failure
```

默认 `ASR_BUILD_SDK=OFF`，无需厂商头文件即可构建 `asr_frontend`、`asr_preflight`
静态库。`ASR_BUILD_TESTS` 默认 OFF；测试明确使用 API 替身，不认证真实 ABI。
构建 `asr_sdk` 时指定 `-DASR_BUILD_SDK=ON`，环境必须具备 `dnn/hb_dnn.h`、
`hb_ucp.h`、`hb_ucp_sys.h`、`libdnn`、`libhbucp`。CMake 要求全部存在，绝不
静默替换为主机替身。自定义安装可设置 `ASR_DNN_INCLUDE`、`ASR_UCP_INCLUDE`、
`ASR_UCP_SYS_INCLUDE`、`ASR_DNN_LIBRARY`、`ASR_UCP_LIBRARY`；交叉编译需
提供正确目标编译器和 sysroot。X5 与 UCP 头文件同时可见时会拒绝编译。

目前未核定最低 SDK/BSP 版本；本机验证了开启 SDK 构建时因缺少厂商头文件明确失败。
部署 CLI、自动词表 JSON 加载及完整原生转录报告仍未集成，现阶段可使用已实现的
Python 流程。[SDK 证据](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-sdk/)
记录五项主机测试、身份拒绝、带步长张量和部分分配失败复现。真实板端/模型推理未执行。
