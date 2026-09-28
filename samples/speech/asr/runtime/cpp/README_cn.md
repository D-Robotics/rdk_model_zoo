# ASR C++ contract core

[English](README.md) | 简体中文

<a id="supported-boards"></a>
## 支持范围
统一原生实现仍在迁移中。S100/S600 有已发布模型，但此目录目前仅有可移植解码/归一化核心与主机测试，还没有板端推理可执行程序。X5/S100P 没有 ASR 制品。所有板测均未执行，主机编译不能证明 SDK 兼容。迁移期间保留[源分支 C++ 实现](../../../../../platforms/s/samples/speech/asr/runtime/cpp/)，不会取消原生能力。

<a id="dependencies"></a>
## 依赖
核心仅需 C++17 标准库。测试需 CMake >=3.16；Clang/GCC 默认启用 AddressSanitizer 和 UndefinedBehaviorSanitizer。核心测试不使用 OpenCV、板端 SDK 或音频库。后续原生集成须覆盖源实现的 libsndfile/libsamplerate 音频路径和 UCP SDK 资源，其运行版本尚未认证。

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
CTest 运行 `asr_contract`，覆盖 CTC/legacy、负 logits、平局、非有限值、非法词表/ID、音频块几何和归一化，不加载模型或转录音频。已实现的推理入口见 [Python 文档](../python/README_cn.md)；原生音频/SDK/CLI 集成待完成。

<a id="parameters"></a>
## 参数
尚无推理 CLI，测试不接收运行参数。`ASR_SANITIZERS` 为 CMake 选项，默认 `ON`。`normalize_probe.cc` 仅用于前端比较的二进制文件测试适配，不是公开音频前端或启动器。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期
`inc/contract.h` 在 `asr` 命名空间提供 `DecodeMode`、`decode_ids`、`decode_logits`、`source_chunk_size`、`PreparedChunk`、`normalize_and_pad`。函数读取调用方输入、返回自有结果，无全局解码状态、SDK 句柄或任务分配。词表必须唯一且首项为 `<pad>`。CTC 先折叠连续 ID 再移除 blank 0；legacy 仅移除 blank。每次调用独立解码，相同 logits 选择最小索引；非有限值和非法形状会报错。

`normalize_and_pad` 只接收已经混为单声道并重采样的有限值。以方差加 1e-5 归一化，再截断/补零到 30000 点，同时返回有效长度。双精度累加与 Python float32 累加可能存在微小差异。该函数不读文件、不重采样，局部输出一致不代表完整原生音频流程等价。

<a id="results-interpretation"></a>
## 结果解释
CTest 退出 0 仅表示主机契约断言通过。解码原样拼接词表字符串，保留 `|` 与非 blank 特殊 token；不做文本清理、置信度、语言模型或跨块拼接。历史原生归一化和解码不同，迁移须遵循这里明确的契约，不能悄悄保留原先 CTC 未折叠的问题。
