# ASR 原生运行时

[English](README.md) | 简体中文

<a id="supported-boards"></a>
## 支持范围
原生流程实现 S100/S600 选择、固定词表 ASR 与完整文件分块处理。X5/S100P 无对应发布制品，明确拒绝。SDK 适配已用主机 API 替身验证；真实 SDK 编译/ABI、模型推理及板测仍为 **not-run**，主机通过不代表 BSP 认证。原生源实现 (historical `../../../../../platforms/s/samples/speech/asr/runtime/cpp/` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)保留供对照。

<a id="dependencies"></a>
## 依赖
C++17、CMake >=3.18、libsndfile/libsamplerate 开发头文件与库；CLI 与源 sample 一样使用 `nlohmann/json.hpp`。主机证据版本为 libsndfile 1.2.2、libsamplerate 0.2.2、nlohmann JSON 3.11.3。测试的 libsndfile 关闭外部编解码依赖，已验证 WAV PCM/float，其他格式是否支持取决于安装版本。不需要 OpenCV 或 gflags。

真实推理还需匹配 UCP SDK 的 `dnn/hb_dnn.h`、`hb_ucp.h`、`hb_ucp_sys.h`、`libdnn`、`libhbucp`，目前未核定最低 BSP 版本。启动器需 Python 3.10+、NumPy、PyYAML 以使用共享选择工具，不依赖 Python 音频/SciPy 或 `hbm_runtime`。依赖与模型须显式准备，不自动安装或下载。

<a id="build"></a>
## 构建
在匹配板卡上，`run.sh --build` 先验证身份/模型/输入，再以 Release、`ASR_BUILD_SDK=ON`、`ASR_BUILD_CLI=ON`、关闭测试进行构建并运行 `asr_demo`，保留构建日志。模型加载仍核对实际元数据。已有可执行程序可通过 `--binary` 指定。

库构建默认关闭 SDK 和 CLI，`asr_frontend`、`asr_preflight` 不需厂商头文件；`asr_sdk` 需开启 SDK，`asr_demo` 需同时开启 SDK 和 CLI。只开 CLI 会明确失败，缺厂商依赖不会偷偷换成替身。交叉编译应提供目标编译器/sysroot；自定义安装可设置 `ASR_DNN_INCLUDE`、`ASR_UCP_INCLUDE`、`ASR_UCP_SYS_INCLUDE`、`ASR_DNN_LIBRARY`、`ASR_UCP_LIBRARY`、`ASR_JSON_INCLUDE` 或 CMake 安装前缀。X5/UCP 头文件同时可见会拒绝构建。

准备音频和 JSON 开发依赖后，可在主机验证。自定义位置通过 `ASR_AUDIO_PREFIX`、`ASR_JSON_INCLUDE` 指定：
```bash
# cwd: repository root; audio and JSON development dependencies already installed
cmake -S samples/speech/asr/runtime/cpp -B /tmp/rdk-asr-library -DASR_BUILD_TESTS=ON -DASR_AUDIO_TESTS=ON -DASR_CLI_TESTS=ON -DCMAKE_PREFIX_PATH="${ASR_AUDIO_PREFIX:-}" -DASR_JSON_INCLUDE="${ASR_JSON_INCLUDE:-/usr/include}"
cmake --build /tmp/rdk-asr-library
ctest --test-dir /tmp/rdk-asr-library --output-on-failure
```
预期七项测试通过，使用真实音频库及明确命名的 `asr_cli_fixture`。`ASR_BUILD_TESTS`、`ASR_AUDIO_TESTS`、`ASR_CLI_TESTS` 默认 OFF；测试的 `ASR_SANITIZERS` 默认 ON，Release 仍保留断言。替身输出标记 `execution_backend: host-fixture`，公开启动器不会将其接受为 SDK 成功。主机元数据/传输夹具不认证真实模型属性。

<a id="run"></a>
## 运行
从仓库根目录执行以下检查，不需要板卡或模型：
```bash
python3 samples/speech/asr/runtime/cpp/launcher.py --list-models
python3 samples/speech/asr/runtime/cpp/launcher.py --target s100 --dry-run
python3 samples/speech/asr/runtime/cpp/launcher.py --target s600 --dry-run
```
在 S100 显式安装依赖后执行：
```sh
# cwd: repository root, on the matching S100 board with its SDK installed
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/runtime/cpp/run.sh --target s100 --build
# Later run: reuse the built binary and choose a new output directory
bash samples/speech/asr/runtime/cpp/run.sh --target s100 --decode-mode legacy --output-dir outputs/asr_cpp_legacy
```
S600 应下载 s600 制品并替换 target，不复用 S100 文件。默认 auto 读取本机身份，每次启动使用新输出目录。`run.sh` 根据自身位置定位并支持 `PYTHON` 环境变量；默认音频/词表相对 sample 定位，用户的相对路径和输出路径相对当前目录。下表原生二进制默认值以仓库根目录为运行目录。

<a id="parameters"></a>
## 参数
| Launcher option | Default | Meaning / 含义 |
| --- | --- | --- |
| `--target` | `auto` | Local identity; explicit s100/s600 for host dry-run / 本机识别，主机预检须指定 |
| `--asset-id` | `None` | Exact publication identity / 精确发布身份 |
| `--model-path` | `None` | External path requires exact asset-id / 外部路径须同时指定身份 |
| `--audio-file` | `samples/speech/asr/test_data/chi_sound.wav` | Sample-relative default / 默认相对 sample 定位 |
| `--vocab-file` | `samples/speech/asr/test_data/vocab.json` | Fixed source vocabulary / 固定源词表 |
| `--decode-mode` | `ctc` | ctc or legacy / CTC 或旧解码 |
| `--output-dir` | `outputs/asr_cpp` | New launch directory / 新启动记录目录 |
| `--build` | `false` | Explicit native build; conflicts with binary / 显式构建，与 binary 互斥 |
| `--binary` | `None` | Otherwise runtime/cpp/build/TARGET/asr_demo / 默认使用目标构建目录 |
| `--list-models` | `false` | List without SDK / 无 SDK 列表 |
| `--dry-run` | `false` | Prepare without building/inference / 只准备，不构建或推理 |
列表与 dry-run 互斥。不提供优先级/核心、采样率或窗口长度覆盖：UCP 使用 `HB_UCP_BPU_CORE_ANY`，已发布模型固定为 16000 Hz 下 30000 点。外部模型路径必须同时指定精确 asset-id。启动器按清单验证后记录实际摘要；发布方未提供 SHA 时，来源仍未独立认证。

原生 `asr_demo` 是另一层接口，参数如下。Required 表示必填、无默认值，重复/未知/缺值参数均拒绝。正常用户使用启动器，它会传入完整身份和绝对路径。
| Binary option | Default |
| --- | --- |
| `--target` | Required: s100 or s600 |
| `--asset-id` | Required: s:asr:TARGET/asr.hbm |
| `--model-path` | Required |
| `--model-sha256` | Required: 64 hexadecimal digits |
| `--audio-file` | samples/speech/asr/test_data/chi_sound.wav |
| `--vocab-file` | samples/speech/asr/test_data/vocab.json |
| `--output-dir` | outputs/asr_cpp/result |
| `--decode-mode` | ctc |
| `--help` | false |

<a id="interface-lifecycle"></a>
## 接口与资源生命周期
`AudioReader` 独占 libsndfile 句柄；`next(AudioChunk&)` 返回自有交错浮点数据及源采样率/声道、帧偏移/块索引。正常 EOF 清空输出并返回 false，读取失败抛异常。每块读取 `ceil(30000 × 原采样率 / 16000)` 帧，不执行归一化或推理。

`ASR` 只包含构造/配置和四个阶段方法，构造接收 Runner、实际正数输出步数、有序 3503 项词表、解码模式（默认 CTC）。

- `pre_process(AudioChunk)` 验证有限值、均值混声、逐窗口使用 `SRC_SINC_BEST_QUALITY` 重采样，以方差加 1e-5 归一化后补零到 30000，返回自有浮点数据和有效长度。空/错误几何/超长/不足一个目标点的输入拒绝。
- `forward(PreparedChunk)` 验证定长有限输入，仅调用 runner 一次，返回自有原始 logits。
- `post_process(raw)` 核对 `[1,T,3503]`、有限值并解码。CTC 先折叠连续 ID 再去 blank 0；legacy 只去 blank。包括 `|` 在内的非 blank 文本原样保留。
- `predict(AudioChunk)` 组合三阶段。读音频、词表解析和报告保存留在任务类外部。

`SdkRunner` 必须在 SDK 调用前执行 `make_preflight(model_digest, vocabulary_path)`：核对本机精确目标（包含 S100P 别名）、模型字节及固定词表 SHA。`load_vocabulary` 将同一份哈希校验后的字节解析为 3503 个有序 token。适配器要求一个有名称模型、无量化 FLOAT32 `[1,30000]` 输入和 `[1,T,3503]` 输出，分配前验证正容量、无重叠的 float 对齐字节步长及 T。原生拒绝整数/动态描述符，不能将 Python SCALE 支持外推为原生支持。

输入补齐区域清零，浮点值按实际步长复制；检查输入缓存清理、同步 UCP 推理与输出缓存失效，并返回拥有独立内存的紧凑 logits。模型/张量所有者清理部分初始化，包括返回错误但取得非空地址的分配；返回成功但地址为空则拒绝。清理不抛异常，真实 SDK 释放失败时不能保证底层已经回收。不要并发复用 SDK 实例，捕获它的 Runner 不能比实例存活更久。

以下完整主机示例使用合成词表和传输，`AA` 只是替身结果，不是识别文本。编译需 `inc`、`src/frontend.cc` 与 libsamplerate；证据执行了这份中英相同的示例。
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

<a id="results-interpretation"></a>
## 结果解释
启动器退出 0 要求子进程退出 0，且完整 `native-sdk` 报告的身份、张量元数据、分块几何和拼接文本全部校验通过。新目录包括 `launch-report.json`、实际执行过的 configure/build/native 各自原始 stdout/stderr 日志，以及 `result/` 原生结果。启动记录包含精确 argv/cwd、UTC 起止时间/返回码、程序/模型/音频/词表/报告摘要与发布方校验状态。仅准备时明确记录未执行推理。

`result/result.json` 使用 `rdk-model-zoo/asr-native-run/v1`，包含后端、目标/制品、摘要、解码/前端/配置、实测模型/张量元数据及 `chunks`。每块记录索引、原始帧偏移/数量/采样率、有效目标点数、文本；`text` 不插入额外分隔符地拼接完整文件。原生输出目录创建后的失败写 `failed.json`，保留已完成块和错误；启动器保留日志并标记失败。报告自身无法写入时，stderr 保留错误。失败结果不算成功转录。结束时再次核对输入哈希，拒绝执行期间变化。不从 logits 虚构置信度或时间戳。

窗口独立，不跨块保留语言模型、重叠或解码/重采样状态。末块补零后仍解码全部帧，因为有效输出长度尚未验证。Python Fourier 与 C++ sinc 前端可能明显不同，尤其是短窗口；这是保留的源行为，不是跨语言逐值等价。[上游 simple API](https://libsndfile.github.io/libsamplerate/api_simple.html)也不是连续流式重采样器。历史性能见[评估文档](../../evaluator/README_cn.md)，不声称本轮模型延迟或 CER。

## 排错
身份不符：使用精确支持板型，不给 S100P 改名冒充。缺模型：按[模型文档](../../model/README_cn.md)显式下载。SDK 配置失败：补齐匹配头文件/库与编译器。词表摘要不符：恢复随附原文件，不能用同长度词表替代。张量契约错误：核对真实制品与 SDK 元数据，不强转整数或虚构步长。目录已存在：改用新目录。子进程成功但报告无效或为替身：保留日志并按失败处理。

[原生 CLI 证据](../../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-native-cli/)区分 API 替身/主机夹具与真实 SDK/板端执行。全分支独立验收仍未关闭。
