[English](README.md) | 简体中文

# Paraformer 原生推理

[Sample 概览](../../README_cn.md)

当前目录提供 S100 原生可执行入口与启动器、CPU CIF／文本解码、三模型应用编排、
UCP SDK 适配器、生产预检以及准备清单／NPY 读取。按[构建](#build)与
[运行](#run)在 S100 上使用匹配的板端 SDK 构建和执行。
Python 前端 → C++ 推理链路按本目录入口使用；此处不以其他算法近似替换 FunASR。

<a id="overview"></a>
## C++ 推理

通过编码器、预测器与解码器 HBM 在 S100 上转录 16 kHz 音频。CPU CIF 将预测器输出连接到解码器输入，程序写入文本与各阶段耗时。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/  # cif.hpp（CIF 契约）、pipeline.hpp（阶段／预检／解码／Pipeline／SdkRunner）、cli.hpp（参数、词表、清单／NPY 读取、运行工作区）
├── src/  # cif.cpp、pipeline.cpp（预检 + UCP 适配 + 三阶段模型）、cli.cpp（参数／词表／特征／报告）、main.cpp
├── tests/  # 自动化测试
├── CMakeLists.txt  # 构建选项
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── launcher.py  # Python 脚本
├── native_report.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 支持板卡

| 板型 | 状态 | 原因 |
| --- | --- | --- |
| S100 | supported | 已发布三个 HBM；板上构建/运行见[构建](#build)与[运行](#run) |
| X5 / S100P / S600 | not-supported | 无匹配的 Paraformer 发布组合 |

<a id="dependencies"></a>

<a id="environment"></a>
## 环境

数值库需要 CMake 3.18 或以上及 C++17 编译器，不包含厂商 SDK、JSON、NumPy、
Torch 或音频库头文件。主机构建支持 macOS arm64／Apple Clang。真实特征生成
使用单独说明的 [Python 环境](../python/README_cn.md#environment)。

<a id="build"></a>
## 构建并运行主机检查

从仓库根目录执行，使用新的构建目录：

```bash
cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-core -DCMAKE_BUILD_TYPE=Release -DPARAFORMER_BUILD_TESTS=ON -DPARAFORMER_SANITIZERS=ON
cmake --build /tmp/rdk-paraformer-core -j 2
ctest --test-dir /tmp/rdk-paraformer-core --output-on-failure
```

成功标准是四个 CTest 检查通过：数值契约、合成三模型编排、隔离 API 替身下的 SDK
控制流以及整组预检。上述命令在 Clang/GNU
下启用地址／未定义行为检查，Release 测试也保留断言。生产构建产物为静态
`paraformer_pipeline` 模型库，测试可执行文件不是推理 CLI。

<a id="run"></a>

<a id="quickstart"></a>
## 运行原生 Sample

以下以仓库根目录为 cwd。启动器需要 Python、NumPy、PyYAML，不加载 Python 推理 SDK。
实际 C++ 执行需要 S100 和匹配的 UCP SDK；特征生成另需已验证的
[FunASR 环境](../python/README_cn.md#environment)。不会自动下载模型、安装依赖、回退板型
或用测试后端替代真实推理。

任意主机均可预览声明的模型组和原生命令：

```bash
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --list-models
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --target s100 --dry-run
```

激活 FunASR 解释器后，准备内置两份 WAV 特征。此 CPU 步骤不需要板卡或模型文件，
输出目录必须是新目录：

```bash
python samples/speech/paraformer/runtime/python/main.py --target s100 --preprocess-only --output-dir outputs/paraformer_features
```

在 S100 上显式准备发布模型包并执行真实原生后端。下列命令需要下载／板端环境，
在具备下载／板端环境后执行：

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
PYTHON=python bash samples/speech/paraformer/runtime/cpp/run.sh --target s100 --build --manifest outputs/paraformer_features/prepared-manifest.json
```

`--build` 以 Release 配置真实 SDK、I/O、CLI，关闭测试，然后构建
`runtime/cpp/build/s100/paraformer_demo`。按后文提供 SDK／JSON 依赖，CMake 失败也会留证。
后续可省略 `--build` 使用该二进制，或通过 `--binary /absolute/path/to/paraformer_demo`
指定已有程序。每轮使用新的输出目录；`--max-utts 1` 仅处理第一条，0 表示全部。

<a id="parameters"></a>
### 启动器参数

| 参数 | 默认值／行为 |
| --- | --- |
| `--target` | `auto` 读取真实本机身份；可选 auto/x5/s100/s100p/s600，但仅 S100 有发布资产 |
| `--list-models` | 主机列表；auto 列出声明的 S100 模型组，不做板卡能力探测 |
| `--dry-run` | 仅预览，须显式 target；与 list 互斥 |
| `--manifest` | `outputs/paraformer_features/prepared-manifest.json` |
| `--vocab-file` | `samples/speech/paraformer/model/s100/tokens.json` |
| `--max-utts` | 0 全部；正数取前 N 条，负数拒绝 |
| `--output-dir` | `outputs/paraformer_cpp`；必须不存在 |
| `--build` | 显式配置／构建，与 binary 覆盖互斥 |
| `--binary` | 未指定时使用 Sample 下的 `build/s100/paraformer_demo` |
| `--encoder-model-path`、`--predictor-model-path`、`--decoder-model-path` | 默认发布模型路径；覆盖时须提供全部三路径和全部三个匹配制品 ID |
| `--encoder-asset-id`、`--predictor-asset-id`、`--decoder-asset-id` | 预检表中的准确 ID，全部提供或全部省略 |
| `--help` | 显示参数帮助，不加载 SDK／模型 |

启动器相对路径按调用时 cwd 解析；子命令在仓库根目录执行，并接收绝对路径。
`PYTHON` 选择启动器解释器。预处理是显式步骤，此入口只消费已准备特征，不会暗中
重算特征或改写清单。

直接可执行程序接受 `--help`；否则必需 `--target s100`、`--manifest`、`--vocab-file`、
`--output-dir`，以及每个阶段的 `--<stage>-model-path`、`--<stage>-asset-id`、
`--<stage>-sha256`。仅 `--max-utts` 默认 0。启动器根据实际所选文件计算摘要并转发
完整参数；手动调用二进制时必须全部提供。重复／未知参数或缺值均 rc=2。
直接二进制不提供 auto／list／dry-run。

### 结果、日志与失败

启动器成功时 rc=0，并生成：

- `launch-report.json`：发布 ID、发布摘要与实际摘要、程序摘要、准确子命令 argv／cwd／
  UTC 时间、状态和结果文件摘要。
- 构建时的 `configure.*.log`、`build.*.log`，以及 `binary-help.*.log`、`native.*.log`：
  每个已启动进程的完整 stdout／stderr。
- `result/result.json`：目标／后端、三模型摘要及完整物理张量元数据、清单／词表身份
  和逐条语音结果。

逐条记录保留输入注释／参考文本、特征摘要、有效／原始帧数、截断状态、识别文本、
ID／token 数、decoder 执行状态和阶段耗时。
阶段计时不含前端、文件 I/O 或 runner／CIF 调用之外的工作。CIF 精确为空时跳过
 decoder，文本和 ID 为空，decoder 耗时为 null。

`main.cpp` 用已验证模型组和门可见地构造命名 `paraformer::Pipeline` 模型，再对每条
语音调用一次 `pipeline.predict(features, item.valid_frames)`；调用方不出现任何 SDK
runner 构造或阶段接线。运行报告生命周期由 `RunWorkspace`（`inc/cli.hpp`）持有：
预留新输出目录、记录实测阶段元数据与每条预测、完成时重校验清单／特征摘要，
并原子写入 `result.json`／`failed.json`。

两层均拒绝已有输出目录。创建目录前的预检／解析失败为 rc=2、stderr 报错，不创建
结果目录。之后的原生失败写 `result/failed.json`，含已完成的部分记录、当前语音 ID
和错误，不写成功结果。启动器记录失败并保留日志，包括构建失败；若报告写入本身
失败，会在 stderr 说明。`inference_attempted` 在调用编排前变为 true；
`inference_executed` 在调用前为 false，首条调用失败时为 null（可能已发生部分模型
执行），任一语音完整处理后为 true。部分输出不能当作整轮成功。

接受成功前，启动器拒绝 `host-fixture`，检查退出码和结果文件，依据 Python 绑定核验
模型身份、物理形状／类型／角色／字节步长，核对每条所选输入和结果、文本／token／
耗时一致性，并重新计算输入／模型／词表摘要。这些一致性检查用于将报告绑定到本次执行。
主机 CLI 替身不会作为公开二进制安装，也不会被接受为原生成功。

## 构建选项与当前入口

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `PARAFORMER_BUILD_TESTS` | `OFF` | 构建并注册四个主机测试 |
| `PARAFORMER_SANITIZERS` | `OFF` | 在 Clang/GNU 下启用 ASan/UBSan 并传播链接选项 |
| `PARAFORMER_BUILD_CLI` | `OFF` | 构建 `paraformer_demo`；须同时开启 SDK 和 I/O 选项 |
| `PARAFORMER_BUILD_IO` | `OFF` | 启用清单／NPY 读取与 CLI（nlohmann JSON）；开启测试时增加对应主机检查 |
| `PARAFORMER_BUILD_SDK` | `OFF` | 使用真实厂商头文件／库把 UCP 适配编入 `paraformer_pipeline` |
| `CMAKE_BUILD_TYPE` | CMake 默认 | 文档检查使用 `Release` |

完整 CLI 与启动器参数见下文。嵌入其他 CMake 项目时，通过 `add_subdirectory` 加入目录并
链接 `paraformer_pipeline`，公共头文件为 `pipeline.hpp` 和 `cif.hpp`。
浮点操作顺序属于源对齐契约，不应对本数值库开启 fast-math；Clang/GNU 构建显式
关闭浮点收缩。

<a id="interface-lifecycle"></a>

<a id="stage-io"></a>
## 数值与阶段契约

向量表示展平的连续 batch-one 张量，浮点数组由返回值独立持有，不修改输入。
使用前检查尺寸和有限性。

| 接口 | 输入 | 返回 |
| --- | --- | --- |
| `cif` | 401 权重、401×512 hidden、显式有效帧 0–400 | 100×512 acoustic、最多 100 的 int32 计数 |
| `decode` | 100×8404 logits、计数 0–100、8,404 项唯一非空有序词表 | 文本与选中 ID |
| encoder 回调 | 400×560 特征 | 400×512 context |
| predictor 回调 | 400×512 context | 401 权重与 401×512 hidden |
| decoder 回调 | context、acoustic、int32 计数、512 个全零 bias | 100×8404 logits |

CIF 在累计前屏蔽有效帧及之后的权重。无触发返回零数组与零计数；仅保留前 100 次
输出。计算保留源 float64 累加后转 float32，以及每时间步最多触发一次的规则，
不是处理权重大于 1 的通用多次触发积分器。原生 CIF 不隐式提供无屏蔽校准模式。

文本解码在有效 token 前缀取 argmax，分数相同时取首个 ID，过滤 `<...>` 包围的
特殊 token，移除所有 `@@` 后无分隔拼接。重复 token 保留，不是 CTC。
返回 ID 仍包含文本渲染过滤掉的特殊 token；即使只使用有效前缀，也要求所有 logits 有限。

`Pipeline(encoder, predictor, decoder, vocabulary).predict(features, valid_frames)`
是应用编排，不是将 CPU CIF 混入多个 SDK 调用之间的单模型 forward。
有效帧须为 1–400，按 encoder → predictor → CIF → decoder 执行，计数为零时跳过
最后一段。回调必须同步返回独立持有的原始数组；DecoderInput 引用仅在回调期间有效，
不能保存供异步使用。调用方须保证捕获的 SDK 资源存活，非线程安全资源需协调访问。
数值库不加载模型、不选板型、不读写文件、不设置调度，也不编译假 SDK 回退。

第二种原生构造 `Pipeline(models, preflight, vocabulary)` 自持有全部三个阶段的
`SdkRunner`：从已验证的 `ModelGroup` 中为每个阶段选定制品，把阶段回调接到这些
runner 上，并通过 `metadata(stage)` 暴露实测张量元数据。`main.cpp` 使用的正是
这种构造：应用代码传入已验证模型组和 `make_preflight` 门，再对每条语音调用
`predict`；runner 构造、板型选择和 SDK 设置都在模型内部，不在调用方。无 SDK 的
库构建中原生构造在运行期以明确的传输错误拒绝。

<a id="results-interpretation"></a>

<a id="results"></a>
## 返回值与计时

`Prediction` 包含文本、ID／计数、`decoder_executed` 与 `Timings`。
Encoder／predictor／CIF 毫秒耗时为数值，decoder 为 `std::optional<double>`，
跳过时为空。阶段失败抛出异常，不返回成功结果。前端尺寸或帧数错误先于模型回调
拒绝，encoder 输出错误先于 predictor 拒绝。

计时覆盖 runner 调用与 CPU CIF，不含前端、文件 I/O、调用之外的校验及文本渲染，
不是端到端延迟。主机回调不能测量 BPU 性能，输出也不含 CER 或精度估计。
原生入口已接入身份校验及成功／失败记录，具体语义见下文。

<a id="integration-example"></a>
## 可执行数值 API 示例

完成上述构建后，仍从仓库根目录执行：

```bash
cat > /tmp/rdk-paraformer-core/example.cc <<'CPP'
#include "pipeline.hpp"
#include <iostream>
int main() {
    std::vector<float> weights(401, 0.f), hidden(401 * 512, 0.f);
    weights[0] = .75f; weights[1] = .75f; weights[2] = .5f;
    for (int h = 0; h < 512; ++h) {
        hidden[h] = 2.f; hidden[512+h] = 6.f; hidden[1024+h] = 10.f;
    }
    const auto result = paraformer::cif(weights, hidden, 3);
    std::cout << result.token_count << " " << result.acoustic[0] << " "
              << result.acoustic[512] << "\n";
}
CPP
c++ -std=c++17 -fsanitize=address,undefined -Isamples/speech/paraformer/runtime/cpp/inc /tmp/rdk-paraformer-core/example.cc /tmp/rdk-paraformer-core/libparaformer_pipeline.a -o /tmp/rdk-paraformer-core/example
/tmp/rdk-paraformer-core/example
```

预期输出为 `2 3 8`。
[test_pipeline.cc](tests/test_pipeline.cc) 另提供明确使用合成模型输出的可执行编排示例。
Python `--preprocess-only` 已能生成内置音频特征，下文可选 I/O 库可直接读取准备清单。

<a id="troubleshooting"></a>
## 验证与限制

CTest 套件覆盖分数积分、padding、空输出、100 次触发截断、非法契约、
重复／特殊／BPE token、平局 ID、回调顺序、零 token 跳过与错误中间张量。

Sanitizer 运行库构建失败时应检查编译／链接器支持；`PARAFORMER_SANITIZERS=OFF`
可关闭插桩，但不能据此宣称完成 sanitizer 检查。SDK 适配器在加载时核验
形状／类型／名称／身份。SDK 构建、板端推理、OE 与 CER 按各自指南执行；
身份／制品预检工厂见[预检](#preflight)章节。

<a id="sdk-adapter"></a>
## S100 SDK 适配器

可选 UCP 适配器由 `PARAFORMER_BUILD_SDK=ON` 编入 `paraformer_pipeline`，该选项设置
显式编译定义 `PARAFORMER_ENABLE_UCP=1`。仅头文件可见不会启用 SDK 绑定；无 SDK 的
库构建改以链接明确抛错的拒绝传输占位。适配器需要真实 S 系列 UCP 头文件
`dnn/hb_dnn.h`、`hb_ucp.h`、
`hb_ucp_sys.h` 及 `dnn`／`hbucp` 库。在匹配的 SDK 开发环境中，使用独立目录配置：
`cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-sdk -DPARAFORMER_BUILD_SDK=ON`，
然后执行 `cmake --build /tmp/rdk-paraformer-sdk -j 2`。非标准 SDK 路径可通过
`CMAKE_PREFIX_PATH` 或 CMake 缓存变量 `PARAFORMER_DNN_INCLUDE`、
`PARAFORMER_UCP_INCLUDE`、`PARAFORMER_UCP_SYS_INCLUDE`、`PARAFORMER_DNN_LIBRARY`、
`PARAFORMER_UCP_LIBRARY` 指定。不会自动安装 SDK。缺真实依赖时配置失败；主机 API
替身仅用于 `test_sdk`，不会作为此库的回退。

每个阶段单独构造 `SdkRunner(SdkModel{path, "s100", Stage::Encoder}, preflight)`。
回调必填，先于所有 SDK 调用执行，必须拒绝本机身份以及阶段／发布制品／模型摘要
不匹配。空操作回调仅适合隔离的主机测试。请使用下文的具体 `make_preflight` 工厂；
下文启动器将此 API 接入完整 CLI。每份制品必须只包含一个有名称的模型；绑定按名称进行，
不依赖张量顺序：

| 阶段 | 输入角色 → 物理名称 | 输出角色 → 物理名称 |
| --- | --- | --- |
| encoder | features → `speech` | context → `/encoder/after_norm/Add_1_output_0` |
| predictor | context → `/encoder/after_norm/Add_1_output_0` | alphas → `/predictor/Add_output_0`；hidden → `/predictor/Concat_5_output_0` |
| decoder | context → encoder context 名称；count → `token_num`；bias → `bias_embed`；acoustic → `onnx::Shape_8609` 或 `shape_8609` | logits → `logits`；可选 count → `token_num` |

形状见上文阶段表。count 使用 int32，其余使用 float32。物理量化、未知／重复角色、
两个 acoustic 别名同时出现、多余张量、错误维度、重叠／非对齐字节步长或分配容量不足
均拒绝。decoder 的可选 count 输出存在时核验并返回；不引入手动反量化。

`RawTensors` 将语义角色映射到 `variant<vector<float>, vector<int32_t>>`。
`infer` 在接触 SDK 缓冲区前验证完整输入，清空 padding，按照每轴真实字节步长拷贝，
清理缓存，只执行一次同步模型调用，再失效输出缓存并返回自有紧凑数组。后续调用不会
覆盖先前结果。浮点输入必须有限，count 范围为 0–100。返回值为原始数组，后续编排／
解码负责在数值处理前校验输出有限性。单个 runner 应串行使用，或由调用方外部加锁。
调度采用同步调用的默认优先级和任意 BPU 核心。

以下完整嵌入函数已针对公开头文件编译检查。它需要调用方传入模型组、词表及已核验特征，
内部创建真实预检回调；不是合成推理结果，也不是独立板端应用：

```cpp
#include "pipeline.hpp"
#include <algorithm>
#include <utility>
std::vector<float> encode_features(const paraformer::ModelGroup &models,
                                  const std::string &vocabulary,
                                  const std::vector<float> &features) {
    auto verify = paraformer::make_preflight(models, vocabulary);
    const auto encoder_model = std::find_if(models.begin(), models.end(),
        [](const auto &a) { return a.model.stage == paraformer::Stage::Encoder; });
    paraformer::SdkRunner encoder(encoder_model->model, std::move(verify));
    auto outputs = encoder.infer({{"features", features}});
    return std::move(std::get<std::vector<float>>(outputs.at("context")));
}
```

模型、张量分配和推理任务复用共享所有者／调用实现。多输入调用支持 decoder 四输入，
图像调用仍保留一／两输入约束。

<a id="preflight"></a>
## 三模型整体预检

预检检查属于 `paraformer_pipeline`（`src/pipeline.cpp`），链接模型库即获得。
`ModelGroup` 是含三条 `ModelArtifact` 的数组；每条包含
`SdkModel{path, "s100", stage}`、`asset_id` 和 64 位预期 SHA-256。顺序任意，
但 encoder、predictor、decoder 必须各一个。`expected_asset_id(stage)` 返回固定发布 ID：

| 阶段 | Asset ID |
| --- | --- |
| Encoder | `s:paraformer:s100/paraformer_large_encoder_400x560_s100.hbm` |
| Predictor | `s:paraformer:s100/paraformer_large_predictor_400x512_s100.hbm` |
| Decoder | `s:paraformer:s100/paraformer_large_decoder_400x512_s100.hbm` |

词表路径单独传入，固定 SHA-256 为
`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`。
模型预期摘要采用准备模型包时记录的实际文件摘要，用于可追溯文件身份。发布方没有
记录模型摘要，本地预期摘要绑定本地字节；加载时仍会执行形状／类型／名称校验。
仅在重新准备模型包后更新预期摘要，不要只为消除 mismatch 就重新计算并覆盖。

`make_preflight(group, vocabulary)` 通过共享平台读取器读取真实本机身份，立即验证
整组模型，再允许创建 runner。S100P 板型别名优先于通用 S100 SoC 身份。未知身份、
其他目标、阶段／制品 ID 错配、重复阶段、缺失／空／非普通文件、阶段间符号链接／
硬链接别名、模型内容变化及不同词表均拒绝；十六进制模型摘要大小写均接受。

返回回调会核验 runner 的阶段／目标／路径，并在每个模型加载前重新检查本机身份、
三份模型摘要和词表，因此 decoder 文件有问题时不会先加载 encoder。整个工厂创建时
以及各 runner 构造时都会计算三文件摘要，每次推理不重算。加载与推理期间须保持制品
不变；预检不会锁定文件以阻止并发替换。

`verify_group(group, vocabulary, actual)` 是主机测试使用的显式身份底层检查器；
客户部署应使用 `make_preflight` 读取实际身份，不应填写虚构身份。没有绕过开关或
隐式 S100P 回退。

<a id="prepared-features"></a>
## 读取 Python 准备的特征

准备清单／NPY 读取位于 `src/cli.cpp`（声明于 `cli.hpp`），直接读取
[Python `--preprocess-only` 流程](../python/README_cn.md)
生成的 `prepared-manifest.json` 和 `.npy`。它不重新计算或近似替代 FunASR、不重采样音频、
不加载模型，也不改写原始清单。先安装或提供 nlohmann JSON 头文件库（已验证 3.11.3）；
NumPy 是生成特征的依赖，不是此 C++ 读取库的依赖。CMake 不自动安装依赖。
非标准路径使用 `CMAKE_PREFIX_PATH` 或 `-DPARAFORMER_JSON_INCLUDE=/path/to/include`。

在仓库根目录使用独立构建目录：

```bash
cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-io -DCMAKE_BUILD_TYPE=Release -DPARAFORMER_BUILD_IO=ON -DPARAFORMER_BUILD_TESTS=ON -DPARAFORMER_SANITIZERS=ON
cmake --build /tmp/rdk-paraformer-io -j 2
ctest --test-dir /tmp/rdk-paraformer-io --output-on-failure
```

应有六项测试通过，包括读取两份已持久化的真实前端特征。`feature_probe` 是主机验证
工具，不是推理可执行程序。以下是完整的特征读取嵌入函数：

```cpp
#include "cli.hpp"
std::vector<float> first_features(const std::string &manifest) {
    const auto items = paraformer::load_prepared_manifest(manifest, 1);
    return paraformer::load_features(items.front());
}
```

CMake 应用将 `src/cli.cpp` 与 `paraformer_pipeline` 模型库及 nlohmann JSON 头文件一起
编译（`feature_probe` 测试目标展示了准确接法）。实际推理应将所选记录的 `valid_frames`
与读取值一起交给 `Pipeline::predict`，不能把短语音的有效帧数替换为 400。
此示例只返回数组以演示读取，不执行模型。

`load_prepared_manifest(path, max_utts=0)` 返回 `FeatureItem`；0 表示全部，正数取前 N 条。
先对整个清单做结构校验，再选择前缀；仅所选特征在加载时要求文件存在。
`feature_file` 相对清单所在目录解析，也接受绝对路径，之后修改 cwd 不影响已解析路径。
各字段要求：

| 字段 | 契约 |
| --- | --- |
| `utt_id` | 唯一非空文件名主体；不能含斜杠、反斜杠、NUL、点／双点或首尾 ASCII 空白 |
| `feat_length` | 1–400 的整数；不对缺失长度做默认回退 |
| `original_frames` | 正整数，不超过 C++ 有符号 int 上限 |
| `truncated` | 布尔值，等于 `original_frames > 400` |
| `feature_file` | 指向普通 NPY 文件的非空路径 |
| `feature_sha256` | 64 位十六进制摘要，大小写均接受，必须匹配实际读取字节 |
| `text` | 可选参考文本字符串，不是识别输出 |

`feat_length` 必须等于 `min(original_frames, 400)`。未知注释字段通过
`original_record_json` 保留语义 JSON 序列化，不保留原空白排版；`reference_text`
为可选字符串。FeatureItem 另提供解析后的路径、规范化摘要和帧数／截断字段。
用户提供的文件摘要标识本地字节；比较不同运行时连同前端版本一起记录。

`load_features(item)` 对同一份自有字节计算摘要并解析，返回包含 224,000 元素的紧凑
浮点数组。支持 NPY 1.0／2.0／3.0、C 顺序、`[1,400,560]` 形状和显式小／大端 float32
（`<f4`／`>f4`），转换字节序时保留数值。包括 padding 在内的所有值必须有限。
头部仅按数据解析，键顺序任意、单双引号均支持，不执行 Python 表达式；头部长度上限
64 KiB。Fortran 顺序、其他类型／形状／版本、重复／未知头部键、尾随语法或数据、
不完整数据体、非法元数据和内容变化都会抛异常；失败文件不会返回部分成功结果。
