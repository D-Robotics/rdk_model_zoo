# Paraformer 原生数值库与 SDK 适配器

[English](README.md) · [Sample 概览](../../README_cn.md)

当前目录提供原生 CPU CIF／文本解码、三模型应用编排和单独的 S100 UCP SDK 适配器。
准备清单读取和完整原生可执行入口仍在迁移，具体身份／制品预检工厂已提供。以下测试是主机检查，不是 HBM 推理。
源 Python 前端 → C++ 推理能力仍在迁移范围内，此处不以其他算法近似替换 FunASR。

<a id="environment"></a>
## 环境

数值库需要 CMake 3.18 或以上及 C++17 编译器，不包含厂商 SDK、JSON、NumPy、
Torch 或音频库头文件。主机构建在 macOS arm64／Apple Clang 上检查，准确版本见
[证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-native-core-review.md)。
不声明 SDK ABI 或 S100 推理通过。真实特征生成使用单独说明的
[Python 环境](../python/README_cn.md#environment)。

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
`paraformer_contract` 库，测试可执行文件不是推理 CLI。

<a id="parameters"></a>
## 构建选项与当前入口

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `PARAFORMER_BUILD_TESTS` | `OFF` | 构建并注册四个主机测试 |
| `PARAFORMER_SANITIZERS` | `OFF` | 在 Clang/GNU 下启用 ASan/UBSan 并传播链接选项 |
| `PARAFORMER_BUILD_SDK` | `OFF` | 使用真实厂商头文件／库构建 `paraformer_sdk` |
| `CMAKE_BUILD_TYPE` | CMake 默认 | 文档检查使用 `Release` |

当前没有板端 CLI 参数。嵌入其他 CMake 项目时，通过 `add_subdirectory` 加入目录并
链接 `paraformer_contract`，公共头文件为 `contract.h` 和 `pipeline.h`。
浮点操作顺序属于源对齐契约，不应对本数值库开启 fast-math；Clang/GNU 构建显式
关闭浮点收缩。

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
旧 C++ 已处理无触发输入；旧 Python 的空输出异常在统一 Python 中修复。

文本解码在有效 token 前缀取 argmax，分数相同时取首个 ID，过滤 `<...>` 包围的
特殊 token，移除所有 `@@` 后无分隔拼接。重复 token 保留，不是 CTC。
返回 ID 仍包含文本渲染过滤掉的特殊 token；即使只使用有效前缀，也要求所有 logits 有限。

`Pipeline(encoder, predictor, decoder, vocabulary).predict(features, valid_frames)`
是应用编排，不是将 CPU CIF 混入多个 SDK 调用之间的单模型 forward。
有效帧须为 1–400，按 encoder → predictor → CIF → decoder 执行，计数为零时跳过
最后一段。回调必须同步返回独立持有的原始数组；DecoderInput 引用仅在回调期间有效，
不能保存供异步使用。调用方须保证捕获的 SDK 资源存活，非线程安全资源需协调访问。
数值库不加载模型、不选板型、不读写文件、不设置调度，也不编译假 SDK 回退。

<a id="results"></a>
## 返回值与计时

`Prediction` 包含文本、ID／计数、`decoder_executed` 与 `Timings`。
Encoder／predictor／CIF 毫秒耗时为数值，decoder 为 `std::optional<double>`，
跳过时为空。阶段失败抛出异常，不返回成功结果。前端尺寸或帧数错误先于模型回调
拒绝，encoder 输出错误先于 predictor 拒绝。

计时覆盖 runner 调用与 CPU CIF，不含前端、文件 I/O、调用之外的校验及文本渲染，
不是端到端延迟。主机回调不能测量 BPU 性能，输出也不含 CER 或精度估计。
后续原生入口还必须提供身份校验及成功／失败记录，才能成为完整部署入口。

<a id="integration-example"></a>
## 可执行数值 API 示例

完成上述构建后，仍从仓库根目录执行：

```bash
cat > /tmp/rdk-paraformer-core/example.cc <<'CPP'
#include "contract.h"
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
c++ -std=c++17 -fsanitize=address,undefined -Isamples/speech/paraformer/runtime/cpp/inc /tmp/rdk-paraformer-core/example.cc /tmp/rdk-paraformer-core/libparaformer_contract.a -o /tmp/rdk-paraformer-core/example
/tmp/rdk-paraformer-core/example
```

预期输出为 `2 3 8`，这是完整合成数值示例，不是识别文本。
[test_pipeline.cc](tests/test_pipeline.cc) 另提供明确使用合成模型输出的可执行编排示例。
Python `--preprocess-only` 已能生成内置音频特征，C++ 消费该准备清单的能力尚待接入。

<a id="troubleshooting"></a>
## 验证与限制

两个数值测试覆盖分数积分、padding、空输出、截断、非法契约、重复／特殊／BPE token、
平局 ID、回调顺序、零 token 跳过与错误中间张量。对照驱动在 27 组数据上与提取的
固定源 C++ CIF 及统一 Python 逐字节比较，并做 20 组原生／Python 文本对照。
准确编译器和复现脚本见报告。提取的源函数／驱动只是主机证据，不是另一份维护中的运行实现。

Sanitizer 运行库构建失败时应检查编译／链接器支持；`PARAFORMER_SANITIZERS=OFF`
可关闭插桩，但不能据此宣称完成 sanitizer 检查。真实 HBM 的形状／类型／名称／身份
已在 SDK 适配器中实现，但 API 替身不能证明实际 HBM 兼容；具体身份／制品预检
工厂已单独提供。真实 SDK 构建、板端推理、完整原生
CLI、OE 与 CER 均按实际状态保持未执行或待完成。

<a id="sdk-adapter"></a>
## S100 SDK 适配器

可选 `paraformer_sdk` 库需要真实 S 系列 UCP 头文件 `dnn/hb_dnn.h`、`hb_ucp.h`、
`hb_ucp_sys.h` 及 `dnn`／`hbucp` 库。在匹配的 SDK 开发环境中，使用独立目录配置：
`cmake -S samples/speech/paraformer/runtime/cpp -B /tmp/rdk-paraformer-sdk -DPARAFORMER_BUILD_SDK=ON`，
然后执行 `cmake --build /tmp/rdk-paraformer-sdk -j 2`。非标准 SDK 路径可通过
`CMAKE_PREFIX_PATH` 或 CMake 缓存变量 `PARAFORMER_DNN_INCLUDE`、
`PARAFORMER_UCP_INCLUDE`、`PARAFORMER_UCP_SYS_INCLUDE`、`PARAFORMER_DNN_LIBRARY`、
`PARAFORMER_UCP_LIBRARY` 指定。不会自动安装 SDK。缺真实依赖时配置失败；主机 API
替身仅用于 `test_sdk`，不会作为此库的回退。本机缺厂商 SDK，已验证配置明确失败，
未将其记为真实 SDK 构建通过。

每个阶段单独构造 `SdkRunner(SdkModel{path, "s100", Stage::Encoder}, preflight)`。
回调必填，先于所有 SDK 调用执行，必须拒绝本机身份以及阶段／发布制品／模型摘要
不匹配。空操作回调仅适合隔离的主机测试。请使用下文的具体 `make_preflight` 工厂；
完整清单／CLI 入口仍待接入，此 API 本身不是可部署命令。每份制品必须只包含一个有名称的模型；绑定按名称进行，
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
调度采用共享同步调用的默认优先级和任意 BPU 核心，不宣称提供自定义调度接口。

以下完整嵌入函数已针对公开头文件编译检查。它需要调用方传入模型组、词表及已核验特征，
内部创建真实预检回调；不是合成推理结果，也不是独立板端应用：

```cpp
#include "preflight.h"
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

模型、张量分配和推理任务复用 Ultralytics 的共享所有者／调用实现。新增多输入调用
支持 decoder 四输入，现有图像调用仍保留一／两输入约束。API 替身测试覆盖全部阶段、
乱序／带 padding 张量、两个 acoustic 别名、可选输出、自有结果、非法输入／元数据及
分配／推理／缓存失败释放。详见[SDK 验证](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-sdk-review.md)。

<a id="preflight"></a>
## 三模型整体预检

不依赖 SDK 的预检链接 `paraformer_preflight`，或链接已传递依赖它的 `paraformer_sdk`。
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
记录模型摘要；本地预期摘要**不能认证发布来源或证明 HBM 实现了所选模型**，仍须
进行实际运行时元数据校验。不要只为消除 mismatch 就重新计算并覆盖预期摘要。

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
隐式 S100P 回退。详见[预检证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-preflight-review.md)。
