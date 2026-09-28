# Paraformer 原生数值库

[English](README.md) · [Sample 概览](../../README_cn.md)

当前目录提供原生 CPU CIF／文本解码及三模型应用编排。SDK 适配器、准备清单读取
和完整原生可执行入口仍在迁移。以下测试是主机检查，不是 HBM 推理。
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

成功标准是两个 CTest 检查通过：数值契约与合成三模型编排。上述命令在 Clang/GNU
下启用地址／未定义行为检查，Release 测试也保留断言。生产构建产物为静态
`paraformer_contract` 库，测试可执行文件不是推理 CLI。

<a id="parameters"></a>
## 构建选项与当前入口

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `PARAFORMER_BUILD_TESTS` | `OFF` | 构建并注册两个主机测试 |
| `PARAFORMER_SANITIZERS` | `OFF` | 在 Clang/GNU 下启用 ASan/UBSan 并传播链接选项 |
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
本库不加载模型、不选板型、不读写文件、不设置调度，也不编译假 SDK 回退。

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

两个原生测试覆盖分数积分、padding、空输出、截断、非法契约、重复／特殊／BPE token、
平局 ID、回调顺序、零 token 跳过与错误中间张量。对照驱动在 27 组数据上与提取的
固定源 C++ CIF 及统一 Python 逐字节比较，并做 20 组原生／Python 文本对照。
准确编译器和复现脚本见报告。提取的源函数／驱动只是主机证据，不是另一份维护中的运行实现。

Sanitizer 运行库构建失败时应检查编译／链接器支持；`PARAFORMER_SANITIZERS=OFF`
可关闭插桩，但不能据此宣称完成 sanitizer 检查。真实 HBM 的形状／类型／名称／身份
核验属于后续 SDK 适配器，主机数组检查不能替代。真实 SDK 构建、板端推理、完整原生
CLI、OE 与 CER 均按实际状态保持未执行或待完成。
