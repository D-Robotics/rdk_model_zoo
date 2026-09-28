# Paraformer Python 流程与 CPU 中间处理

[English](README.md)

当前目录已提供 CPU CIF（连续积分触发）实现与三模型应用编排。SDK 适配器和模型绑定也已实现；真实音频前端、
命令行入口及 C++ 桥接仍在迁移，当前还不是可直接上板运行的完整 Sample。
下述主机检查不代表模型推理或精度验证。归档的
[S 运行说明](../../../../../platforms/s/samples/speech/paraformer/runtime/python/README_cn.md)
保留作历史参考，不是统一入口。

## 在流程中的位置

源流程为：音频 → FunASR 前端 → encoder → predictor → CPU CIF → decoder → 文本。
CIF 接收 predictor 的权重和隐藏状态，生成 decoder 使用的固定形状声学嵌入与
有效 token 数。它不执行模型、不读写文件、不解码词表，也不选择板卡。
独立的 [cif.py](cif.py) 让运行与校准复用同一份数值逻辑，避免把辅助函数塞进模型推理类。

## 依赖与可执行主机示例

需要 Python 和 NumPy；仅运行此数值模块不需要 Torch、FunASR、厂商 SDK 或板卡。
在仓库根目录执行：

```bash
python - <<'PYCODE'
import numpy as np
from samples.speech.paraformer.runtime.python.cif import cif_numpy

weights = np.zeros((1, 401), dtype=np.float32)
hidden = np.zeros((1, 401, 512), dtype=np.float32)
weights[0, :3] = [0.75, 0.75, 0.5]
hidden[0, :3] = np.array([2, 6, 10], dtype=np.float32)[:, None]
embeddings, token_count = cif_numpy(weights, hidden, real_T=3)
print(embeddings.shape, token_count.tolist(), embeddings[0, :2, 0].tolist())
PYCODE
```

预期输出：`(1, 100, 512) [2] [3.0, 8.0]`。两次累计权重跨越整数边界，分别积分
得到对应的声学嵌入。这是合成数据示例，不是语音识别结果。

## 接口契约

| 参数／返回值 | 契约 |
| --- | --- |
| `alphas` | 有限且非负的 `float32 [1,401]` 权重 |
| `concat5` | 有限的 `float32 [1,401,512]` 隐藏状态 |
| `real_T` | 必须显式指定；推理使用 `0…400` 整数，仅无屏蔽校准使用 `None` |
| 声学嵌入 | 独立持有的 `float32 [1,100,512]`，未使用的行补零 |
| token 数 | 独立持有的 `int32 [1]`，最大为 100 |

推理时先将 `real_T` 及之后的权重置零，再累计。有效帧为零或总权重不足 1 时，
返回零嵌入和零计数；调用方须处理此计数，本函数不决定是否执行 decoder。
不足下一整数的残余权重不产生 token；超过 100 个嵌入时保留前 100 个，与源契约一致。

计算保留源实现的 float64 累加后转 float32，以及每帧最多触发一次的规则；它不是
面向权重大于 1 的通用多次触发积分器。输入不会被修改。形状、类型、有限性、
权重非负性及有效帧数不符合契约时直接报错，不自动转换类型或扩展 batch。
`real_T=None` 保留源校准流程的无屏蔽分布，不能用来替代推理时的 padding 屏蔽。

## 验证与剩余工作

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

7 项 CIF 主机测试覆盖空输出、手算分数边界、padding 与校准模式、100 token 截断、
24 组源实现对照、输入所有权与非法契约。对照使用归档的 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`；源实现无 token 时抛出 `IndexError`，
统一实现已修复。详见[评审记录与证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cif-review.md)。

源清单仅发布 S100 模型，本次不声明 X5、S100P 或 S600 适配。真实前端等价性、
真实 SDK 验证、板端三段推理、原生 C++、完整转换和评测流程以及各层双语 Sample 文档
仍待完成。板端推理、OE 编译、数据集 CER 与延迟均未执行。

## 三模型应用流程

[pipeline.py](pipeline.py) 组合三个独立的原始模型调用对象。每个对象接收
“物理输入名 → 数组”字典，返回“物理输出名 → 数组”字典。`TensorNames` 显式
提供模型绑定解析出的准确名称，流程不按首个输出或子串猜测。这是应用层
编排，不是在某个模型 `forward` 内夹入 CPU 处理和多次 SDK 调用。

| 阶段 | 所需输入 | 所需输出 |
| --- | --- | --- |
| Encoder | float32 `[1,400,560]` | float32 context `[1,400,512]` |
| Predictor | context `[1,400,512]` | float32 权重 `[1,401]`、hidden `[1,401,512]` |
| CPU CIF | predictor 数组及有效帧数 | float32 acoustic `[1,100,512]`、int32 count `[1]` |
| Decoder | context、acoustic、count、全零 float32 bias `[1,1,512]` | float32 logits `[1,100,8404]` |

`predict(features, feature_length)` 接收有限 float32 特征与 1–400 的有效帧数。
每个被使用的阶段边界均校验形状、类型和有限性；复制中间数组，避免后续 runner
复用缓冲区时覆盖保留的 encoder context。流程不加载 SDK，也不设置调度；物理模型
校验与调度由下述运行适配器负责。制品名称中的 INT16 不能证明物理输入输出类型。

`Prediction` 包含文本、选中 token ID、CIF 计数、各阶段毫秒耗时和
`decoder_executed`。文本沿用 S 源规则：在有效前缀上取 argmax，过滤 `<...>`
包围的特殊 token，移除 `@@`，直接拼接。重复 token 不折叠，不能套用 CTC。
计数和 ID 列表包含随后被过滤的特殊 token。CIF 计数为零时跳过 decoder，返回
空文本、空 ID、`decoder_executed=False`，decoder 耗时为 `None`。
任何阶段异常直接上抛，不生成成功结果。

计时分别覆盖三次 runner 调用与 CPU CIF，不包括前端、加载、调用之外的校验／复制、
文本解码及文件 I/O；它们的总和不代表端到端延迟。本轮未测 SDK 或板端性能。

### 可执行的合成流程示例

下例注入三个合成函数而非模型，使用 NumPy 演示实际 CPU 编排与文本解码。
从仓库根目录执行，预期输出 `中中文 3 True`。

```bash
python - <<'PYCODE'
import numpy as np
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline, TensorNames

names = TensorNames("speech", "context", "context", "alphas", "hidden",
                    "context", "count", "bias", "acoustic", "logits")
vocabulary = [f"token{i}" for i in range(8404)]
vocabulary[3] = "中"
vocabulary[4] = "文@@"

def encoder(inputs):
    return {"context": np.zeros((1, 400, 512), np.float32)}

def predictor(inputs):
    weights = np.zeros((1, 401), np.float32)
    weights[0, :3] = 1
    return {"alphas": weights, "hidden": np.ones((1, 401, 512), np.float32)}

def decoder(inputs):
    logits = np.zeros((1, 100, 8404), np.float32)
    logits[0, np.arange(3), [3, 3, 4]] = 1
    return {"logits": logits}

pipeline = ParaformerPipeline(encoder, predictor, decoder, names, vocabulary)
result = pipeline.predict(np.zeros((1, 400, 560), np.float32), 3)
print(result.text, result.token_count, result.decoder_executed)
PYCODE
```

另有 6 项流程测试覆盖调用顺序和具名输入、padding 屏蔽、重复字符／BPE／特殊 token、
空结果跳过 decoder、非法前端及 decoder 数据、词表大小与 predictor 缺失输出。
从活动清单 URL 读取的发布词表有 8,404 个不重复条目，实测 SHA-256 为
`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`。
此下载摘要不等于发布方补齐了清单中目前为空的哈希，也不能验证任何编译模型。
详见[流程证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-pipeline-review.md)。

## 发布制品选择与运行绑定

[model_binding.py](model_binding.py) 读取活动 S 发布清单。
`resolve_selections("s100")` 按 encoder、predictor、decoder 顺序返回三项选择；
`auto` 使用共享的本机板型检测。X5、S100P、S600 没有对应发布组合，明确拒绝。
选择过程不加载 SDK、不连接板卡、不下载文件，也不代表模型文件已经存在。

从仓库根目录执行以下主机示例，会输出三项完整制品 ID，并明确拒绝 S100P：

```bash
python - <<'PYCODE'
from samples.speech.paraformer.runtime.python.model_binding import resolve_selections
for selection in resolve_selections("s100"):
    print(selection.stage, selection.asset.reference)
try:
    resolve_selections("s100p")
except ValueError as error:
    print(error)
PYCODE
```

默认文件位于 `samples/speech/paraformer/model/s100/`，保留发布名称：
`paraformer_large_encoder_400x560_s100.hbm`、
`paraformer_large_predictor_400x512_s100.hbm`、
`paraformer_large_decoder_400x512_s100.hbm`。外部路径必须同时传入完整的
`model_paths={"encoder": ..., "predictor": ..., "decoder": ...}` 和
`asset_ids={"encoder": ..., "predictor": ..., "decoder": ...}`；每项 ID 必须准确
匹配其阶段的发布制品。缺项、混用、重新标记阶段均拒绝。活动清单没有这些文件的
发布方 SHA-256，计算本地摘要仅记录字节身份，不能独立证明官方来源或模型兼容性。

每个 HBM 必须暴露一个模型。所有物理输入输出的名称、形状和类型都必须匹配，
顺序不影响绑定。名称依据固定 S 转换图与原生查找代码：

| 角色 | 准确名称 |
| --- | --- |
| Encoder 输入 | `speech` |
| Encoder 输出／predictor 输入／decoder context | `/encoder/after_norm/Add_1_output_0` |
| Predictor 权重 | `/predictor/Add_output_0` |
| Predictor hidden | `/predictor/Concat_5_output_0` |
| Decoder count／bias | `token_num`／`bias_embed` |
| Decoder acoustic | `onnx::Shape_8609` 或源 Python 支持的 `shape_8609` |
| Decoder logits | `logits` |

Decoder 若暴露可选的 `token_num` 透传输出，它必须是 int32 `[1]`。其余额外张量、
重复或歧义名称、错误形状和类型一律拒绝。形状见前述流程表，只有 count 使用 int32，
其他张量均为 float32。若实际编译接口不同，应核定并显式适配，不能悄悄转换类型。

[runtime.py](runtime.py) 创建三个共享 `NamedArrayRunner`。
`load_runtime` 在创建任何 SDK 对象之前校验完整组合。正常路径中，每个 runner
先核对本机板型和本地模型文件，再导入／创建 `hbm_runtime`，然后绑定实际元数据。
仅测试通过显式 `runtime_factory` 注入 SDK 替身并跳过真实板型／文件门禁；
这个接口不能作为客户部署模式。

### 板端集成 API（本轮未执行）

下面是集成示意，**不是完整音频命令**。需要 S100、匹配的 `hbm_runtime`、
三个本地模型、准确词表与匹配前端生成的特征；当前尚未提供完整音频 CLI。

```python
import json
from pathlib import Path
from samples.speech.paraformer.runtime.python.model_binding import resolve_selections
from samples.speech.paraformer.runtime.python.runtime import load_runtime

# Board-only integration sketch: files must already exist; features must be
# generated by the matching FunASR frontend (still being migrated).
vocabulary = json.loads(Path("samples/speech/paraformer/model/s100/tokens.json").read_text())
bundle = load_runtime(resolve_selections("s100"), vocabulary)
bundle.set_scheduling_params(priority=7, bpu_cores=[0])
# features: float32 [1,400,560]; feature_length: valid integer 1..400.
result = bundle.pipeline.predict(features, feature_length)
```

调度可选。`priority` 为 0–255 整数，`bpu_cores` 为非空的非负整数序列；
实际硬件／核心组合由 SDK 判断。参数以模型名为键传递给**全部三个模型**。
任一 SDK 缺少 setter 时，在调用任何 setter 前拒绝。非法值由共享 runner 拒绝；
SDK 执行错误直接上抛，若前面的模型已接受设置，不保证回滚。两个参数均省略时
不调用 SDK setter。这修正了源包装器静默忽略调度参数的问题。

新增 8 项绑定／适配测试，整个套件共 23 项（含 2 项模型包准备检查）：覆盖制品和路径身份、元数据顺序／
形状／类型／名称、声学输入别名及可选 count 输出、SDK 加载前拒绝，以及共享
runner 实际参与编排并向 SDK 替身传递调度。详见[绑定证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md)。
这些主机结果不代表真实 SDK 或板测通过，两者仍未执行。
