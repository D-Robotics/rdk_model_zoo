# Paraformer Python 流程与 CPU 中间处理

[English](README.md)

当前目录已提供 CPU CIF（连续积分触发）实现与三模型应用编排。SDK 适配器和模型绑定也已实现；真实 CPU 音频前端也已提供；
完整 Python CLI 已提供，[C++ 桥接](../cpp/README_cn.md#quickstart)已可通过完整原生入口读取准备特征；真实板端推理尚未执行。
下述主机检查不代表模型推理或精度验证。归档的
[S 运行说明](../../../../../platforms/s/samples/speech/paraformer/runtime/python/README_cn.md)
保留作历史参考，不是统一入口。

先看 [CLI 用法](#usage)、[全部参数](#parameters) 和 [结果文件](#results)；它们前面的章节解释数值与 API 契约。

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

源清单仅发布 S100 模型，本次不声明 X5、S100P 或 S600 适配。真实 SDK 验证、板端三段推理和专用评测仍待完成；
[转换工具](../../conversion/README_cn.md) 已提供主机验证的 FP32 导出、真实音频校准及显式 OE 编排。原生应用主机验证单独记录在 C++ 说明中。板端推理、OE 编译、数据集 CER 与延迟均未执行。

<a id="stage-io"></a>
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

<a id="integration-example"></a>
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

### 板端集成 API（板端未执行）

下面是集成示意，**不是完整音频命令**。需要 S100、匹配的 `hbm_runtime`、
三个本地模型、准确词表与匹配前端生成的特征（见下文）；完整 CLI 见下文。此 API 示例还在主机上以真实前端和显式 SDK 替身执行，
其合成文本不是 HBM 预测结果。

```python
import json
from pathlib import Path
from samples.speech.paraformer.runtime.python.model_binding import resolve_selections
from samples.speech.paraformer.runtime.python.runtime import load_runtime

# Board-only integration: the model package must already be prepared.
import soundfile as sf
from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
vocabulary = json.loads(Path("samples/speech/paraformer/model/s100/tokens.json").read_text())
bundle = load_runtime(resolve_selections("s100"), vocabulary)
bundle.set_scheduling_params(priority=7, bpu_cores=[0])
sample = Path("samples/speech/paraformer")
waveform, rate = sf.read(sample / "test_data/audio/BAC009S0724W0121.wav", dtype="float32")
prepared = ParaformerFrontend(sample / "model/am.mvn").pre_process(waveform, rate)
result = bundle.pipeline.predict(prepared.tensor, prepared.valid_frames)
print(result.text, prepared.truncated)
```

调度可选。`priority` 为 0–255 整数，`bpu_cores` 为非空的非负整数序列；
实际硬件／核心组合由 SDK 判断。参数以模型名为键传递给**全部三个模型**。
任一 SDK 缺少 setter 时，在调用任何 setter 前拒绝。非法值由共享 runner 拒绝；
SDK 执行错误直接上抛，若前面的模型已接受设置，不保证回滚。两个参数均省略时
不调用 SDK setter。这修正了源包装器静默忽略调度参数的问题。

新增 8 项绑定／适配测试，整个套件共 33 项（含模型包、音频和 CLI 检查）：覆盖制品和路径身份、元数据顺序／
形状／类型／名称、声学输入别名及可选 count 输出、SDK 加载前拒绝，以及共享
runner 实际参与编排并向 SDK 替身传递调度。详见[绑定证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md)。
这些主机结果不代表真实 SDK 或板测通过，两者仍未执行。

## 真实 CPU 音频前端

[frontend.py](frontend.py) 调用实际的 FunASR `WavFrontend`，不以 NumPy 近似替代
fbank。输入为已加载的有限 float32 数组 `[samples]` 或 `[samples,channels]`，
样本与声道维度非空，采样率必须是整数 16000。文件加载位于数值类之外。
多声道按 float32 均值转单声道，不额外归一化波形；其他采样率拒绝，不自动重采样。

固定配置为 hamming 窗、80 mel 频带、25 ms 窗长、10 ms 帧移、LFR 堆叠 7／步长 6，
以及字节固定的 `am.mvn`。保留源 FunASR 默认的样本缩放和 dither。
短于 25 ms 的音频由 FunASR 缩短分析窗，10 ms 输入已与源实现对照；更短音频仍可能
不满足上游特征算法要求，此时异常直接上抛。不新增 VAD、标点恢复、时间戳、流式或热词能力。

`ParaformerFrontend(cmvn_path, random_seed=191009)` 先核对固定 CMVN，再加载依赖。
`pre_process(waveform, sample_rate)` 返回 `PreparedFeatures`：

| 字段 | 含义 |
| --- | --- |
| `tensor` | 独立持有的连续 float32 `[1,400,560]`，补齐行全零 |
| `valid_frames` | 提供给 encoder／CIF 的有效帧数，上限 400 |
| `original_frames` | 截断前的前端帧数 |
| `truncated` | 原帧数超过 400 时为 true |
| `sample_count` | 提取特征前的单声道样本数 |

将 `prepared.tensor` 与 `prepared.valid_frames` 传给 `bundle.pipeline.predict`。
保留源行为的前 400 帧，但显式报告原长度与截断；这**不是**长音频分块。
30 秒用例产生 500 个 LFR 帧，保留前 400 帧并标记截断。调用方必须展示这个标记，
不能把结果描述成整段完整转写。

默认种子与源 fbank dither 一致。CPU 随机状态只在调用期间改变，即使前端异常也恢复；
不重新设置 GPU 随机种子。本适配器的调用在随机状态上下文周围串行化，其他线程若也
使用 Torch 全局随机状态，仍需要应用层协调，不能理解为进程全局的并发隔离。
`random_seed` 接受 `[0,2**63)` 整数，改动会改变 dither，不属于固定种子的源对照范围。

<a id="environment"></a>
### 显式安装依赖

从仓库根目录创建仅用于主机前端的环境：

```bash
python3.12 -m venv .venv-paraformer
.venv-paraformer/bin/python -m pip install -r samples/speech/paraformer/runtime/python/requirements-frontend.txt
```

已验证环境为 macOS arm64／Python 3.12，Torch、torchaudio 2.6.0，FunASR 1.3.14，
NumPy 1.26.4，SoundFile 0.14.0，protobuf 4.23.0。依赖文件保留源直接版本约束；
完整实测依赖列表放在证据中，不是板端锁定文件。Linux 源脚本通过官方 CPU wheel
索引安装 Torch／torchaudio，需匹配 CPU 架构和 Python 版本。这个前端环境不提供
`hbm_runtime`，使用 SDK API 前还需单独准备板端运行环境。运行代码不调用 pip，
不自动创建虚拟环境。

### 无板卡运行前端

将上述环境作为 `python`（激活环境或替换解释器路径），从仓库根目录执行。
内置源 WAV 的预期输出为 `(1, 400, 560) 71 71 False`。FunASR 可能同时输出 ffmpeg
可用性提示；本示例通过 SoundFile 加载音频，不依赖 ffmpeg。

```bash
python - <<'PYCODE'
from pathlib import Path
import soundfile as sf
from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend

sample = Path("samples/speech/paraformer")
waveform, sample_rate = sf.read(sample / "test_data/audio/BAC009S0724W0121.wav", dtype="float32")
frontend = ParaformerFrontend(sample / "model/am.mvn")
prepared = frontend.pre_process(waveform, sample_rate)
print(prepared.tensor.shape, prepared.valid_frames, prepared.original_frames, prepared.truncated)
PYCODE
```

7 组真实 FunASR 用例——两条内置 WAV、派生立体声、静音、30 秒输入、25 ms 单窗和
10 ms 短窗——在记录的依赖环境中与固定 S 源前端逐字节一致。重复计算相同，复现了
源随机状态变化，新适配器在成功和注入后端错误时均恢复 CPU 随机状态。这证明特征
计算，不代表识别文本或板端推理通过。详见[前端证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-frontend-review.md)。

<a id="usage"></a>
## CLI 用法

默认调用（不指定模式或输入参数）会检测本机板型，使用默认模型路径，在 S100 上
推理内置清单的两条音频，写入全新的 `outputs/paraformer`。这需要真实 SDK 和模型。
帮助、模型列表、选择预览、真实前端准备可在无板环境运行，前端准备需要上述依赖环境。
所有模式都不下载文件、不安装依赖。以下命令均从仓库根目录执行。

```bash
# cwd: repository root; use the documented frontend environment as python
python samples/speech/paraformer/runtime/python/main.py --list-models
python samples/speech/paraformer/runtime/python/main.py --target s100 --dry-run
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

预处理成功时退出码为 0，`result.json` 的 status 为 `completed`，生成两个 NPY 特征
和独立 `prepared-manifest.json`；有效帧数应为 71、78，不改写原 `test_data/manifest.json`。
前两条命令只需基本主机依赖，不加载 Torch／FunASR／SDK。显式不支持的 target 即使
在预览／预处理模式也拒绝；dry-run 的 auto 必须改为显式 `--target s100`。
处理单条音频并使用不同输出目录：

```bash
# cwd: repository root
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --audio-file samples/speech/paraformer/test_data/audio/BAC009S0724W0168.wav --output-dir outputs/paraformer-one
```

在 S100 上显式准备模型包和运行环境后，使用下列推理命令；本轮无板迁移没有执行它：

```bash
# cwd: repository root; S100 with matching SDK and frontend dependencies only
bash samples/speech/paraformer/model/download_model.sh --target s100
python samples/speech/paraformer/runtime/python/main.py --target s100 --output-dir outputs/paraformer-inference
```

`bash samples/speech/paraformer/runtime/python/run.sh` 原样转发参数，并读取 `PYTHON`
环境变量选择解释器。不同于源包装器，它不会创建虚拟环境、安装依赖或下载模型；
也不会悄悄接受旧的位置参数形式数据目录。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `auto` | auto / x5 / s100 / s100p / s600；仅 S100 有制品 |
| `--list-models` | `false` | 列出三模型，不执行 |
| `--dry-run` | `false` | 预览选择，不加载模型文件/SDK；显式指定 target |
| `--preprocess-only` | `false` | 仅执行真实 CPU 前端；auto 选择 S100 配方，不检测板卡 |
| `--manifest` | `unset → test_data/manifest.json` | JSON 列表；与 audio-file 互斥 |
| `--audio-file` | `unset` | 单条 WAV；文件名去扩展名作为 ID |
| `--audio-dir` | `unset → manifest parent/audio` | 清单音频目录；不能与 audio-file 同用 |
| `--output-dir` | `outputs/paraformer` | 必须是尚不存在的目录 |
| `--max-utts` | `0` | 非负；0 为全部，正数为前 N 条 |
| `--cmvn-path` | `model/am.mvn` | 默认 Sample 内的固定 CMVN |
| `--tokens-path` | `model/s100/tokens.json` | 推理词表必须匹配固定内容；预处理不读取 |
| `--random-seed` | `191009` | CPU 前端种子；整数 [0,2**63) |
| `--priority` | `unset` | 可选 0–255；仅用于推理 |
| `--bpu-cores` | `unset` | 可选非负整数列表；仅用于推理 |
| `--encoder-model-path` | `unset → published default` | 外部路径：必须同时提供三个路径和对应 ID |
| `--encoder-asset-id` | `unset` | 准确阶段 ID；指定时需完整提供三项 |
| `--predictor-model-path` | `unset → published default` | 外部路径：必须同时提供三个路径和对应 ID |
| `--predictor-asset-id` | `unset` | 准确阶段 ID；指定时需完整提供三项 |
| `--decoder-model-path` | `unset → published default` | 外部路径：必须同时提供三个路径和对应 ID |
| `--decoder-asset-id` | `unset` | 准确阶段 ID；指定时需完整提供三项 |

模型、CMVN 和词表的默认值在运行时为 Sample 内绝对路径，仅输出默认路径相对于 cwd。
argparse 提供 `-h`／`--help`。三个模式参数互斥。清单必须非空，ID 为唯一文件名主体，
不能含 `/`、`\`、NUL，也不能是 `.` 或 `..`；可选 `text` 必须是字符串。
先校验全部记录结构，再按 max-utts 取前 N 条；选中音频缺失直接失败，不静默缩小集合。

<a id="results"></a>
## 结果文件与失败行为

- `result.json` 仅表示完成。记录 UTC 起止、声明目标／模型身份、实测输入／模型摘要、
  可为空的发布方摘要、前端种子、与预测分开的参考文本、帧数、截断及耗时。
- 预处理生成 `feats/<utt_id>.npy` 和 `prepared-manifest.json`。新条目保留原标注，添加
  重新计算的 `feat_length`、`original_frames`、`truncated`、相对 `feature_file` 及摘要。
  若原来有这些派生字段，只在新清单中更新，不改原文件。
- 推理条目另含 `text`、`token_ids`、`token_count`、`decoder_executed` 与 `timings_ms`，
  不导出 NPY。`metadata` 为实际绑定元数据。不能由两条转写推断数据集 CER。
  `frontend_ms` 不含文件加载，各阶段计时也不是完整端到端延迟。
- 输出目录创建后失败会写 `failed.json`，包含当前语音、此前已完成条目、错误类型／
  消息与已采集身份。前置检查可能在创建目录前失败，只输出 stderr；磁盘满或不可写
  导致失败记录也无法保存时，会同时报告这项错误，不保证凭空生成证据。

`inference_attempted` 表示进入过推理流程；`inference_executed` 在预处理时为 false，
成功获得推理结果后为 true，首轮尝试失败、无法确认执行完成时为 null。如果此前已有
成功语音，后续失败仍保留 true。它不是板测通过证明。零 token 明确记录跳过 decoder，
对应耗时为 null。

成功完成前复核音频、清单、CMVN、词表和模型字节未变化。已有输出目录拒绝使用；
中断运行可能保留部分特征，不能当作已完成。请选择全新输出路径，保留原证据。
机器读取应使用 `result.json`；FunASR 可能在 stdout 输出依赖提示，stdout 不保证纯 JSON。

<a id="troubleshooting"></a>
## 常见问题

| 现象 | 原因／处理 |
| --- | --- |
| Output directory must be new | 选择新路径并保留旧记录 |
| Missing selected WAV | 修正清单或音频目录，不会隐式跳过 |
| Paraformer requires 16000 Hz | 显式准备匹配音频，CLI 不重采样 |
| Target mismatch／无板型身份 | 主机使用 preprocess-only，实际推理需要 S100 |
| 词表或 CMVN 摘要不匹配 | 使用固定模型包，不能给其他模型文件改名冒充 |
| 张量名／类型／形状不匹配 | 核对实际元数据和发布身份 |

真实主机已检查帮助／列表／预览、两条默认 WAV、单条 WAV、前 N 条、重复输出目录、
错误采样率和主机推理拒绝。完整 CLI 推理分支仅以显式 SDK 替身运行实际共享 runner
和 pipeline，该测试不是 HBM 推理结果。详见[CLI 证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cli-review.md)。
