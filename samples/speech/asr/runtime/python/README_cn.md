# ASR Python runtime

[English](README.md) | 简体中文

<a id="overview"></a>
## Python 推理

本目录提供Python 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── asr.py  # Python 脚本
├── audio_io.py  # Python 脚本
├── cli.py  # 参数与结果展示
├── decoding.py  # Python 脚本
├── frontend.py  # Python 脚本
├── main.py  # 命令行入口
├── model_binding.py  # Python 脚本
├── model_runner.py  # Python 脚本
├── postprocess.py  # Python 脚本
├── run.sh  # 运行示例
└── vocabulary.py  # Python 脚本
```

<a id="environment"></a>
## 环境
Python 3.10+、NumPy、PyYAML、SciPy、SoundFile（依赖 libsndfile）。S100/S600 推理需要匹配的板端系统及其 `hbm_runtime`。运行推理前请先[准备模型](../../model/README_cn.md)；运行时不安装依赖或下载模型。

<a id="usage"></a>
## 使用
从仓库根目录运行。默认命令识别板卡、选择对应制品并处理随附录音。每次使用一个新输出目录。
```sh
# Repository root; matching board and explicitly downloaded model required.
python3 samples/speech/asr/runtime/python/main.py
python3 samples/speech/asr/runtime/python/main.py --target s600 --decode-mode legacy --output-dir outputs/asr-legacy
```
处理成功时命令返回 0，且 `result.json` 记录 `status: completed`。帮助、列表和显式 target 的 dry-run 无需板端 SDK；dry-run 用于解析目标和制品选择，不会执行推理。

<a id="parameters"></a>
## 参数
| Parameter | Type | Default | Meaning / 含义 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Detect board; host dry-run requires s100/s600 / 自动识别，主机预检须指定 |
| `--asset-id` | str | `None` | Exact published identity / 精确制品身份 |
| `--model-path` | str | `None` | External path requires asset-id / 外部路径必须同时指定身份 |
| `--audio-file` | Path | `samples/speech/asr/test_data/chi_sound.wav` | Audio input / 音频输入 |
| `--vocab-file` | Path | `samples/speech/asr/test_data/vocab.json` | Hash-pinned vocabulary / 固定哈希词表 |
| `--audio-maxlen` | int | `30000` | Fixed compiled length / 编译固定长度 |
| `--new-rate` | int | `16000` | Fixed sample rate / 固定采样率 |
| `--decode-mode` | str | `ctc` | ctc or legacy / CTC 或源实现兼容模式 |
| `--priority` | int | `0` | Scheduling priority 0–255 / 调度优先级 |
| `--bpu-cores` | int list | `[0]` | Nonnegative core IDs / 非负核心编号 |
| `--output-dir` | Path | `outputs/asr` | Must be new / 必须为新目录 |
| `--list-models` | bool | `false` | List publications without SDK / 无 SDK 列举 |
| `--dry-run` | bool | `false` | Resolve without inference / 只解析不推理 |
默认音频、词表相对 sample 定位，不受调用目录影响。用户传入的相对路径与输出路径相对当前目录。列表和 dry-run 互斥。前端参数与编译模型固定，不能通过改参数让模型支持其他输入形状。

<a id="results"></a>
## 结果
`result.json` 包含目标、制品身份、本地模型/音频/词表 SHA-256、发布方摘要（若有）、实测张量元数据、前端配置和解码模式。`chunks` 逐块记录索引、原始帧偏移/数量/采样率、有效重采样点数和文本；`text` 直接拼接各块文本，不额外插入分隔符。不输出置信度或时间戳。创建输出目录后的失败会写入 `failed.json`，其中包含已完成块和错误详情；更早的失败会输出错误。若失败报告无法写入，stderr 会同时显示推理错误和报告写入错误。错误返回 2。

<a id="integration-example"></a>
## 集成示例
在 S100 的仓库根目录执行，先按模型文档准备模型。录音和词表已随仓库提供；S600 将选择目标改为 `s600`。
```python
from samples.speech.asr.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.speech.asr.runtime.python.model_runner import RuntimeModelRunner
from samples.speech.asr.runtime.python.vocabulary import load_vocabulary
from samples.speech.asr.runtime.python.audio_io import read_chunks
from samples.speech.asr.runtime.python.asr import ASR

selection = resolve_selection("s100")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = ASR(runner, binding, load_vocabulary(SAMPLE_DIR / "test_data/vocab.json"))
texts = []
for chunk in read_chunks(SAMPLE_DIR / "test_data/chi_sound.wav", task.config):
    prediction = task.predict(
        chunk.waveform, chunk.sample_rate, return_details=True)
    texts.append(prediction.text)  # prediction.prepared 持有该块自身几何信息
print("".join(texts))
```

<a id="stage-io"></a>
## 三阶段接口
- `preprocess(waveform, sample_rate)` 接收有限浮点 `[frames]` 或 `[frames,channels]` 波形，长度上限为 `ceil(30000 × 原采样率 / 16000)`。先均值混为单声道，使用 SciPy Fourier 重采样，以 `sqrt(var + 1e-5)` 归一化，再补零；返回自有 float32 `[1,30000]` 张量及几何信息。
- `infer({input_name: tensor})` 只调用 runner 一次；runner 验证名称/形状/类型并返回拥有独立内存的原始输出，不做解码或激活。
- `postprocess(raw)` 校验实测 `[1,T,3503]` 元数据，argmax 解码为字符串。FLOAT32 输出照旧直接使用；整数 SCALE 输出经共享量化模块以 float64 比较精度反量化，argmax 前不同整数的大小关系不会丢失——float32 会把 `2**24` 与 `2**24 + 1` 这类相邻整数舍入成假平局。argmax 无需 softmax。
- `predict(waveform, sample_rate, *, return_details=False)` 组合单块三阶段，默认返回解码文本。`return_details=True` 时返回 `ChunkPrediction(text, prepared)`：同一文本加上本次调用的 prepared 块（自有张量、有效采样数、源几何信息）。每次调用仅发起一次 runner 请求，模型不保留逐调用状态。CLI 使用此形式记录逐块结果。读文件、载入词表、保存结果都在调用方，不放入 `ASR` 模型类。

既有 `pre_process`、`forward`、`post_process` 名称仍是 `preprocess`、`infer`、`postprocess` 的可导入薄别名——同一实现，两个名称。

CTC 先折叠连续相同 ID，再去掉 blank 0；legacy 只去 blank。`[5,5,0,5]` 在 CTC 下为 `AA`，legacy 为 `AAA`。只有完全相等的分数才平局并取最小 ID：float32 输出按 float32 比较，整数 SCALE 输出按 float64 比较，不同整数不会因舍入变成假平局。所有非 blank 词表字符串按原文保留，包括 `|` 和特殊 token。不跨块保留状态或去重。末块补零后仍解码全部输出帧，因为模型元数据不提供有效帧数。独立窗口可能截断词语，分块流程不会执行带重叠拼接。Python 使用 Fourier 重采样，C++ 运行时使用 sinc 重采样。

<a id="troubleshooting"></a>
## 排错
| 错误 | 处理 |
| --- | --- |
| `Host dry-run requires an explicit target` | 指定 `--target s100` 或 `s600`。 |
| `ASR is published only for s100 and s600` | 不用 S100 模型冒充 S100P/X5。 |
| `An external model path requires the exact --asset-id` | 按模型文档提供精确身份。 |
| `Output directory must be new` | 改用新目录，保留旧证据。 |
| `ASR input must be float32 [1,30000] for the fixed frontend` | 核对实际制品和 SDK 元数据，不能只看文件名。 |
| `Audio file changed during streaming` | 固定输入文件后另选目录重跑。 |
