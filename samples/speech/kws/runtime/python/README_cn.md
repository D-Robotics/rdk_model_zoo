[English](README.md) | 简体中文

# KWS Python runtime

<a id="overview"></a>
## Python 推理

在 S100 上检测唤醒词。`KWS.from_model` 初始化 Runtime；`predict` 计算 fbank 特征、运行模型并返回概率分数。

<a id="directory"></a>
## 目录结构

```text
python/
├── audio_io.py  # 音频文件读取
├── cli.py  # 参数、模型选择与结果展示
├── frontend.py  # 音频特征准备
├── kws.py  # 模型阶段与预测
├── main.py  # 命令行入口：构造模型并调用 predict
├── model_binding.py  # 模型选择与物理张量契约
└── run.sh  # 定位 Python 入口并转发参数
```

<a id="environment"></a>
## 环境

命令从仓库根目录执行。任务需要 Python 3.10+、NumPy、PyYAML、SoundFile、PaddlePaddle、PaddleAudio，以及 S100 BSP 的 `hbm_runtime`。help/list/dry-run 只需轻量选择模块（PyYAML）及 NumPy 选项检查；真实执行前不加载板端 SDK。

请在独立 Python 环境中安装前端依赖。环境示例：

```sh
python3 -m venv /path/to/kws-venv
/path/to/kws-venv/bin/python -m pip install numpy PyYAML soundfile paddlepaddle paddleaudio tqdm scipy resampy scikit-learn
```

PaddleAudio 导入数据集/指标模块，因此即使只用 fbank 也需要 tqdm/scikit-learn。启动器不向系统自动装包；编译的 `hbm_runtime` 应来自板卡镜像，且所选环境可以访问。

<a id="usage"></a>
## 使用

主机只读命令：

```bash
bash samples/speech/kws/runtime/python/run.sh --help
bash samples/speech/kws/runtime/python/run.sh --list-models
bash samples/speech/kws/runtime/python/run.sh --target s100 --dry-run
```

完成[显式模型准备](../../model/README_cn.md)后，在 S100 上运行：

```sh
bash samples/speech/kws/runtime/python/run.sh --target s100 \
  --audio-file samples/speech/kws/test_data/sample.wav --output-dir outputs/kws-run1
```

`run.sh` 不下载、不安装、不改变调用目录。默认模型/音频相对 sample，传入相对路径及输出位置相对调用目录。`--target auto` 识别实际板卡；主机 dry-run 必须指定目标，不验证模型字节或 SDK 描述符。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | auto | 实际目标，仅 S100 有制品 |
| `--asset-id` | `null` | 精确 `s:kws:s100/kws.hbm` |
| `--model-path` | `null` | 已有文件，需精确 asset ID |
| `--audio-file` | `samples/speech/kws/test_data/sample.wav` | 单声道 16 kHz，解码为 float32 |
| `--output-dir` | `outputs/kws` | 保存 `result.json` 的新目录 |
| `--audio-maxlen` | 60000 | 发布前端固定采样点数 |
| `--frame-shift` | 10 | 固定毫秒数 |
| `--frame-length` | 25 | 固定毫秒数 |
| `--n-mels` | 80 | 固定特征宽度 |
| `--priority` | 0 | SDK 调度优先级 0..255 |
| `--bpu-cores` | `[0]` | 一个或多个非负 SDK 核索引 |
| `--threshold` | 0.5 | 有限 [0,1]，分数大于等于阈值时判正 |
| `--list-models` / `--dry-run` | false | 互斥只读模式 |

`--audio-maxlen`、`--frame-shift`、`--frame-length` 和 `--n-mels` 必须匹配该模型发布的取值。改变窗口或特征维度需要为新特征契约导出的模型。调度经共享 runner 设置；SDK 不支持时会报错。

<a id="results"></a>
## 输出

CLI 写入并打印 `result.json`：分数/判定、阈值规则、选定身份、实际 metadata、模型/音频 SHA-256，以及原始/使用/补零/截断数量。分数是 Python float 概率，不是时间区间或转写文本。报告复制清单的 `publisher_sha256`，该值可以为 null。拒绝复用目录。前处理、metadata 或概率契约错误退出 2，失败预测不写结果报告。

<a id="integration-example"></a>
## 库接口

S100 上准备好模型和前端依赖后，下面示例读取随附音频、加载 runner 并逐阶段调用。

```python
from samples.speech.kws.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.speech.kws.runtime.python.audio_io import load_audio
from samples.speech.kws.runtime.python.kws import KWS
selection = resolve_selection("s100")
audio, sample_rate = load_audio(SAMPLE_DIR / "test_data/sample.wav")
task = KWS.from_model(selection)
task.set_scheduling_params(priority=0, bpu_cores=[0])
score = task.predict(audio, sample_rate)
print(score)
# Explicit stages are available when intermediate tensors are needed.
```

<a id="stage-io"></a>
## 阶段及生命周期

`preprocess` 接受非空、有限、单声道 float32 `[N]` 波形，幅值 [-1,1]，采样率 16000。复制前 60000 个点，短段补零，不暗中重采样或平均声道。PaddleAudio fbank 使用 25 ms 帧、10 ms 移动、80 bins 及源默认值（包括 dither=0、snip_edges=True），得到 373 帧。具名 `[1,373,80]` 张量独立持有连续数据。

`infer` 仅调用一次共享 runner，返回原始输出，不做 sigmoid、反量化、文件访问或最大值计算。runner 核验名称/形状/类型/有限值并复制 SDK 输出，后续调用不会覆盖旧结果。`postprocess` 核验绑定输出，仅对整数用共享 SCALE 转换，要求结果在 [0,1] 后取最大值；float 输出即使附带历史量化描述符也不再转换，不额外 sigmoid。

`predict` 仅组合三阶段，不缓存上一次音频状态。既有 `pre_process`、`forward`、`post_process` 名称仍是 `preprocess`、`infer`、`postprocess` 的可导入薄别名——同一实现，两个名称。实例用于串行执行，不承诺 SDK 并发安全。`frontend.py` 准备特征；`kws.py` 负责评分与 Runtime 初始化；`audio_io.py` 读取文件，`cli.py` 保存报告。

<a id="troubleshooting"></a>
## 排错

| 错误 | 处理 |
| --- | --- |
| 仅发布 s100 / 目标不匹配 | 使用真实 S100 和对应制品，不冒充 S100P/S600 |
| 缺模型或 SDK | 显式下载并使用 BSP runtime 环境 |
| 缺 PaddleAudio 依赖 | 显式准备前端环境，查看缺失包名 |
| 输入必须为 [1,373,80] | 检查制品和前端版本，不任意 reshape 绕过检查 |
| 要求单声道/16000 Hz | 显式转换音频并保留转换后输入身份 |
| 概率或 SCALE 无效 | 核对模型/SDK 描述符，不猜 sigmoid 或 scale |
| 输出目录必须为新目录 | 为本次运行选择新目录 |

随附音频仅为一个正例。选择阈值前，应准备正负有标签集合并使用[评估器](../../evaluator/README_cn.md)。
