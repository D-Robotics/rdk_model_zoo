# Paraformer 评测

使用 CPU 上的三个 FP32 ONNX 模型，或 OE 环境的 HMCT executor 与三个量化
`*_ptq_model.onnx` 模型，对预处理后的 16 kHz 语音特征进行评测。这是主机仿真，
不是 HBM 执行或板端性能测试。评测复用运行时的 encoder → predictor → CPU CIF →
decoder 流程及贪心解码，不自动下载数据集、模型或工具链。

## 环境与准备

以下命令均在仓库根目录执行。实际验证环境为 Python 3.12、NumPy 1.26.4、
ONNX 1.17.0、ONNX Runtime 1.20.1。需要预处理音频时，在独立环境安装完整的
[导出及前端依赖](../conversion/requirements-export.txt)。HMCT 来自匹配的厂商 OE
环境，不要从不明包源安装同名替代包。本次迁移尚未执行真实 HMCT 评测。

先按照[转换说明](../conversion/README_cn.md)，将三个自包含 ONNX 模型导出到
`outputs/paraformer-export`。按照[模型准备](../model/README_cn.md)获取固定版本
`tokens.json`，其顺序必须与 8404 类解码器一致。无需板卡即可用真实前端准备两条内置语音：

```bash
python samples/speech/paraformer/runtime/python/main.py \
  --target s100 --preprocess-only \
  --manifest samples/speech/paraformer/test_data/manifest.json \
  --audio-dir samples/speech/paraformer/test_data/audio \
  --output-dir outputs/paraformer-features
```

每次运行使用新的输出目录。评测数据集时，替换为自己的 WAV 清单和音频目录。
音频必须为 16 kHz，前端不静默重采样；超过 400 个 LFR 帧会截断并记录元数据。
如果参考文本仍覆盖完整语音，截断会影响 CER，不能忽略此条件直接引用数据集精度。

## FP32 评测

```bash
python samples/speech/paraformer/evaluator/main.py \
  --pipeline fp32 \
  --encoder outputs/paraformer-export/encoder.onnx \
  --predictor outputs/paraformer-export/predictor.onnx \
  --decoder outputs/paraformer-export/decoder.onnx \
  --manifest outputs/paraformer-features/prepared-manifest.json \
  --vocab samples/speech/paraformer/model/s100/tokens.json \
  --output-dir outputs/paraformer-eval-fp32
```

成功返回 0，生成 `evaluation.json`，其中 `status: completed`，包含全部选中条目和
汇总指标。流程成功不代表识别准确率达标，程序没有内置 CER 放行阈值。
执行或校验失败返回 2，在新建输出目录保留失败报告；已有输出目录会直接拒绝，保持原文件不变。

## 量化仿真

在 OE 成功编译后，从每个阶段的实际输出目录找到 `*_ptq_model.onnx`。
在对应的 OE 环境执行：

```bash
python samples/speech/paraformer/evaluator/main.py \
  --pipeline int16 \
  --encoder /absolute/path/to/paraformer_encoder_int16_ptq_model.onnx \
  --predictor /absolute/path/to/predictor_int16_ptq_model.onnx \
  --decoder /absolute/path/to/decoder_int16_ptq_model.onnx \
  --manifest outputs/paraformer-features/prepared-manifest.json \
  --vocab samples/speech/paraformer/model/s100/tokens.json \
  --output-dir outputs/paraformer-eval-int16
```

三个模型路径是占位符，必须替换为实际制品。适配器保留源脚本使用的
`ORTExecutor(path).create_session().forward(feed)` API。`int16` 只选择执行器，
不能证明输入图的量化精度或编译产物正确性；本入口不接受 HBM 文件。

## 参数与输入契约

| 参数 | 默认值／含义 |
|---|---|
| `--pipeline` | 必填：`fp32` 为 CPU ORT，`int16` 为 HMCT 仿真 |
| `--encoder`、`--predictor`、`--decoder` | 必填，自包含 ONNX 路径；别名为 `--enc`、`--pred`、`--dec` |
| `--manifest` | 必填，预处理清单或历史 JSON 列表 |
| `--vocab` | 必填，已发布词表；检查 SHA-256 和 8404 个有序 token |
| `--output-dir` | 必填，必须是新目录 |
| `--max-utts` | `0` 表示全部；正整数选取前缀，但仍校验整个清单结构 |
| `--threads` | `4`，仅控制 FP32 ORT 算子内线程；算子间为 1；不覆盖 HMCT 线程设置 |

每条清单必须包含唯一且不带路径的 `utt_id`、字符串 `text`（允许空参考文本），以及
1–400 范围内的整数 `feat_length`。预处理清单提供 `feature_file`，可以是相对清单
目录的路径或绝对路径；可选 `feature_sha256` 和一致的 `original_frames`、`truncated`。
历史条目没有 `feature_file` 时，读取清单旁的 `feats/<utt_id>.npy`。
未知注释字段原样保留。每个选中特征必须是有限的 float32 `[1,400,560]` 单个 NPY
数组；缺文件、非有限数值、错误类型、摘要不符、NPZ 或尾随数据都会明确失败。

先校验整个清单，再选择前缀；只打开选中条目的特征文件，并对实际读入字节计算摘要。
模型按准确的语义名称、形状和类型绑定输入输出，支持已声明的 acoustic 名称别名及
解码器可选的 `token_num` 输出，不按位置猜测，也不隐式转换类型。

## 输出与指标口径

`evaluation.json` 记录 UTC 起止时间、执行器及版本、模型／清单／词表路径和摘要、
张量接口、选中条目数，以及每条的源元数据、特征摘要、参考文本、识别文本、token
ID／数量和阶段耗时。结束前再次检查模型、清单和词表摘要。失败时保留已完成条目和
`current_utterance`，但 `metrics` 保持 null，不把部分结果包装成完整评测。

CER 为 Unicode 字符编辑距离之和除以参考字符总数，`cer` 字段存比例而非百分比。
同时保留替换、删除、插入次数及逐条错误。不进行空白、大小写或标点归一化；全部
参考文本为空时 CER 为 null，但插入次数仍有意义。解码删除 `<...>` 特殊 token 和
`@@` 标记，直接拼接，不进行 CTC 重复折叠。旧评测脚本没有去掉 `@@`，新入口与
统一运行时保持一致。CIF 零 token 时输出空文本并跳过 decoder。
阶段耗时不含前端、模型加载及文件 I/O，不是 BPU 耗时或端到端延迟。

## 历史结果与验证边界

[归档 S 转换说明](../../../../platforms/s/samples/speech/paraformer/conversion/README_cn.md)
记录 AISHELL dev 300 条语音的 CER：FP32 5.20%、HMCT INT16 5.02%、S100 Python
3.13%、S100 C++ 3.13%。这些是源分支历史记录，本次没有复现，也不据此断言某后端更准确。

本次 CPU FP32 对两条内置语音实际测得 4 次编辑／28 个参考字符，CER 14.2857%。
这是小规模流程验证，不是上述 300 条基准的复测；具体源脚本对照与证据见
[评测记录](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-evaluator-review.md)。
真实 HMCT、OE 编译、HBM 执行、板端延迟和完整数据集精度仍未验证。转换文档已披露
随机输入的 Torch／ORT 差异，两条语音成功不能证明任意输入等价。

## 故障处理与源码阅读

缺少 HMCT 时应使用匹配的 OE 环境，不能把带厂商量化算子的图直接交给普通 CPU ORT
代替。词表摘要不符要获取正确发布版本，改文件名无效；帧数元数据不符要从对应音频
重新准备特征；输出目录已存在要使用新目录。进程被中断可能留下 `status: running`，
应按未完成处理。

`inputs.py` 负责清单／NPY 校验，`backends.py` 负责模型执行适配，`main.py` 负责
编排与报告。运行时的 `pipeline.py`、`cif.py`、`decoding.py` 提供推理数学逻辑，
共享 `text_metrics.py` 计算 CER，文件和指标工具不混入模型推理代码。
