[English](./README.md)

# Paraformer 语音识别

本示例在 **RDK S100** 上部署固定输入形状的 **Paraformer-large-contextual** ASR pipeline。部署使用 3 个 INT16 HBM，并严格保持已验证的执行顺序：

```text
16 kHz WAV -> FunASR WavFrontend（fbank + LFR + CMVN）
  -> [1, 400, 560] 特征 -> Encoder HBM -> Predictor HBM -> CPU CIF -> Decoder HBM -> 贪心 token 解码
```

FunASR 仅用于复现原模型的 WAV 前处理；RDK 端的识别推理仍由 3 段 HBM、CPU CIF 和 Decoder 完成。前端输出遵循原模型的 fbank、LFR 与 CMVN 配置，不改变已验证的 HBM 图输入、CIF 计算、Decoder token 数和解码行为。

## 模型来源

- FunASR 原始仓库：`https://github.com/modelscope/FunASR`
- FunASR Contextual Paraformer 实现：`https://github.com/modelscope/FunASR/tree/main/funasr/models/contextual_paraformer`
- ModelScope 原始模型：`https://www.modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`
- 参考论文：*Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition*

原始 ModelScope 模型为面向 16 kHz 中文语音的 Contextual Paraformer，使用 8,404 个 token 的通用词表。本示例提供在 RDK 上的固定 shape INT16 部署实现，并内置原始 FunASR WAV 前处理；训练、微调、VAD、标点恢复和时间戳等能力请使用上游 FunASR 工具链。

## 算法介绍与模型功能

Paraformer 是非自回归端到端语音识别模型。它并行预测 token 对齐的声学表示，再由双向 Decoder 生成识别文本，避免自回归解码逐 token 串行生成的延迟。部署图分为四个逻辑阶段：

- **Encoder**：将 fbank+LFR 语音特征转换为上下文声学表示。
- **Predictor**：通过 Continuous Integrate-and-Fire（CIF）激活值预测 token 数量，并生成 token 对齐声学 embedding。
- **CPU CIF**：根据 Predictor 输出完成 CIF 累积，构造 Decoder 所需的固定 shape 声学输入。
- **Decoder**：结合 Encoder 上下文、CIF embedding、contextual-bias embedding 和 token 数量，输出 token logits；随后贪心解码为 UTF-8 文本。

该模型可用于 16 kHz 中文语音转写、离线语音控制、会议语音转写、内容检索，以及结合上游 FunASR 工具链进行领域微调后的端侧部署。本 RDK Runtime 直接接收 WAV，内部调用与原模型一致的 FunASR 前处理后送入 Paraformer HBM pipeline。

## 平台说明

本样例已在 **RDK S100（Nash-e）** 上完成模型编译、功能和性能验证。其他平台需要使用对应平台重新编译生成的模型文件。

## 目录结构

```text
paraformer/
├── README.md                                  # English overview and quick-start
├── README_cn.md                               # 中文模型说明与运行入口
├── conversion/
│   ├── README.md / README_cn.md               # 完整量化、转换与部署指南
│   ├── 01_reexport_fixed_shape.py             # FunASR 固定 shape 导出与 CIF patch
│   ├── 02_extract_decoder.py                  # 抽取 Decoder 子图
│   ├── 03_convert_gather_int64_to_int32.py    # Gather 索引类型修复
│   ├── 04_topsort.py                          # ONNX 拓扑排序
│   ├── 05_fold_range.py                       # Range 常量折叠
│   ├── 06_shape_freeze.py                     # 动态 shape / Gather 图修复
│   ├── 07_extract_predictor.py                # 抽取 Predictor 子图
│   ├── 08_extract_encoder.py                  # 抽取 Encoder 子图
│   ├── 09_gen_calib_features.py               # Encoder 真实特征校准数据
│   ├── 10_gen_real_calib.py                   # Predictor / Decoder 真实中间量校准数据
│   ├── 11_eval_pipeline.py                    # FP32 / INT16 pipeline 精度评测
│   ├── cif_numpy.py                            # CPU CIF 参考实现
│   └── configs/
│       ├── encoder_int16.yaml
│       ├── predictor_int16.yaml
│       └── decoder_int16.yaml
├── evaluator/
│   └── README.md / README_cn.md               # 数据格式、CER 和性能验证记录
├── model/
│   ├── am.mvn                                 # FunASR CMVN 统计
│   ├── paraformer_config.yaml                 # FunASR 前端参数
│   ├── download_model.sh                      # 三段 HBM、tokens 与前端资源部署
│   └── README.md / README_cn.md
├── runtime/
│   ├── python/
│   │   ├── main.py                            # Python 命令行入口
│   │   ├── paraformer.py                      # WAV 前处理、HBM Runtime 模型封装与 CPU CIF
│   │   ├── run.sh
│   │   └── README.md / README_cn.md
│   └── cpp/
│       ├── CMakeLists.txt
│       ├── inc/paraformer.hpp
│       ├── src/main.cpp
│       ├── src/paraformer.cpp                 # hbUCP / hbDNN pipeline
│       ├── run.sh
│       └── README.md / README_cn.md
└── test_data/
    ├── manifest.json                          # 两条可直接运行的测试样本清单
    ├── audio/
    │   ├── BAC009S0724W0121.wav
    │   └── BAC009S0724W0168.wav
    └── README.md / README_cn.md               # 测试数据与自定义数据格式
```

## 快速体验

1. 在板端下载模型：

```bash
cd model
bash download_model.sh
```

2. 示例已在 `test_data/` 内置两条 16 kHz WAV 样例，无需额外准备数据即可运行：

```text
test_data/
├── manifest.json
└── audio/
    ├── BAC009S0724W0121.wav
    └── BAC009S0724W0168.wav
```

`manifest.json` 中每条记录包含 `utt_id`，可选 `text` 作为参考转写；Runtime 自动生成固定 `[1, 400, 560]` 特征和未 padding 的有效帧数。

3. 直接执行 Runtime：

```bash
cd runtime/python
bash run.sh

cd ../cpp
bash run.sh
```

如需运行自己的数据，将含有 `manifest.json` 与 `audio/<utt_id>.wav` 的 `<AUDIO_DATA_DIR>` 作为第一个参数：`bash run.sh <AUDIO_DATA_DIR>`。

## 精度保持条件

已验证的结果依赖以下条件：使用相同的 FunASR 前端与 16 kHz 音频处理、固定 `torch.manual_seed(191009)` 的 fbank dither、固定 `[1, 400, 560]` 特征、三段 INT16 HBM、模型 tensor 连线、基于实际帧数的 CIF padding mask、`max_label_len=100`、全零 `bias_embed` 以及特殊 token 过滤。请勿替换为近似的自定义音频前处理或随机特征。

## 模型转换

Model Zoo 已提供完成编译的 S100 HBM 模型，普通用户执行 `model/download_model.sh` 后即可直接运行。如需从 FunASR 原始模型复现固定 shape 导出、ONNX 图手术、真实数据校准和三段 INT16 编译，请阅读 [`conversion/README_cn.md`](./conversion/README_cn.md)。

## 模型推理

- **Python Runtime**：适合原型验证。输入为 WAV，`paraformer.py` 完成前处理、三段 HBM、CPU CIF 和解码；`main.py` 调度 manifest 并输出时延。详见 [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)。
- **C++ Runtime**：适合生产部署。保持 hbUCP/hbDNN 推理实现；脚本内部从 WAV 生成临时特征后执行 C++ pipeline。详见 [`runtime/cpp/README_cn.md`](./runtime/cpp/README_cn.md)。

## 推理结果

内置样例包含两条 AISHELL dev WAV。Python 和 C++ Runtime 均可识别为中文文本；第二条样例在已验证链路中存在“搅 / 绞”同音字替换，这是两条样例 C++ CER 为 `3.57%` 的原因。运行时会输出各阶段时延，Python 额外单列 WAV 前处理时间。

词表中的英文 BPE 子词使用 `@@` 表示续接关系；Python 与 C++ Runtime 在最终输出阶段会自动移除该标记并拼接英文单词，终端识别结果不会显示 `@@`。

## 模型评估

[`evaluator/README_cn.md`](./evaluator/README_cn.md) 说明评测数据格式、FP32/INT16 主机 CER 评测、板端 WAV 评测命令、输出文件和已完成的 300 条 AISHELL dev 验证结果。


## 模型量化与部署

完整的固定 shape 导出、ONNX 图手术、真实 WAV 校准、三段 INT16 HBM 编译、CER 评测和板端部署流程见 [`conversion/README_cn.md`](./conversion/README_cn.md)。

量化脚本、编译配置与 CPU CIF 参考实现均在 [`conversion/`](./conversion/) 目录中；生成的 ONNX、校准数据、日志与 HBM 为本地构建产物，不应提交到仓库。
## 许可证

本示例遵循仓库顶层 `LICENSE`。
