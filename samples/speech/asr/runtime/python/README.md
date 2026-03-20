# ASR 语音识别示例（Python）

本示例展示如何在 BPU 上使用量化后的 ASR 模型执行自动语音识别（中文），支持流式音频预处理、推理和 CTC 贪心解码。

> 本模型支持 **RDK S100** 和 **RDK S600** 平台。

## 目录结构

```text
.
├── asr.py      # ASR 推理封装类（预处理、推理、后处理）
├── main.py     # Python 推理入口脚本
├── run.sh      # 示例运行脚本（自动安装依赖、下载模型、运行）
└── README.md   # 使用说明
```

## 参数说明

| 参数              | 说明                                        | 默认值                                      |
|------------------|---------------------------------------------|---------------------------------------------|
| `--model-path`   | 模型文件路径（.hbm 格式）                    | `/opt/hobot/model/<soc>/basic/asr.hbm`      |
| `--audio-file`   | 输入音频文件路径（.wav 或 .flac）            | `../../test_data/chi_sound.wav`             |
| `--vocab-file`   | 词表 JSON 文件路径（token -> id）            | `../../test_data/vocab.json`                |
| `--audio-maxlen` | 每次推理的音频采样点数（在 new_rate Hz 下）   | `30000`                                     |
| `--new-rate`     | 目标采样率（Hz）                             | `16000`                                     |
| `--priority`     | 推理优先级（0~255，0 最低，255 最高）         | `0`                                         |
| `--bpu-cores`    | 使用的 BPU 核心列表                          | `[0]`                                       |

## 快速运行

### 方式一：一键运行（推荐）

```bash
cd runtime/python/
./run.sh
```

脚本会自动完成：依赖安装 → 模型下载 → 推理执行。

### 方式二：手动运行

- 使用默认参数

    ```bash
    python3 main.py
    ```

- 指定参数运行

    ```bash
    python3 main.py \
        --model-path /opt/hobot/model/s100/basic/asr.hbm \
        --audio-file ../../test_data/chi_sound.wav \
        --vocab-file ../../test_data/vocab.json
    ```

### 输出结果

运行成功后，识别文字将打印到终端：

```text
Transcription:
今天天气不错
```

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。

## 注意事项

- **平台兼容性**：本模型（`asr.hbm`）支持 RDK S100 和 RDK S600 平台。
- 音频文件需为单声道或多声道的 `.wav`/`.flac` 格式，采样率不限（会自动重采样到 16000 Hz）。
- 词表文件为 JSON 格式：`{"<token>": <id>, ...}`。
