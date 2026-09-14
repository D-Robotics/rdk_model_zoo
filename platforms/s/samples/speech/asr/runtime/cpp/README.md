# ASR 语音识别示例（C++）

本示例展示如何在 BPU 上使用量化后的 ASR 模型执行自动语音识别（中文），支持流式音频读取、预处理、推理和 CTC 贪心解码。

> 本模型支持 **RDK S100** 和 **RDK S600** 平台。

## 环境依赖

```bash
sudo apt install libgflags-dev libsndfile1-dev libsamplerate0-dev
```

## 目录结构

```text
.
├── inc/
│   ├── asr.hpp                # ASR 模型封装接口与函数声明
│   └── audio_chunk_reader.hpp # 流式音频读取工具类
├── src/
│   ├── asr.cpp                # ASR 推理与前后处理实现
│   ├── audio_chunk_reader.cpp # 流式音频读取实现
│   └── main.cpp               # 推理程序入口（参数解析与流程控制）
├── CMakeLists.txt             # CMake 构建配置
├── run.sh                     # 示例运行脚本（自动安装依赖、下载模型、编译、运行）
└── README.md                  # 使用说明
```

## 编译工程

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译产物为 `build/asr`。

## 参数说明

| 参数              | 说明                                        | 默认值                                      |
|------------------|---------------------------------------------|---------------------------------------------|
| `--model_path`   | 模型文件路径（.hbm 格式）                    | `/opt/hobot/model/<soc>/basic/asr.hbm`      |
| `--test_sound`   | 输入音频文件路径（.wav 或 .flac）            | `../../../test_data/chi_sound.wav`          |
| `--vocab_file`   | 词表 JSON 文件路径（token -> id）            | `../../../test_data/vocab.json`             |
| `--audio_maxlen` | 每次推理的音频采样点数（在 new_rate Hz 下）   | `30000`                                     |
| `--new_rate`     | 目标采样率（Hz）                             | `16000`                                     |

## 快速运行

### 方式一：一键运行（推荐）

```bash
cd runtime/cpp/
./run.sh
```

脚本会自动完成：依赖安装 → 模型下载 → 编译 → 推理执行。

### 方式二：手动运行

- 使用默认参数

    ```bash
    cd build/
    ./asr
    ```

- 指定参数运行

    ```bash
    cd build/
    ./asr \
        --model_path /opt/hobot/model/s100/basic/asr.hbm \
        --test_sound ../../../test_data/chi_sound.wav \
        --vocab_file ../../../test_data/vocab.json
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
- 音频文件需为 `.wav`/`.flac` 格式，采样率不限（会自动重采样到 16000 Hz）。
- 词表文件为 JSON 格式：`{"<token>": <id>, ...}`。
