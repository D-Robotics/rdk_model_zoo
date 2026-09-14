[English](./README.md) | 简体中文

# KWS 关键词检测样例（Python）

本示例展示如何在 RDK 平台上使用 KWS 模型进行关键词检测推理，输入 .wav 格式的音频数据，输出关键词检测的置信度分数。

## 环境依赖

本样例需确保安装了以下依赖：

```bash
pip install numpy==1.26.4 paddlepaddle paddleaudio
```

## 目录结构

```text
.
├── README.md               # 示例说明文档
├── main.py                 # 示例主入口，执行关键词检测推理
├── kws.py                  # KWS 模型封装
└── run.sh                  # 一键运行脚本
```

## 参数说明

| 参数              | 说明                                         | 默认值                                           |
|-------------------|----------------------------------------------|--------------------------------------------------|
| `--model-path`    | 模型文件路径（.hbm 格式）                     | 自动解析                                          |
| `--audio-file`    | 测试音频文件路径（.wav 格式）                  | `../../test_data/sample.wav`                     |
| `--audio-maxlen`  | 音频截断最大采样点数                           | `60000`                                          |
| `--frame-shift`   | fbank 帧移（毫秒）                            | `10`                                             |
| `--frame-length`  | fbank 帧长（毫秒）                            | `25`                                             |
| `--n-mels`        | fbank 梅尔滤波器组数                           | `80`                                             |
| `--priority`      | 模型优先级（0~255，越大优先级越高）            | `0`                                              |
| `--bpu-cores`     | 推理使用的 BPU 核心编号列表                   | `[0]`                                            |

## 快速运行

- 使用脚本自动运行
    ```bash
    ./run.sh
    ```
- 使用默认参数
    ```bash
    python main.py
    ```
- 指定参数运行
    ```bash
    python main.py \
    --model-path /opt/hobot/model/s100/basic/kws.hbm \
    --audio-file ../../test_data/sample.wav
    ```

## 输入格式

模型输入为 fbank 特征张量，由 paddleaudio 提取。`.wav` 音频文件加载后会自动进行截断/补零和 fbank 特征提取。

## 接口说明

本示例代码提供了详细的注释。为了获取最准确、最新的接口定义，请直接查阅源码中的文档字符串：

- **KWSConfig** 与 **KWS**: 详见 `kws.py`
