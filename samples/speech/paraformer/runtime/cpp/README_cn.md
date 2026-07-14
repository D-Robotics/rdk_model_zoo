[English](./README.md)

# C++ Runtime

C++ 实现保留 hbUCP/hbDNN pipeline，包括 BPU 内存布局所需的对齐 tensor 拷贝，以及原有 3 段 HBM 与 CPU CIF。用户仍从 WAV 开始：`run.sh` 调用 `../python/main.py --preprocess-only`，使用 `paraformer.py` 中的 FunASR 前处理生成临时 `[1, 400, 560]` 特征与有效帧长度；C++ 可执行程序只消费该临时数据，不改变 HBM 推理逻辑。

## 运行

```bash
cd runtime/cpp
bash run.sh [test_data_directory]
```

脚本会下载缺失模型、创建隔离的 Python 前端环境、从 WAV 生成 `build/eval/feats/`、使用 CMake 编译，并执行内置 manifest。`N_UTT=10 bash run.sh` 可限制处理数量。临时 `.npy` 特征仅供 C++ bridge 使用，用户无需准备。

## 输入数据

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    └── <utt_id>.wav
```

WAV 必须为 16 kHz；每条 manifest 记录至少含 `utt_id`，可选 `text`。板端需要提供 S100 的 `dnn`、`hbucp` 库以及 `nlohmann/json.hpp` 开发头文件。
