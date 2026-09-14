[English](./README.md) | 简体中文

# Depth Anything V2 Python 运行时

本目录提供 RDK S100/S100P 上 Depth Anything V2 单目深度估计的 Python 推理入口。

## 目录结构

```text
.
├── main.py
├── depth_anything_v2.py
├── run.sh
├── README.md
└── README_cn.md
```

## 依赖

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 torch==2.3.1
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | HBM 模型路径 | `../../model/s100/depth_any.hbm` |
| `--test-img` | 测试图片路径 | `../../test_data/furseal.jpg` |
| `--img-save-path` | 输出深度图保存路径 | `result.jpg` |
| `--priority` | hbm_runtime 调度优先级 | `0` |
| `--bpu-cores` | BPU 核心索引 | `0` |

## 快速运行

```bash
cd samples/vision/depth_anything_v2/runtime/python
bash run.sh
```

## 手动运行

```bash
python3 main.py \
  --model-path ../../model/s100/depth_any.hbm \
  --test-img ../../test_data/furseal.jpg \
  --img-save-path result.jpg
```

`main.py` 会读取输入图片，执行 NCHW RGB 预处理、HBM 推理、深度图插值和归一化，并保存彩色深度图。

## 接口说明

`DepthAnythingV2` wrapper 提供标准接口：

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
