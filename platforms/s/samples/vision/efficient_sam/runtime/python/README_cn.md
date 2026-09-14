[English](./README.md) | 简体中文

# 运行时(EfficientSAM)

用 `hbm_runtime` 在板端完成 EfficientSAM 整图掩码推理。

## 目录结构

```text
.
├── main.py
├── efficient_sam.py
├── run.sh
├── README.md
└── README_cn.md
```

## 依赖

```bash
pip install numpy opencv-python
```

`hbm_runtime` 由板端 RDK-S 运行时提供,无需 pip 安装。

## 快速运行

```bash
bash run.sh
```

`run.sh` 自动探测板卡,缺 `.hbm` 时下载对应模型对,再执行 `python3 main.py`。框提示已烤进解码器 ONNX,无需传提示参数。

## 手动运行

```bash
python3 main.py --bpu-cores 0 1
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--encoder-model-path` | 自动 | 覆盖编码器 `.hbm` 路径。 |
| `--decoder-model-path` | 自动 | 覆盖解码器 `.hbm` 路径。 |
| `--test-img` | `test_data/dogs.jpg` | 输入图像。 |
| `--img-save-path` | `test_data/efficient_sam_full_mask_result.jpg` | 叠加图输出。 |
| `--mask-save-path` | `test_data/efficient_sam_binary_mask_result.png` | 二值掩码输出(参考 `efficient_sam_binary_mask.png` 保留不动)。 |
| `--priority` | `0` | 模型调度优先级。 |
| `--bpu-cores` | `0` | BPU 核索引。 |

## 输出

- `efficient_sam_full_mask_result.jpg` — 掩码+轮廓叠加图。
- `efficient_sam_binary_mask_result.png` — 二值掩码(已提交的 `efficient_sam_binary_mask.png` 为参考,保持不动)。

## 文件

- `main.py` — 入口;解析板卡,加载两个 `.hbm`,跑推理。
- `efficient_sam.py` — `EfficientSAMSegment` 流水线,基于 `hbm_runtime.HB_HBMRuntime`。
- `run.sh` — 启动器,含自动板卡探测与缺模型下载。

## 接口说明

`EfficientSAMSegment` 提供标准接口:

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

本目录遵循 [Apache 2.0 License](../../../../../LICENSE)。