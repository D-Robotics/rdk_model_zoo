[English](./README.md) | 简体中文

# 运行时(MobileSAM)

用 `hbm_runtime` 在板端完成 MobileSAM 整图掩码推理。

## 目录结构

```text
.
├── main.py
├── mobile_sam.py
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

`run.sh` 自动探测板卡,缺 `.hbm` 时下载对应模型对,再执行 `python3 main.py`。

## 手动运行

```bash
python3 main.py --box 100,50,400,460 --bpu-cores 0 1
```

`main.py` 执行 NCHW RGB 预处理、编码器与解码器推理、掩码上采样,保存叠加图与二值掩码。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--encoder-model-path` | 自动 | 覆盖编码器 `.hbm` 路径。 |
| `--decoder-model-path` | 自动 | 覆盖解码器 `.hbm` 路径。 |
| `--test-img` | `test_data/dogs.jpg` | 输入图像。 |
| `--img-save-path` | `test_data/mobile_sam_full_mask_result.jpg` | 叠加图输出。 |
| `--mask-save-path` | `test_data/mobile_sam_binary_mask_result.png` | 二值掩码输出(参考 `mobile_sam_binary_mask.png` 保留不动)。 |
| `--box` | `185,120,380,445` | 框提示 `x1,y1,x2,y2`(512×512 坐标)。 |
| `--priority` | `0` | 模型调度优先级。 |
| `--bpu-cores` | `0` | BPU 核索引。 |

## 输出

- `mobile_sam_full_mask_result.jpg` — 掩码+轮廓叠加在缩放后图像上。
- `mobile_sam_binary_mask_result.png` — 二值掩码(已提交的 `mobile_sam_binary_mask.png` 为参考,保持不动)。

## 文件

- `main.py` — 入口;解析板卡,加载两个 `.hbm`,跑推理。
- `mobile_sam.py` — `MobileSAMSegment` 流水线(预处理→编码器→解码器→后处理),基于 `hbm_runtime.HB_HBMRuntime`。
- `run.sh` — 启动器,含自动板卡探测与缺模型下载。

## 接口说明

`MobileSAMSegment` 提供标准接口:

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