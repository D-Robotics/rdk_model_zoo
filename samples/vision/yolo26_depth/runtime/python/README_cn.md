# 运行时(YOLO26 Depth,RDK-S)

用 `hbm_runtime` 在板端完成 YOLO26 Depth 单目深度推理。仅 Python(X5 sample 另带 C++ runtime;本 RDK-S 适配只保留 Python 流水线,足以支撑 demo 与评估)。

## 运行

```bash
bash run.sh        # 默认规格 `n`
bash run.sh m       # 规格 `m`
```

`run.sh` 自动探测板卡,缺 `.hbm` 时下载,再执行 `python3 main.py --variant <v> --input ../../test_data/bus.jpg --output ./output`。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 自动 | 覆盖 `.hbm` 路径。 |
| `--variant` | `n` | 规格 `n`/`s`/`m`/`l`/`x`。 |
| `--input` | (必填) | 输入图像。 |
| `--output` | (必填) | 输出目录。 |
| `--warmup` | `3` | 计时前的预热推理次数。 |
| `--priority` | `0` | 模型调度优先级。 |
| `--bpu-cores` | `0` | BPU 核索引。 |

## 输出

- `depth.png` — Turbo 伪彩色深度。
- `overlay.png` — 深度与输入图叠加。
- `raw_logit.npy`(仅 lite)/ `log_depth.npy` / `depth_native.npy` — 原始及解码后张量。
- `report.json` — 模型/输入 SHA-256、形状、延迟和后处理常量。

## 文件

- `main.py` — 入口;解析板卡+规格,加载 `.hbm`,跑推理。
- `yolo26_depth.py` — `Yolo26Depth`,一个类内含两个按规格自动选择的档位:
  `n`/`s`/`m` 走 NV12(letterbox→NV12→图内解码),`l`/`x` 走 lite
  (scale-fill→RGB `/255` float32 NCHW featuremap→`HB_HBMRuntime`→外置解码)。两档输出一致。
- `run.sh` — 启动器,含自动板卡探测与缺模型下载。
