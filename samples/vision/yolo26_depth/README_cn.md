[English](README.md) | 简体中文

# YOLO26 Depth(RDK-S)

YOLO26 Depth 从单张 RGB 图像估计单目深度。本 sample 部署 YOLO26 Depth 的五个规格(`n`/`s`/`m`/`l`/`x`)适配 RDK-S 系列(S100/S100P/S600)，采用**按规格选型的混合发布档位**，使五个规格在所有板卡上均通过实测验收：

- **`n`/`s`/`m` —— NV12 档位。** ONNX 图内保留校准后的 `clip → scale/bias → exp → resize4x` 后处理，运行时送 letterbox 后的 NV12 图像。板端实测 raw cosine 对 FP32：0.9984–0.9996。
- **`l`/`x` —— lite 档位。** ONNX 边界为 192×192 原始深度 logit，运行时送 scale-fill 的 float32 featuremap，CPU 侧执行 `clip → scale/bias → exp` 与最终 resize。板端实测 raw cosine 对 FP32：0.9997，零饱和像素。

之所以拆分：NV12 编译的 `l`/`x` 输出会被定标上界截断(quant max 钉在校准 max 上)，而 lite 编译的 `n`/`s`/`m` 达不到 0.999 cosine 线。混合组合是**五规格全部通过**的唯一配置。

## 平台兼容

| 板卡 | SoC | march | `n`/`s`/`m`(NV12) | `l`/`x`(lite) |
|---|---|---|---|---|
| S100 | s100 | nash-e | `model/nash-e/…_nv12.hbm` | `model/nash-e/…_lite_….hbm` |
| S100P | s100p | nash-m | `model/nash-m/…_nv12.hbm` | `model/nash-m/…_lite_….hbm` |
| S600 | s600 | nash-p | `model/nash-p/…_nv12.hbm` | `model/nash-p/…_lite_….hbm` |

## 目录结构

```
yolo26_depth/
├── conversion/          # ONNX 导出 + hb_compile 量化
│   ├── ptq_yamls/       # committed YAML:NV12 n/s/m + lite l/x,各 march 一份
│   ├── scripts/         # quantize.py、export.py、prepare_calibration.py、...
│   ├── export.py  extract_sunrgbd_subset.py  prepare_calibration.py
├── evaluator/           # SUNRGBD 数值评估
├── model/               # download_model.sh + 各 march .hbm(下载)
├── runtime/python/      # hbm_runtime 推理:main.py、yolo26_depth.py、run.sh
└── test_data/           # bus.jpg
```

## 快速开始

在板端:

```bash
cd samples/vision/yolo26_depth/runtime/python
bash run.sh            # 默认规格 `n`(NV12 档位)
bash run.sh l          # 规格 `l`(lite 档位)
# -> 生成 output/depth.png、output/overlay.png、output/log_depth.npy、...
```

运行时按规格自动选择输入契约(`n`/`s`/`m` 走 NV12，`l`/`x` 走 featuremap)，两个档位输出一致。

## 转换

见 [`conversion/README_cn.md`](./conversion/README_cn.md)。两种 ONNX 边界出自同一上游权重：导出校准 log-depth ONNX(NV12 档位，`n`/`s`/`m`)或 raw-logit lite ONNX(`l`/`x`)，准备对应校准数据，执行 `hb_compile --config <yaml>`，拷贝 `.hbm` 到 `model/<march>/`。

## 运行时

见 [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)。

## 评估

见 [`evaluator/README_cn.md`](./evaluator/README_cn.md) 的 SUNRGBD 数值评估脚本。

## 验证

重新生成 HBM 时，应分别记录 raw-logit 与后处理深度的验证结果。发布验收线为 bus.jpg 上 raw 域 cosine ≥ 0.999 且零饱和像素，按(规格, march)在板端实测。

## License

本 sample 遵循 RDK Model Zoo 许可。上游 YOLO26 权重保留其原始许可。
