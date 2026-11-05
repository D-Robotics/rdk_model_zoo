[English](README.md) | 简体中文

# YOLOv5 模型制品

<a id="artifacts"></a>
## 制品清单

历史清单包含 9 个 X5 制品和 2 个 S 制品；本次新增制品见下方已转换表。下表的 `{size}` 只是表示 `s`、`m`、`l`、`x` 四种大小的文档简写，不能原样作为文件名。

| 目标板卡 | variant | filename | format | URL | SHA-256 |
|---|---|---|---|---|---|
| X5 | n-v7.0 | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v2.0 | `yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v2.0 | `yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v2.0 | `yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v2.0 | `yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v2.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | s-v7.0 | `yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5s_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | m-v7.0 | `yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5m_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | l-v7.0 | `yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5l_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| X5 | x-v7.0 | `yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | bin | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin` | null (unknown) |
| S100 | x-672 | `s100/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |
| S600 | x-672 | `s600/yolov5x_672x672_nv12.hbm` | hbm | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm` | null (unknown) |

S100P 没有 YOLOv5 制品。上表历史制品的发布校验值仍为 `sha256: null (unknown)`；新增转换制品有精确摘要。

新增转换规格 `s-v2.0-native-20261102`、`n-v7.0-native-20261102` 使用独立文件名与精确清单身份。X5 默认 `n-v7.0`、S 默认 `x-672` 均不变。新模型为 640×640，几何为最近邻直接拉伸，三个 NHWC float32 raw head；S 的新 F32 输出不执行旧制品的反量化。新 S 规格仅绑定 Python，历史 C++ S32/672 实现不适用。板端性能、精度和执行状态继续待实测。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── download_model.sh  # Shell 脚本
```

<a id="preparation"></a>
## 准备步骤

在允许显式联网的环境中从仓库根目录运行。使用下面显式 `--variant`/`--output-dir` 的写法，让制品落在已知路径：

```bash
python3 -m samples.vision.yolov5.model.download \
  --target x5 --variant n-v7.0 \
  --output-dir samples/vision/yolov5/model
```

S100/S600 使用 `--target s100 --variant x-672` 或 `--target s600 --variant x-672`。脚本会创建目标子目录，并打印观测 digest 和发布值（`unknown`），只下载所选清单行。`run.sh` 和运行时不会调用它。

<a id="accompanying-files"></a>
## 伴随文件

- `../test_data/coco_classes.names`：仅用于可视化的 80 行 COCO 标签。
- `../test_data/bus.jpg`：X5 默认图；`../test_data/kite.jpg`：S 默认图。
- `conversion/` YAML：X5 源转换配置参考，不是模型二进制。

<a id="local-paths"></a>
## 本地路径

执行上述命令后，X5 文件位于 `samples/vision/yolov5/model/`；S 文件位于 `samples/vision/yolov5/model/s100/` 或 `model/s600/`。自定义 `--model-path` 必须和精确匹配的 `--asset-id` 一起使用，不能仅凭文件名猜目标板卡。

<a id="formats-checksums"></a>
## 格式与校验值

历史 X5 制品为 640×640、Bayes-e、紧凑 NV12 的 `.bin`；历史 S100/S600 制品为 672×672 拆分 NV12 `.hbm`。上表历史制品的发布 SHA-256 为 `null`；新增 640×640 转换制品见已转换表，并登记精确摘要。

<a id="license"></a>
## 许可

仓库辅助脚本遵循 Apache-2.0；模型来源和上游权重许可仍以 manifest/源发布为准。

<!-- completed-s-converted-20261102 -->
## 已完成转换的 S 制品

| 目标 | 规格 | 文件 | 板端 |
|---|---|---|---|
| S100 | `n-v7.0-native-20261102` | `s100/yolov5n_tag_v7.0_native_640x640_nashe_nv12_20261102.hbm` | not-run |
| S600 | `n-v7.0-native-20261102` | `s600/yolov5n_tag_v7.0_native_640x640_nashp_nv12_20261102.hbm` | not-run |
| S100 | `s-v2.0-native-20261102` | `s100/yolov5s_tag_v2.0_native_640x640_nashe_nv12_20261102.hbm` | not-run |
| S600 | `s-v2.0-native-20261102` | `s600/yolov5s_tag_v2.0_native_640x640_nashp_nv12_20261102.hbm` | not-run |

```bash
python3 -m samples.vision.yolov5.model.download --target s100 --variant n-v7.0-native-20261102
python3 -m samples.vision.yolov5.runtime.python.main --target s100 --variant n-v7.0-native-20261102 --dry-run
```

这些 640×640 规格使用拆分 UINT8 Y/UV 输入和三个 NHWC float32 raw head。已核对 Python 绑定和主机对照；板端推理、精度和性能均为 not-run。历史 672×672 C++ 绑定不适用。精确 URL、SHA-256 和模型许可见 `docs/release/s/models.yaml` 及公开 release 清单。
<!-- /completed-s-converted-20261102 -->

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100p | `n-v7.0-native-20261102-s100p-o1-20261103` | model | 200 |
| s100p | `s-v2.0-native-20261102-s100p-o1-20261103` | model | 200 |

```bash
python3 samples/vision/yolov5/model/download.py --target s100p --asset-id s:yolov5:s100p/yolov5n_tag_v7.0_native_640x640_nashm_nv12_20261103_o1.hbm
python3 samples/vision/yolov5/runtime/python/main.py --target s100p --variant n-v7.0-native-20261102-s100p-o1-20261103 --asset-id s:yolov5:s100p/yolov5n_tag_v7.0_native_640x640_nashm_nv12_20261103_o1.hbm --model-path /path/to/yolov5n_tag_v7.0_native_640x640_nashm_nv12_20261103_o1.hbm --test-img samples/vision/yolov5/test_data/kite.jpg
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。
