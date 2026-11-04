[English](README.md) | 简体中文

# YOLOE 模型制品

<a id="artifacts"></a>
## 制品

下表列出 16 个原始发布制品。Python 运行时直接支持 S 发布制品的混合浮点/SCALE 输出；也可按[转换说明](../conversion/README_cn.md)准备本地浮点模型。S100 11m 与 11l 在本仓库没有记录的 benchmark。

| Asset ID | Target | Variant | Output route |
| --- | --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | x5 | 11s | floating |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | x5 | 11m | floating |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | x5 | 11l | floating |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11s | 原生 SCALE outputs (Python) |
| `s:yoloe11_seg:yoloe_11m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11m | 原生 SCALE outputs (Python) |
| `s:yoloe11_seg:yoloe_11l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11l | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26n | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26s | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26m | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26l | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26x | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26n | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26s | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26m | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26l | 原生 SCALE outputs (Python) |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26x | 原生 SCALE outputs (Python) |

## 新转换的 X5 11s（2026-11-02）

独立变体 `11s-native-20261102` 对应 `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12_float_20261102.bin`，大小 12,841,075 字节，SHA-256 为 `352e8b8e4b445a8223c2a76a7fdad5594b3317a6ea04e47cd70bdc2066fdc7df`。保留完整十个 NHWC FLOAT32 输出头和固定 4585 类词表。OE X5 1.2.8 使用 200 张真实 COCO 图片完成校准；浮点导出按原门禁通过三张实图及 seed-0 随机输入。运行输入为与 SDK 校准转换器一致的 full-range packed NV12（float32 系数、uint8 截断、左上角色度采样）；BIN 图描述符为 INT8 NHWC `[1,640,640,3]`，源 ONNX 输入为 FLOAT32 NCHW `[1,3,640,640]`。预处理保持原生线性等比缩放、截断尺寸、127 填充和一次 RGB /255。

下载器及 Python/C++ 启动器显式选择 `--target x5 --variant 11s-native-20261102`。旧 `11s` 默认选项与历史实测数据保持不变。该模型制品遵循 AGPL-3.0；[公开源码及权重身份](../conversion/UPSTREAM_NOTICE.md)保留许可原文。

完整 OE 报告保留十个输出的余弦诊断值（0.798033–0.981209），比较样本数未知。这些值是量化诊断，不是数据集精度。本制品尚未测量数据集精度、板端验证或板端性能。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── vocabulary.py  # Python 脚本
```

<a id="preparation"></a>
## 准备

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
```

下载失败或哈希不符会非零退出；已存在且哈希不符的文件不覆盖。离线复制原制品后使用精确 `--asset-id` 与 `--model-path`。Python 入口可使用 S 原制品，按运行时 SCALE 元数据反量化整数输出头。

```bash
# cwd: repository root; original quantized publication only
bash samples/vision/yoloe/model/download.sh --target s100p --variant 26n
```

<a id="accompanying-files"></a>
## 配套文件

必需词表 [classes.names](../test_data/classes.names)，4585 行顺序与模型类别 ID 对齐；CLI 拒绝哈希变化。示例图 [office_desk.jpg](../test_data/office_desk.jpg) 可以替换，但不能更改词表来获得新类别。

<a id="local-paths"></a>
## 本地路径

默认路径为 `model/<target>/<manifest filename>`，S26 保留 `nash-e/` 或 `nash-m/` 子目录；`--output-dir` 则在指定目录后直接拼接清单 filename。`--model-path=null` 按选择推导。X5/S100 默认 11s，S100P 默认 26n；显式 asset-id 在未指定 variant 时优先。

自行转换浮点 HBM 时必须同时指定 `--model-path` 和 `--local-float-sha256`：摘要用于精确识别本地字节并完成选择，`source_asset_id` 记录该转换的协议来源。加载时核验目标平台和全部十个 NHWC float32 输出角色，目标兼容性由这些加载检查与板端运行共同确认。

ONNX 检查、平台校准与可选编译见[转换准备说明](../conversion/README_cn.md)。生成文件名中的 `_float` 表示目标协议；`compiled_unverified` 表示编译产物还需按验证步骤完成元数据与数值检查。


## X5 模型文件大小


| Model | Download | Size (MB) |
| --- | --- | ---: |
| YOLOE-11s-Seg-PF | [yoloe_11s_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11s_seg_pf_bayese_640x640_nv12.bin) | 13.17 |
| YOLOE-11m-Seg-PF | [yoloe_11m_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11m_seg_pf_bayese_640x640_nv12.bin) | 26.73 |
| YOLOE-11l-Seg-PF | [yoloe_11l_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11l_seg_pf_bayese_640x640_nv12.bin) | 32.83 |

<a id="formats-checksums"></a>
## 格式与校验和

X5 为 BIN，S 为 HBM。已知值取自活跃发布清单；S11 发布哈希未知，相邻制品摘要不互用。

| Asset ID | SHA-256 | Authority |
| --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | `8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | `ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | `a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2` | `docs/release/x5/models.yaml` |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe11_seg:yoloe_11m_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe11_seg:yoloe_11l_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | `bf4a03cb75fcfe67cdf6549ef0dec9541b5ae009cdef05e7879a4af9c75a5439` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | `a27c313dc922b778d3537187c7b55a82f87b826edc66abdef027e15770243ee6` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | `d257d502a5245de823cdb76cc734974a1568e68bff02bece232895db29aceeef` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | `695cd10aba09873bb9cd4f838113882945e979c745514185c7ab6f5cb17690c4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | `119a6b069b32282915948e161f548b7b22136b765b70101c33d22680ac6fef24` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | `58922c669bcccec7a00c11719db6c3b9c4eaa4e0388b58bd120fda914fe9d048` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | `73f838c5d52a73451b8af50002c020d571d8fd75beeb29c54e0a6c956e0c45a5` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | `8621baa458a94d3d1946665609633f1ecdd930efeeebe91d975f822ffd273af4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | `314df1d0d60c839f9ff4248ab0d4ff2adc3cc8e514a8e9cf9acca62527696738` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | `2abf9d30dd487f2d5eb53444b7272a71d7be18265a39bd463cfd5b0cd6c14539` | `docs/release/s/models.yaml` |

## 本轮 S100 转换规格（2026-11-03）

`26n-native-cal32-20261103-o1` · S100 / nash-e · OE 3.7.0 (hb_compile 3.5.3, HMCT 2.6.5, HBDK 4.7.5) · O1.

显式选择本轮独立规格。原默认规格及其历史数据保留。实际 HBM 输出为 float32。

| Role | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| model | `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1.hbm` | `00340e52995a559be99fce15909babc7cad09e66785f8a596ab71f67bdeb61c4` | 167770984 |

```bash
bash samples/vision/yoloe/model/download.sh --target s100 --variant 26n-native-cal32-20261103-o1
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n-native-cal32-20261103-o1 --test-img /path/to/image.jpg
```

校准图片数量：32；输入来自独立的 COCO 训练图片。板端 smoke、精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。源码、checkpoint、ONNX、实际配置、原生配方与完整 OE 的摘要见 [converted.json](../conversion/converted.json)。

<!-- converted-all-20261103 -->

## 本轮 S 系列转换规格（2026-11-03）

这些独立规格已完成转换、完整 OE 和实际 HBM 接口核验。源权重、ONNX、每目标配置、校准数量和摘要见 [converted_targets_20261103.json](../conversion/converted_targets_20261103.json)。

| 目标 | Native variant | 组件 | 校准图片 |
| --- | --- | --- | ---: |
| s100 | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |
| s100p | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |
| s600 | `26n-native-cal32-cpuptq-20261103-o1` | model | 32 |

```bash
python3 samples/vision/yoloe/model/download.py --target s100 --asset-id s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n-native-cal32-cpuptq-20261103-o1 --asset-id s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm --model-path /path/to/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm --test-img samples/vision/yoloe/test_data/office_desk.jpg
```

命令中的路径需要替换为实际文件。每套多组件模型须同时准备全部组件。旧默认与既有指标保留原身份；本轮规格的板端 smoke、数据集精度、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。

| Target | Exact asset ID | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| s100 | `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `6582c867d89472f3b5c6aed29944fb15768fcc07adb376965dd831c617a0ed30` | 167799968 |
| s100p | `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `9358de5bc440818e81cd51fbdf669e68b65dbc32060b95fabc7304e5238ad0f2` | 174226456 |
| s600 | `s:yoloe26_seg:nash-p/yoloe_26n_seg_pf_nashp_640x640_nv12_float_cal32_20261103_o1_cpuptq.hbm` | `bd831c5607a7091c19d63ba4e8ffa0db41d9b818f3a65139550fb19010253efd` | 12553336 |
