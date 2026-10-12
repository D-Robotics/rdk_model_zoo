[English](README.md) | 简体中文

# PaddleOCR 模型准备

<a id="artifacts"></a>
## 板卡与模型

检测器和识别器须配套使用。X5、S100 模型可通过命令下载；S600 C++ 推理使用板卡镜像或 BSP 模型包提供的匹配模型。

| 板卡 | 检测器 | 识别器 | 准备方式 |
| --- | --- | --- | --- |
| X5 | `en_PP-OCRv3_det_640x640_nv12.bin` | `en_PP-OCRv3_rec_48x320_rgb.bin` | Python `--prepare`，PP-OCRv3 英文模型 |
| S100 | `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | Python `--prepare`，PP-OCRv6 |
| S600 | `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | S600 BSP 模型，位于 `/opt/hobot/model/s600/basic/`；使用 C++ |

X5/S100 下载地址和校验值来自 `docs/release/x5/models.yaml` 和 `docs/release/s/models.yaml`。S600 模型从 S600 板卡镜像或配套模型包获取，选择 S600 编译版本。S100P 未提供可用模型对。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md
└── README_cn.md
```

<a id="preparation"></a>
## 准备步骤

在仓库根目录执行对应板卡的命令。主机安装 Python 和 PyYAML 后即可准备模型；命令会打印保存路径和文件 SHA-256。

X5：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir samples/vision/paddle_ocr/model
```

S100：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target s100 \
  --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
  --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
  --model-dir samples/vision/paddle_ocr/model/s100
```

S600：在 S600 板卡上确认配套模型文件：

```bash
test -s /opt/hobot/model/s600/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm
test -s /opt/hobot/model/s600/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm
```

文件缺失时，从 S600 镜像或 BSP 模型包获取匹配的模型对并放入上述路径。存放到其他目录时，将 `--det-model-path` 和 `--rec-model-path` 传给 [C++ 入口](../runtime/cpp/README_cn.md#run)。Python 准备命令接受 X5 和 S100。

<a id="accompanying-files"></a>
## 配套文件

| 板卡 | 词典 | 位置 |
| --- | --- | --- |
| X5 | 96 字符英文词表 | `runtime/python/cli.py` 中的 `X5_ALPHABET` |
| S100 / S600 | PP-OCRv6 UTF-8 词典 | [`ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt)，18,708 行 |

解码时加入 blank 类别，PP-OCRv6 另加末尾空格。Python `--vocabulary-path` 要求词典摘要与下表一致。C++ 渲染使用 `../test_data/FangSong.ttf`；通过 `--font-path` 选择其他字体。

<a id="local-paths"></a>
## 模型路径

- X5 Python 默认读取本示例下的 `model/<BIN 文件名>`。
- S100 Python 默认读取 `model/s100/<HBM 文件名>`；上述下载命令保存到这些位置。
- C++ 在 S100 或 S600 上默认读取 `/opt/hobot/model/<板卡>/basic/<HBM 文件名>`。
- Python 使用自定义目录时，传入两个模型路径及对应资产 ID。准备命令的 `--model-dir` 设置保存目录，并按文件基本名保存模型对。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `en_PP-OCRv3_det_640x640_nv12.bin` | null | 清单行未记录发布方摘要 |
| `en_PP-OCRv3_rec_48x320_rgb.bin` | null | 未记录发布方摘要 |
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | null | 未记录发布方摘要 |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | null | 未记录发布方摘要 |
| `test_data/s100/ppocrv6_dict.txt` | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` | 已记录；`--vocabulary-path` 替换时强制校验 |

发布方未提供模型校验值；准备命令会打印下载文件的 SHA-256。
