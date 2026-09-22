# PaddleOCR 模型准备

<a id="artifacts"></a>
## 制品

本目录不保存模型二进制。sample 消费四条已发布清单行——每个模型对两条
——且检测器与识别器在任何情况下都不跨对混用：

| 目标 | 角色 | 完整引用 | 格式 |
| --- | --- | --- | --- |
| X5 | 检测器 | `x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin` | `.bin`，march `bayes-e`，packed NV12 640×640 |
| X5 | 识别器 | `x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin` | `.bin`，march `bayes-e`，RGB featuremap 48×320 |
| S100 | 检测器 | `s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | `.hbm`，march `nash-e`，split NV12 640×640 |
| S100 | 识别器 | `s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | `.hbm`，march `nash-e`，RGB 48×320 |

这些行位于平台发布清单（迁移窗口期内为
`platforms/x5/docs/release/models.yaml` 与
`platforms/s/docs/release/models.yaml`）。S100P 与 S600 没有经审计的
PaddleOCR 行，因此不对这两个目标声明任何制品。

<a id="preparation"></a>
## 准备

准备是显式操作，统一由 Python 入口的 `--prepare` 模式承担——它是
sample 内唯一可联网的操作。该模式读取清单 URL、取回同一对的两个文件并
打印实测 SHA-256（cwd：仓库根目录；成功：退出码 0，`--model-dir` 下
两个文件）：

```bash
python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
  --target x5 \
  --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
  --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
  --model-dir /tmp/rdk-models
```

S100 将两条引用换成 `s:paddle_ocr:s100/...`，`--model-dir` 例如
`/opt/hobot/model/s100/basic`。常规推理不联网：要求两个本地路径已经
存在（见[运行时参数](../runtime/python/README.md#parameters)）。手动
复制先前下载过的制品同样是合法准备；首次使用时将实测摘要与
`--prepare` 打印值核对即可。

<a id="accompanying-files"></a>
## 伴随文件

每个识别器把字符词典作为模型对契约的一部分：

| 目标 | 词典 | 位置 | 说明 |
| --- | --- | --- | --- |
| X5 | 固定 96 字符字母表 | 内嵌于 `runtime/python/model_binding.py`（`X5_ALPHABET`） | 非文件；解码时前置 blank 类 |
| S100 | PP-OCRv6 UTF-8 词典 | [`test_data/s100/ppocrv6_dict.txt`](../test_data/s100/ppocrv6_dict.txt) | 18,708 行；加载时前置 blank、追加末尾空格 → 18,710 类 |

`--vocabulary-path` 替换 S100 词典时，文件 SHA-256 必须命中下方审计
摘要；其他内容一律拒绝。S 系列 C++ 运行时另需 TrueType 字体渲染结果，
字体自审计源交付携带于
`platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf`，可用
C++ `--font_path` 指定。

<a id="local-paths"></a>
## 本地路径

- X5 模型对：`--prepare --model-dir` 写入的位置，例如
  `/tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin`；通过
  `--det-model-path`/`--rec-model-path` 连同两条完整引用一起传入。
- 默认查找：两个路径参数都缺省时，模型对解析到本目录
  `samples/vision/paddle_ocr/model/<filename>`——把制品复制到这里即可
  省去显式路径参数（自定义路径必须检测器+识别器成对提供）。
- S100 模型对：RDK S 镜像在 `/opt/hobot/model/s100/basic/` 自带该
  模型对；完整检出也可以在该目录缺失时 `--prepare` 写入。
- 词典与测试图随仓入库，无需下载。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
| `en_PP-OCRv3_det_640x640_nv12.bin` | null | 清单行未记录发布方摘要 |
| `en_PP-OCRv3_rec_48x320_rgb.bin` | null | 未记录发布方摘要 |
| `PP-OCRv6_det_infer-deploy_640x640_nv12.hbm` | null | 未记录发布方摘要 |
| `PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm` | null | 未记录发布方摘要 |
| `test_data/s100/ppocrv6_dict.txt` | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` | 已审计；`--vocabulary-path` 替换时强制校验 |

未知摘要保持 `null`，不复制、不猜测；板端运行需要溯源时，请把
`--prepare` 打印的实测摘要记入自己的证据。各制品的运行时张量契约
（名称、形状、dtype）在加载时强制校验，文档见
[运行时阶段 I/O](../runtime/python/README.md#stage-io)。
