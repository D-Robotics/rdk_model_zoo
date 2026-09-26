[English](README.md) | 简体中文

# PP-LiteSeg 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | 文件 | 格式／用途 |
| --- | --- | --- |
| x5 | `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` | BIN，STDC1 Cityscapes 推理 |

精确 asset-id：`x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`。来源：[X5 模型清单](../../../../platforms/x5/docs/release/models.yaml)。没有已发布的 S 系列制品或其他变体。

<a id="preparation"></a>
## 准备步骤

```bash
# cwd: repository root; network required only for preparation
bash samples/vision/pp_liteseg/model/download.sh --target x5
```

下载失败时可从[发布 URL](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin) 获取相同文件，放到下述位置。不要通过改名把其他模型冒充为本制品。

<a id="accompanying-files"></a>
## 伴随文件

无需外部标签文件：visualization.py 内置 Cityscapes 19 类名称和调色板。../test_data 中的两张图片用于示例，不构成精度评估数据集。

<a id="local-paths"></a>
## 本地路径

运行时默认路径为 `samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`，按示例位置解析，不依赖 cwd。下载器的 `--output-dir /data/models` 只修改下载位置。使用该文件时须同时传 `--model-path /data/models/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` 和 `--asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`；自定义相对路径相对于当前工作目录。

<a id="formats-checksums"></a>
## 格式与校验值

`sha256: null (unknown)`。发布清单没有可信摘要；下载时打印的实测摘要可追踪本地副本，但不能独立证明发布者身份。运行时校验文件存在与张量元数据；精确 asset-id 声明请求的契约，不代表自定义文件的来源已经认证。该 BIN 用于 X5，不能与 Nash HBM 互换。
