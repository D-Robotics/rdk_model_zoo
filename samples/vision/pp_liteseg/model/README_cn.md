[English](README.md) | 简体中文

# PP-LiteSeg 模型准备

<a id="artifacts"></a>
## 制品清单

| Target | 文件 | 格式／用途 |
| --- | --- | --- |
| x5 | `pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` | BIN，STDC1 Cityscapes 推理 |

精确 asset-id：`x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`。来源：X5 模型清单。没有已发布的 S 系列制品或其他变体。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
└── download.sh  # 模型准备命令
```

<a id="preparation"></a>
## 准备步骤

```bash
# cwd: repository root; network required only for preparation
bash samples/vision/pp_liteseg/model/download.sh --target x5
```

下载失败时可从[发布 URL](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin) 获取相同文件，放到下述位置。使用与运行时输入输出协议匹配的模型。

<a id="accompanying-files"></a>
## 伴随文件

cli.py 内置 Cityscapes 19 类名称和调色板，无需外部标签文件。../test_data 中的两张图片用于运行示例，精度评估请准备 Cityscapes 标注数据。

<a id="local-paths"></a>
## 本地路径

运行时默认路径为 `samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`，按示例位置解析，不依赖 cwd。下载器的 `--output-dir /data/models` 只修改下载位置。使用该文件时须同时传 `--model-path /data/models/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` 和 `--asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin`；自定义相对路径相对于当前工作目录。

<a id="formats-checksums"></a>
## 格式与校验值

`sha256: null (unknown)`。发布清单暂无发布摘要；下载时打印的实测摘要用于跨主机追踪本地副本，发布者认证以发布摘要为准。运行时校验文件存在与张量元数据；精确 asset-id 声明请求的契约，自定义文件需自行记录来源。该 BIN 用于 X5，不能与 Nash HBM 互换。

## S100 / S100P / S600 logits 版本

新增 `stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104` 使用 PaddleSeg 官方训练的 STDC1 Cityscapes 权重。S 系列输入为归一化 RGB float32 NCHW `(1,3,512,1024)`，处理为 RGB/255、均值 .5、标准差 .5。HBM 输出 19 类浮点 logits，Python 运行时执行一次 CPU argmax。须选择对应转换版本或精确资产 ID。现有 X5 NV12 输入、int32 类别图输出的默认行为保持不变。

```bash
python3 samples/vision/pp_liteseg/model/download.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104
python3 samples/vision/pp_liteseg/runtime/python/main.py --target s100 --variant stdc1-cityscapes-1024x512-logits-int16-cpuptq-o1-20261104 --test-img /path/to/street.png
```

指定外部 `--model-path` 前，先通过 `--list-models` 获取精确资产 ID。S100P、S600 使用各自 target 和 HBM。板端功能、Cityscapes 精度、Runtime 1/2 并发、C++ 1/2 流仍未实测；编译器估算和源模型主机校验属于转换证据。详见 `conversion/converted_targets_20261103.json` 与 `conversion/source_export_20261104.json`。
