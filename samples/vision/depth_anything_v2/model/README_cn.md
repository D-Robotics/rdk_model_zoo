[English](README.md) | [简体中文](README_cn.md)

# 模型准备

<a id="artifacts"></a>
## 已发布制品

| 目标 | 精确 asset ID | 本目录下路径 | 发布者 SHA-256 |
| --- | --- | --- | --- |
| S100 | `s:depth_anything_v2:s100/depth_any.hbm` | `s100/depth_any.hbm` | 未知（`null`） |

已发布文件名和 URL 以 S 清单为准。
源文档也提到 S100P，但清单没有独立的 S100P 制品。S100P、S600、X5 均
显式拒绝，传入外部路径也不例外。`auto` 选择唯一 S100 契约，真实执行仍在加载
SDK 前检查本机身份。

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
## 显式准备

从仓库根目录执行：

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
bash samples/vision/depth_anything_v2/model/download.sh --target s100
```

第一条仅读清单，第二条下载所选制品。`download_model.sh` 转发相同参数，使用显式
`--target` 参数（不再接受位置式 SoC 名称）。推理不下载模型、不安装依赖。

<a id="accompanying-files"></a>
## 配套文件

本模型不需要分类标签。内置 `../test_data/furseal.jpg` 与六张图按字节保留自源目录，
不是校准数据或真值。HBM、源权重、ONNX 不随仓库附带。复现所缺前提见
[转换说明](../conversion/README_cn.md)。上游同名 checkpoint 与本制品以各自摘要区分。

<a id="local-paths"></a>
## 路径与外部副本

默认运行模型路径以本目录定位，不依赖当前目录。shell 包装先切到仓库根目录，用户
相对路径由此解析；直接调用 Python 时，用户路径以调用时当前目录为准。

下载器 `--output-dir` 只改变目标目录，保留 `s100/` 子目录，不改运行时默认值。
使用外部副本示例：

```bash
bash samples/vision/depth_anything_v2/model/download.sh --target s100 \
  --output-dir /work/depth-models
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 \
  --asset-id s:depth_anything_v2:s100/depth_any.hbm \
  --model-path /work/depth-models/s100/depth_any.hbm \
  --output /work/depth-results/external-copy
```

显式模型路径必须配精确 asset ID。该引用声明预期契约；字节身份以发布摘要与加载检查为准。
运行时记录实测摘要并检查 IO 元数据。自制模型若归一化、几何或语义不同，需要新
绑定；改文件名不能使其兼容。

<a id="formats-checksums"></a>
## 格式、摘要与来源

HBM 是源异构模型格式。预期公开 IO 为 float32 RGB NCHW `[1,3,518,686]` 和
float32 深度 `[1,518,686]`。图内 int16 量化不改变公开输出的浮点契约。加载时
校验张量元数据；绑定失败应调查原因，不能把不兼容张量 reshape 后绕过检查。

发布者 SHA-256 未知。下载器/运行时计算本地摘要来绑定后续证据，不能独立认证
发布者身份。板测结果应与 URL、实测摘要、运行时版本一并保留。
