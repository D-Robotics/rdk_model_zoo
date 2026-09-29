[English](README.md) | [简体中文](README_cn.md)

# 模型准备

<a id="artifacts"></a>
## 已发布制品

| 目标 | 精确 asset ID | 本目录下路径 | 发布者 SHA-256 |
| --- | --- | --- | --- |
| S100 | `s:depth_anything_v2:s100/depth_any.hbm` | `s100/depth_any.hbm` | 未知（`null`） |

已发布文件名和 URL 以 [S 清单](../../../../platforms/s/docs/release/models.yaml)为准。
源文字还提到 S100P，但清单没有独立 S100P 制品或兼容证据。S100P、S600、X5 均
显式拒绝，传入外部路径也不例外。`auto` 选择唯一 S100 契约，真实执行仍在加载
SDK 前检查本机身份。

<a id="preparation"></a>
## 显式准备

从仓库根目录执行：

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
bash samples/vision/depth_anything_v2/model/download.sh --target s100
```

第一条仅读清单，第二条下载所选制品。`download_model.sh` 转发相同参数，历史位置
SoC 参数改为显式 `--target`。推理不下载模型、不安装依赖。本轮主机迁移未下载模型。

<a id="accompanying-files"></a>
## 配套文件

本模型不需要分类标签。内置 `../test_data/furseal.jpg` 与六张图按字节保留自源目录，
不是校准数据或真值。HBM、源权重、ONNX 不随仓库附带。复现所缺前提见
[转换说明](../conversion/README_cn.md)。不声称上游同名 checkpoint 与本制品一致。

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

显式模型路径必须配精确 asset ID。该引用选择契约，不证明任意字节就是已发布 HBM。
运行时记录实测摘要并检查 IO 元数据。自制模型若归一化、几何或语义不同，需要新
绑定；改文件名不能使其兼容。

<a id="formats-checksums"></a>
## 格式、摘要与来源

HBM 是源异构模型格式。预期公开 IO 为 float32 RGB NCHW `[1,3,518,686]` 和
float32 深度 `[1,518,686]`。源图内 int16 量化不证明公开输出为整数。本轮未观察
真实 HBM 元数据；绑定失败应调查原因，不能把不兼容张量 reshape 后绕过检查。

发布者 SHA-256 未知。下载器/运行时计算本地摘要来绑定后续证据，不能独立认证
发布者身份。非空文件、形状相同或成功下载不证明精度或兼容性。后续板测需保留
URL、实测摘要、运行时版本；当前板端验证仍为 `not-run`。
