# KWS model artifacts

[English](README.md) | 简体中文

<a id="artifacts"></a>
## 已发布制品

| 身份 | 目标 | 本地路径 | 发布方 SHA-256 |
| --- | --- | --- | --- |
| `s:kws:s100/kws.hbm` | S100 | 本样例下 `model/s100/kws.hbm` | 未记录 |

下载 URL 由[当前清单](../../../../docs/release/s/models.yaml)提供。S100P/S600/X5 无 KWS 发布制品，改名或强行更改板卡别名不能创建适配。

<a id="preparation"></a>
## 显式准备

在仓库根目录单独下载，推理不负责下载：

```sh
bash samples/speech/kws/model/download.sh --target s100
```

`--asset-id` 默认且仅接受上表身份；`--output-dir /path/to/models` 会将文件保存到该目录下的 `s100/kws.hbm`。`PYTHON` 指定解释器。共享下载器检查文件及已配置摘要；本记录没有发布方校验值，打印的观察 SHA 只能绑定字节，不能认证来源。下载不要求板卡。失败返回 2，不能将未完成下载当作可用模型。

<a id="accompanying-files"></a>
## 配套数据

固定 80-bin PaddleAudio 特征协议及“hey snips”语义属于该制品，不提供可编辑文本词表或第二个模型。原始 [sample.wav](../test_data/sample.wav) 仅作演示，不是校准集或验证集，详见[输入身份](../test_data/README_cn.md)。

<a id="local-paths"></a>
## 已有文件

默认模型位置相对 sample 解析，不受调用目录影响。已有文件需向[运行入口](../runtime/python/README_cn.md)同时传 `--model-path` 和 `--asset-id s:kws:s100/kws.hbm`，这是声明预期发布身份；没有校验值的路径本身不能证明来源。真实执行先核验本机身份再加载 SDK，加载后检查严格单模型、单输入、单输出。

<a id="formats-checksums"></a>
## 格式及运行契约

HBM 是 S100 已编译运行文件，不是 ONNX 或 Paddle 权重。输入为有限 float32 `[1,373,80]`，来自 16 kHz 下 60000 个单声道采样点。输出要求 batch=1、静态正维度且值有限；输出形状读取实际 SDK metadata，不从历史控制台分数猜测。Float32 概率直接使用；整数输出必须有有效 SCALE 描述符，经共享反量化后取最大值。超出 [0,1] 的结果直接拒绝，不再次 sigmoid 或静默裁剪。

本轮没有新的板端 metadata 或模型执行证据。[转换说明](../conversion/README_cn.md)记录可复现导出路径的缺失情况。后续板测应同时保留下载文件摘要。
