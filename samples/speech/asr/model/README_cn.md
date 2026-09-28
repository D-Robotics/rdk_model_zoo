# ASR model preparation

[English](README.md) | 简体中文

<a id="artifacts"></a>
## 制品

| 目标 | 精确身份 | 相对 sample 的本地路径 |
| --- | --- | --- |
| S100 | `s:asr:s100/asr.hbm` | `model/s100/asr.hbm` |
| S600 | `s:asr:s600/asr.hbm` | `model/s600/asr.hbm` |

URL 来自[当前 S 清单](../../../../docs/release/s/models.yaml)，两项均未记录发布方 SHA-256。X5/S100P 没有制品；观察摘要仅绑定本地字节，不认证发布来源。

<a id="preparation"></a>
## 显式下载

在仓库根目录选择真实目标：

```sh
bash samples/speech/asr/model/download.sh --target s100
bash samples/speech/asr/model/download.sh --target s600
```

通常只运行所需目标对应的一条；明确准备两者时才同时下载。下载器要求 `--target`，可选 `--asset-id` 必须与之匹配；`--output-dir /path/to/models` 会在该目录保存 `<target>/asr.hbm`。不执行推理或安装 SDK，`PYTHON` 指定解释器，失败返回 2。

<a id="accompanying-files"></a>
## 词表与音频

`test_data/vocab.json` 含 3503 个 token→ID，ID 连续为 0..3502，blank `<pad>` 为 0。运行时核验固定源摘要再构建有序词表；仅输出宽度一致不能保证词序正确，任意换词表可能静默改变文本，因此拒绝替换。见[输入摘要](../test_data/README_cn.md)。随附 WAV 是演示数据，不是校准或有标签准确率语料。

<a id="local-paths"></a>
## 已有模型

在 S600 使用已准备文件：

```sh
bash samples/speech/asr/runtime/python/run.sh --target s600 \
  --asset-id s:asr:s600/asr.hbm --model-path /path/to/asr.hbm \
  --output-dir outputs/asr-s600-run1
```

覆盖路径必须提供精确 ID；默认路径相对 sample，用户相对路径相对调用目录。构造 SDK 前核验实际板卡，清单选择本身不证明硬件或制品兼容性。

<a id="formats-checksums"></a>
## 运行契约

要求严格单个具名模型、一个 float32 `[1,30000]` 输入和一个 `[1,T,3503]` 输出，T 为正。张量名称和 T 读取实际 SDK metadata；float32 logits 保持原值，整数 logits 必须有有效 SCALE 描述符，经共享反量化后 argmax。NaN/Inf 或非法张量拒绝。无需增加不改变有限 float argmax 的 softmax。

本轮没有板端新采集的模型描述符或推理结果。HBM 不是 ONNX 或训练权重；[转换说明](../conversion/README_cn.md)列出缺失前提，不虚构导出命令。
