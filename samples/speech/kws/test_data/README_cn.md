[English](README.md) | 简体中文

# KWS test audio



## 目录结构

```text
test_data/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```


`sample.wav` 从固定 S 源 `380e1a2bf42041af54be6f34935e50197cfadff9` 逐字节复制。它是源中的“hey snips”演示录音：单声道 PCM16、16000 Hz、40000 帧、2.5 秒。

SHA-256：`eb39ea9bff0e37e262ee3735eba4111a52bb53a84bb776d28a43d7cea6b88cad`。

固定前端补 20000 个零到 60000 点，再生成 float32 `[1,373,80]`。源 S100 对该片段的记录分数约为 0.985。

替换音频时向[运行命令](../runtime/python/README_cn.md)传 `--audio-file /path/to/mono-16k.wav`。非单声道或非 16 kHz 输入显式拒绝；外部转换应保留为独立输入。超过 3.75 秒会截断并记录数量。使用带标签的正负音频和[评估说明](../evaluator/README_cn.md)计算应用指标。
