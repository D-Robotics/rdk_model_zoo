# ASR bundled data

[English](README.md) | 简体中文


## 目录结构

```text
test_data/
├── readme_img/  # readme_img 相关文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── vocab.json  # 结构化数据
```

## 录音
`chi_sound.wav` 为单声道 PCM16、16000 Hz、73440 帧，共 4.59 秒。运行时处理全部三个窗口，有效长度为 30000、30000、13440；末块补零到 30000。

SHA-256：`55faf2ac6f06f4355337f4466c9589fb7dd8e38b22eb80c24826081aa79cbcd8`。

## 词表
`vocab.json` 将 3503 个唯一 token 映射到连续 ID 0–3502。0 是 `<pad>`（blank）；其他特殊 token 与 `|` 在解码中按原文保留。加载器核对文件 SHA-256：`33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7`。使用与模型配套的上述有序词表。

## 参考图片
`readme_img/acc.jpg`、`perf.jpg`、`print.jpg` 展示参考识别与性能结果。测量语料 CER 时，先准备录音及对应转写真值，再按评估指南执行。这些文件按其所附许可使用。
