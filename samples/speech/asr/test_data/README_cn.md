# ASR bundled data

[English](README.md) | 简体中文

## 录音
`chi_sound.wav` 从 S sample 原样复制：单声道 PCM16、16000 Hz、73440 帧，共 4.59 秒。运行时处理全部三个窗口，有效长度为 30000、30000、13440；末块补零到 30000。源说明中“前三秒”不能准确描述此循环。

SHA-256：`55faf2ac6f06f4355337f4466c9589fb7dd8e38b22eb80c24826081aa79cbcd8`。

## 词表
`vocab.json` 将 3503 个唯一 token 映射到连续 ID 0–3502。0 是 `<pad>`（blank）；其他特殊 token 与 `|` 在解码中原样保留。加载器核对源文件 SHA-256：`33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7`。词表长度相同不代表可随意替换。

## 历史图片与边界
`readme_img/acc.jpg`、`perf.jpg`、`print.jpg` 均为未修改的源分支插图，只代表历史结果，不是迁移后实测。目录没有已验证的标注数据集，不能凭这段录音宣称语料 CER 或新模型精度。保留源许可，不推断额外数据集授权。
