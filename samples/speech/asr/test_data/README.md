# ASR bundled data

English | [简体中文](README_cn.md)


## Directory structure

```text
test_data/
├── readme_img/  # Files for readme_img
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── vocab.json  # Structured data
```

## Recording
`chi_sound.wav` contains mono PCM16, 16000 Hz, 73440 frames (4.59 seconds). The runtime processes all three windows: 30000, 30000 and 13440 valid target samples; the last is padded to 30000.

SHA-256: `55faf2ac6f06f4355337f4466c9589fb7dd8e38b22eb80c24826081aa79cbcd8`.

## Vocabulary
`vocab.json` maps 3503 unique token strings to contiguous IDs 0–3502. ID 0 is `<pad>` (blank); other special tokens and `|` remain literal decoded text. The loader verifies the exact SHA-256 `33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7`. Use this ordered vocabulary with its matching model.

## Reference figures
`readme_img/acc.jpg`, `perf.jpg`, and `print.jpg` illustrate reference recognition and performance results. To measure corpus CER, prepare recordings with reference transcripts and follow the evaluator guide. Use these files under their supplied license.
