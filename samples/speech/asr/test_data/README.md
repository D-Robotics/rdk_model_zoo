# ASR bundled data

English | [简体中文](README_cn.md)

## Recording
`chi_sound.wav` is copied unchanged from the S sample: mono PCM16, 16000 Hz, 73440 frames (4.59 seconds). The runtime processes all three windows: 30000, 30000 and 13440 valid target samples; the last is padded to 30000. Source prose describing only the first three seconds does not describe this loop.

SHA-256: `55faf2ac6f06f4355337f4466c9589fb7dd8e38b22eb80c24826081aa79cbcd8`.

## Vocabulary
`vocab.json` maps 3503 unique token strings to contiguous IDs 0–3502. ID 0 is `<pad>` (blank); other special tokens and `|` remain literal decoded text. The loader verifies the exact source SHA-256 `33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7`. An arbitrary replacement vocabulary is not compatible merely because it has the same length.

## Historical figures and limits
`readme_img/acc.jpg`, `perf.jpg`, and `print.jpg` are unchanged source illustrations. They are historical results, not measurements of the migrated code. There is no verified labeled dataset in this directory, so this recording cannot substantiate corpus CER or a new model-accuracy claim. Source licensing is retained; no additional dataset license is inferred.
