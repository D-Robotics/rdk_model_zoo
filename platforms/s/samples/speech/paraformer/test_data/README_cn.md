# 测试数据

本目录内置两条 16 kHz WAV 样例，用户可直接在板端运行：

- `manifest.json`：包含两条语音的 ID 和参考文本。
- `audio/BAC009S0724W0121.wav`
- `audio/BAC009S0724W0168.wav`

在任意 Runtime 目录执行 `bash run.sh` 而不传参数时，脚本使用 FunASR `WavFrontend` 将这两条 WAV 转为 fbank、LFR 和 CMVN 特征，再送入 RDK Paraformer HBM。运行自己的数据时，将包含 `manifest.json` 与 `audio/<utt_id>.wav` 的目录作为第一个参数传入即可。
