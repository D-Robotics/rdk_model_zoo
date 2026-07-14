# Test Data

This directory contains two validated 16 kHz WAV samples for a direct board-side smoke run:

- `manifest.json`: two utterance IDs and reference text.
- `audio/BAC009S0724W0121.wav`
- `audio/BAC009S0724W0168.wav`

From either runtime directory, run `bash run.sh` without arguments. The script uses FunASR `WavFrontend` to create fbank, LFR, and CMVN features before invoking the RDK Paraformer HBM pipeline. Pass another directory containing `manifest.json` and `audio/<utt_id>.wav` as the first argument to run your own data.
