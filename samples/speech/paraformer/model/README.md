[简体中文](./README_cn.md)

# Model Download

Paraformer has been compiled and validated on RDK S100. Download the three HBM stages, `tokens.json`, and WAV frontend resources to the standard board path:

```bash
bash download_model.sh
```

The script stores these files in `/opt/hobot/model/s100/basic/paraformer/`:

- `paraformer_large_encoder_400x560_s100.hbm`
- `paraformer_large_predictor_400x512_s100.hbm`
- `paraformer_large_decoder_400x512_s100.hbm`
- `tokens.json`
- `am.mvn` (CMVN statistics for FunASR WavFrontend)
- `paraformer_config.yaml` (frontend parameter record)

HBM files and the token list come from the official RDK Model Zoo archive; frontend resources are shipped with this sample. Recompile the matching model files for other platforms.
