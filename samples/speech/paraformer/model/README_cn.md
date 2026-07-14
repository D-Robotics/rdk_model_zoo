[English](./README.md)

# 模型下载

Paraformer 已完成 RDK S100 模型编译和验证。执行以下命令将 3 个 HBM 阶段、`tokens.json` 与 WAV 前处理资源部署到标准板端路径：

```bash
bash download_model.sh
```

脚本将以下文件保存到 `/opt/hobot/model/s100/basic/paraformer/`：

- `paraformer_large_encoder_400x560_s100.hbm`
- `paraformer_large_predictor_400x512_s100.hbm`
- `paraformer_large_decoder_400x512_s100.hbm`
- `tokens.json`
- `am.mvn`（FunASR WavFrontend 的 CMVN 统计）
- `paraformer_config.yaml`（前端参数记录）

HBM 与词表从官方 RDK Model Zoo archive 下载；前端资源随样例提供。其他平台请重新编译对应的模型文件。
