# 模型下载说明

本目录提供 ASR HBM 模型的下载脚本，支持 RDK S100 和 RDK S600 平台。

## 下载方式

运行以下脚本自动检测当前平台并下载对应模型：

```bash
./download_model.sh
```

模型将下载到 `/opt/hobot/model/<soc>/basic/asr.hbm`（`<soc>` 为当前平台型号，如 `s100` 或 `s600`）。
