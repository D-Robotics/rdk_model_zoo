# 模型下载方式

直接运行 `download_model.sh` 即可将转换好的 hbm 模型下载到此目录。脚本会读取 `/sys/class/boardinfo/soc_name`，根据当前板卡自动选择对应的预编译版本：

| SOC 解析结果 | 模型来源 |
|---|---|
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv2_224x224_nv12.hbm` |
| 其它（`s100` / `s100p` / `(null)` / 未知） | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm` |

```bash
./download_model.sh           # 自动按当前 SOC 拉取
./download_model.sh s100      # 强制下载 S100 版
./download_model.sh s600      # 强制下载 S600 版
```
