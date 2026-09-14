[English](./README.md) | 简体中文

# 模型下载方式

运行 `download_model.sh` 下载预编译 HBM 模型。脚本会读取
`/sys/class/boardinfo/soc_name`（也可通过第一参数指定 SoC），下载到
`/opt/hobot/model/<soc>/basic/`。这与 runtime 示例脚本和 `main.py` / `main.cpp`
的默认 `--model-path` 保持一致，所有入口都从同一位置加载模型。

| SOC 解析结果 | 模型来源 |
|---|---|
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/MobileNet/mobilenetv2_224x224_nv12.hbm` |
| 其它（`s100` / `s100p` / `(null)` / 未知） | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm` |

```bash
./download_model.sh           # 自动按当前 SOC 拉取
./download_model.sh s100      # 强制下载 S100 版
./download_model.sh s600      # 强制下载 S600 版
```

下载后的文件位于：

```text
/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm
```

若文件已存在，脚本会直接跳过下载。