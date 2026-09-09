简体中文 | [English](./README.md)

# YOLOE-11 PF 模型

目标平台 RDK X5，640x640 NV12。以下 BIN 已提供下载，服务器文件与本地编译产物一致。

| 模型 | 下载 | 大小（MB） |
| --- | --- | ---: |
| YOLOE-11s-Seg-PF | [yoloe_11s_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11s_seg_pf_bayese_640x640_nv12.bin) | 13.17 |
| YOLOE-11m-Seg-PF | [yoloe_11m_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11m_seg_pf_bayese_640x640_nv12.bin) | 26.73 |
| YOLOE-11l-Seg-PF | [yoloe_11l_seg_pf_bayese_640x640_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe/yoloe_11l_seg_pf_bayese_640x640_nv12.bin) | 32.83 |

## 下载方法

在本目录运行。无参数时默认下载 s，兼容 runtime 的默认启动流程。需要 Bash、curl 和 sha256sum。

```bash
bash download_model.sh
bash download_model.sh m
bash download_model.sh l
bash download_model.sh all
```

文件保存在脚本所在目录，不受执行时工作目录影响。已有文件校验一致时直接复用；校验不一致时拒绝覆盖，请先移走旧文件再重试。下载先写入临时文件，只有 SHA256 校验通过后才生成最终文件。

## SHA256

| 模型 | SHA256 |
| --- | --- |
| 11s-PF | `8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa` |
| 11m-PF | `ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31` |
| 11l-PF | `a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2` |

转换和运行方法见 [转换说明](../conversion/README_cn.md) 与 [运行说明](../runtime/python/README_cn.md)。运行 m/l 时使用 `--model-path` 指定模型；三款共用仓库现有的 4585 类标签文件。
