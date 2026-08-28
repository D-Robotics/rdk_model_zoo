# 模型(YOLO26 Depth,RDK-S)

五个规格 × 三个 march 的预量化 `.hbm`,按混合发布档位命名(见样例 README)。二进制**不提交**;由 `download_model.sh` 从模型服务器拉取。

命名按档位拆分:

- `n`/`s`/`m`(NV12 档位):`yolo26{variant}_depth_{suffix}_768x768_nv12.hbm`
- `l`/`x`(lite 档位):`yolo26{variant}_depth_lite_{suffix}_768x768.hbm`

其中 `{variant}` ∈ `n/s/m/l/x`,`{suffix}` ∈ `nashe/nashm/nashp`。

| 板卡 | march | 目录 |
|---|---|---|
| S100 | nash-e | `model/nash-e/` |
| S100P | nash-m | `model/nash-m/` |
| S600 | nash-p | `model/nash-p/` |

## 下载

```bash
bash download_model.sh                 # 自动探测板卡,全 5 规格
bash download_model.sh nash-e          # 单 march,全规格
bash download_model.sh nash-e n        # 单 march,单规格
```
