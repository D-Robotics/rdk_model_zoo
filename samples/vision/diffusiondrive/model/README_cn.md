[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 模型资产

<a id="artifacts"></a>
## 已发布资产

| 目标 | March | 资产 ID / 本地后缀 |
| --- | --- | --- |
| S100P | `nash-m` | `s:diffusiondrive:s100p/diffusiondrive_r34_256x1024_s100p.hbm` |
| S600 | `nash-p` | `s:diffusiondrive:s600/diffusiondrive_r34_256x1024_s600.hbm` |

`s:diffusiondrive:` 后的部分为相对本目录的路径。相机分辨率为256×1024，图还接收 LiDAR、自车状态和扩散噪声。不存在 S100/X5 资产，S100P/S600 文件也不能互换。URL 与校验和以 [S 发布清单](../../../../platforms/s/docs/release/models.yaml)为准。

<a id="preparation"></a>
## 显式准备

从仓库根目录执行，只下载需要的目标：

```bash
bash samples/vision/diffusiondrive/model/download.sh --target s100p
bash samples/vision/diffusiondrive/model/download.sh --target s600
```

`download_model.sh` 兼容包装入口转发同样参数。与源包装入口不同，下载必须显式选择目标，推理不会触发下载。新入口不以 `CHIP`、`RDK_SOC`、`MODEL_PATH` 环境覆盖作为资产选择契约。指定其他存储根目录：

```bash
python3 -m samples.vision.diffusiondrive.model.download --target s600 --output-dir /data/diffusiondrive-models
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | 必填 | `s100p` 或 `s600` |
| `--asset-id` | `null` | 按目标推断；显式值须精确匹配 |
| `--output-dir` | 当前 sample 的 `model` 目录 | 自动追加目标相对文件名 |

<a id="accompanying-files"></a>
## 配套文件

[SHA256SUMS](SHA256SUMS) 保留自源分支。无需类别名称文件。[测试数据](../test_data/README_cn.md)提供已准备的四输入浮点 NPZ 与参考输出，不提供原始 NAVSIM 传感器加载器。对照时保持输入噪声不变。[转换目录](../conversion/README_cn.md)保留两份 PTQ 配置，并说明缺失的导出/校准前提。

<a id="local-paths"></a>
## 本地路径与外部副本

默认解析到本 sample 下的 `model/s100p/diffusiondrive_r34_256x1024_s100p.hbm` 或 `model/s600/diffusiondrive_r34_256x1024_s600.hbm`。外部路径必须提供精确资产身份：

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --asset-id s:diffusiondrive:s600/diffusiondrive_r34_256x1024_s600.hbm --model-path /data/diffusiondrive-models/s600/diffusiondrive_r34_256x1024_s600.hbm --dry-run
```

Dry-run 只解析选择，不打开模型。实际加载检查物理身份、发布文件摘要及运行元数据。`auto` 可从显式资产 ID 推导契约；否则要求可识别的本机身份，不在开发主机上回退到 S600。摘要不同的自定义转换文件需显式增加资产/绑定，不能覆盖校验和绕过。

<a id="formats-checksums"></a>
## 格式与完整性

| 目标 | 发布 SHA-256 |
| --- | --- |
| S100P | `5829a1a318116e2eba5c3ba313841fbd064fe2d48fe3cd37eb1ec7d9a0ffa8fa` |
| S600 | `78605b5aaa573fbaa2bf4788c6c8922bcab39d90c195e264aa3a8d3a78321c53` |

下载和真实运行加载均验证这些摘要。源记录采用 OpenExplorer v3.7.0、INT16 优先/max PTQ、GridSample INT8 例外及全 BPU 落位。这些历史转换声明不能证明当前板端兼容或物理张量类型，加载时仍验证真实 IO 与量化信息。模型二进制不提交到 Git。本次主机迁移未下载或加载任一 HBM。
