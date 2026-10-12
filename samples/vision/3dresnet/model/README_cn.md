[English](README.md) | 简体中文

# R3D-18 模型准备

<a id="artifacts"></a>
## 制品清单

| Manifest asset | Target | 阶段 | 格式 | 来源 |
| --- | --- | --- | --- | --- |
| `s:3dresnet:s100/r3d_18.hbm` | `s100` | 运行时推理 | HBM | `docs/release/s/models.yaml` |

本示例只有一个已发布制品；没有 x5、S100P、S600、ONNX、权重文件或校准制品。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── download_model.sh  # Shell 脚本
```

<a id="preparation"></a>
## 准备步骤

在仓库根目录显式准备精确的清单行：

```bash
# cwd：仓库根目录
bash samples/vision/3dresnet/model/download.sh s100
# 预期：samples/vision/3dresnet/model/s100/r3d_18.hbm
```

命令按 S 模型清单下载 `s:3dresnet:s100/r3d_18.hbm`，并报告本地 SHA-256；发布者校验和未提供。已有文件会先检查再复用。`download_model.sh` 执行相同准备命令。推理前先完成模型准备。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 用于 | 作用 |
| --- | --- | --- |
| `../test_data/video0.npy` | example inference | 已归一化的 RGB float32 片段，形状 `(1,3,16,112,112)` |
| `../test_data/kinetics_classnames.json` | CLI 标签展示 | 400 条 `name → id` 映射，读取后反转为 `id → name`（名称中内嵌的引号字符会被去除） |

`../test_data/readme_img/` 下的截图只用于文档证据。

<a id="local-paths"></a>
## 本地路径

准备完成后的模型位置：

```text
samples/vision/3dresnet/model/s100/r3d_18.hbm
```

不提供 `--model-path` 时，运行时根据精确选中的清单 asset 解析该路径。外部模型路径必须同时提供 `--asset-id s:3dresnet:s100/r3d_18.hbm`。

<a id="formats-checksums"></a>
## 格式与校验值

| 制品 | 格式 | SHA-256 |
| --- | --- | --- |
| `s100/r3d_18.hbm` | HBM | `sha256: null (unknown)` |

`null` 与当前清单一致；没有猜测或从其他制品复制校验值。
