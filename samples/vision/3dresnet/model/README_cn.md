[English](./README.md) | 简体中文

# R3D-18 模型准备

<a id="artifacts"></a>
## 制品清单

| Manifest asset | Target | 阶段 | 格式 | 来源 |
| --- | --- | --- | --- | --- |
| `s:3dresnet:s100/r3d_18.hbm` | `s100` | runtime 推理 | HBM | `docs/release/s/models.yaml` |

本 sample 只有一个已发布制品；没有 x5、S100P、S600、ONNX、checkpoint 或校准制品。

<a id="preparation"></a>
## 准备步骤

在仓库根目录显式准备精确的 manifest 行：

```bash
# cwd：仓库根目录
bash samples/vision/3dresnet/model/download.sh s100
# 预期：samples/vision/3dresnet/model/s100/r3d_18.hbm
```

下载器从共享 S manifest 解析 `s:3dresnet:s100/r3d_18.hbm`，并使用共享的原子下载/校验 helper。当前 manifest 没有 publisher SHA-256，因此会报告观测摘要，但不能独立证明来源。已有文件会先校验，不会静默覆盖。`download_model.sh` 是委托到同一显式命令的兼容入口。runtime 不会自动下载模型。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 用于 | 作用 |
| --- | --- | --- |
| `../test_data/video0.npy` | runtime smoke test | 已归一化的 RGB float32 片段，shape `(1,3,16,112,112)` |
| `../test_data/kinetics_classnames.json` | CLI 标签展示 | 400 条 source `name → id` 映射，读取后按 source 去除引号并反转为 `id → name` |

`../test_data/readme_img/` 下的截图只用于文档证据。

<a id="local-paths"></a>
## 本地路径

准备完成后的模型位置：

```text
samples/vision/3dresnet/model/s100/r3d_18.hbm
```

不提供 `--model-path` 时，runtime 根据精确选中的 manifest asset 解析该路径。外部模型路径必须同时提供 `--asset-id s:3dresnet:s100/r3d_18.hbm`。

<a id="formats-checksums"></a>
## 格式与校验值

| 制品 | 格式 | SHA-256 |
| --- | --- | --- |
| `s100/r3d_18.hbm` | HBM | `sha256: null (unknown)` |

`null` 来自当前 manifest；没有猜测或从其他制品复制校验值。
