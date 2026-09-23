[English](./README.md) | 简体中文

# 模型制品 — DINOv2 ViT-S/14

<a id="artifacts"></a>
## 制品清单

每个 target 有独立 HBM 制品和 Nash march。制品是单模型双输出图，提供 `cls_feat` 和 `patch_feat`。

| 制品 | 格式 | Target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | s100 / Nash-E | `cls_feat`, `patch_feat` | download |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | s100p / Nash-M | `cls_feat`, `patch_feat` | download |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | s600 / Nash-P | `cls_feat`, `patch_feat` | download |

manifest 精确 URL：

| Target | 发布 URL |
| --- | --- |
| s100 | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/dinov2/nash-e/dinov2_vits14_224_int16_nashe.hbm> |
| s100p | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/dinov2/nash-m/dinov2_vits14_224_int16_nashm.hbm> |
| s600 | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/dinov2/nash-p/dinov2_vits14_224_int16_nashp.hbm> |

<a id="preparation"></a>
## 准备步骤

从仓库根目录运行，并选择具体 target。脚本不会猜测 target，也不会回退到 S100。

```bash
# cwd：仓库根目录；来源：上表精确 manifest URL
python3 samples/vision/dinov2/model/download.py --target s100
# 预期：samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm

# cwd：仓库根目录；另一个 target 和显式落盘目录
bash samples/vision/dinov2/model/download.sh s600 /tmp/dinov2-model
# 预期：/tmp/dinov2-model/nash-p/dinov2_vits14_224_int16_nashp.hbm
```

`download_model.sh` 是位置兼容入口，使用同样的必需 target 参数。manifest 的 SHA-256 均未知；下载器会打印观测 digest，并说明无法独立验证来源。I/O、选择或下载错误退出 2。本轮未下载制品。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `download.py` | 解析一个精确 manifest 制品并下载。 | 脚本准备时是；已有 HBM 时否。 |
| `download.sh` | 显式 target/output-dir shell wrapper。 | 否。 |
| `download_model.sh` | 委托 `download.sh` 的源兼容 wrapper。 | 否。 |
| `../test_data/dog.jpg`、`../test_data/bus.jpg` | 特征和 cosine 冒烟输入。 | 仅 demo 需要。 |

<a id="local-paths"></a>
## 本地路径

- 默认制品路径：按 target 分别为 `samples/vision/dinov2/model/nash-e/`、`nash-m/`、`nash-p/`。
- 不传 `--model-path` 时，runtime selection 会为具体 target 解析对应的精确 manifest 制品。
- 外部 `--model-path` 必须与精确 `--asset-id` 成对，例如 `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`。

<a id="formats-checksums"></a>
## 格式与校验值

active manifest 将三个发布 SHA-256 均记录为未知。下载观测 digest 不等于来源验证。

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

## 许可

模型由 Meta AI 发布的 Apache-2.0 DINOv2 权重量化而来。准备代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
