[English](./README.md) | 简体中文

# 模型制品 — SigLIP

<a id="artifacts"></a>
## 制品清单

每行对应一个 manifest 制品。`s100/` 是发布存储身份，同时用于支持的 S100（Nash-E）和 S100P（Nash-M）；它不是删去 S100P 支持的回退路径。每个 HBM 都是包含两个视觉特征子模型的打包制品。

| 制品 | 格式 | Target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `s100/bpu-siglip-base-patch16-224.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-base-patch16-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-base-patch16-512.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-large-patch16-256.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-large-patch16-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch14-224.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch14-384.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |
| `s100/bpu-siglip-so400m-patch16-256-i18n.hbm` | HBM | s100, s100p | `pooler_output`, `last_hidden_state` | download |

manifest 中的精确 URL：

| Variant | 发布 URL |
| --- | --- |
| `base-patch16-224` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-224.hbm> |
| `base-patch16-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-384.hbm> |
| `base-patch16-512` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-512.hbm> |
| `large-patch16-256` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-256.hbm> |
| `large-patch16-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-384.hbm> |
| `so400m-patch14-224` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-224.hbm> |
| `so400m-patch14-384` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-384.hbm> |
| `so400m-patch16-256-i18n` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch16-256-i18n.hbm> |

<a id="preparation"></a>
## 准备步骤

以下命令均从仓库根目录运行。运行时不会隐式下载模型。脚本支持 `--target s100|s100p`、`--variant`、`--output-dir`；shell wrapper 按位置接收 target、variant。

```bash
# cwd：仓库根目录；来源：docs/release/s/models.yaml 中记录的精确 URL
python3 samples/vision/siglip/model/download.py --target s100 --variant base-patch16-224
# 预期：samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm，并打印观测 SHA-256

# cwd：仓库根目录；S100P 使用相同发布字节
bash samples/vision/siglip/model/download.sh s100p so400m-patch14-384
# 预期：samples/vision/siglip/model/s100/bpu-siglip-so400m-patch14-384.hbm
```

manifest 没有发布方 SHA-256。下载器会打印观测摘要，并明确说明无法校验来源。I/O、选择或下载错误退出 2。主途径不可用时，可将发布 URL 的精确 HBM 手动放到上述路径，并在 runtime 中同时传入精确 `--asset-id s:siglip:s100/bpu-siglip-<variant>.hbm` 与 `--model-path <path>`；手动来源仍未验证。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `download.py` | 解析 manifest 身份并下载一个 HBM。 | 已有制品时否；脚本准备时是。 |
| `download.sh` | `download.py` 的位置参数兼容 wrapper。 | 否。 |
| `../test_data/dog.jpg` | runtime 冒烟输入，不属于 HBM。 | 仅运行 sample CLI 时需要。 |

<a id="local-paths"></a>
## 本地路径

- 制品：`samples/vision/siglip/model/s100/bpu-siglip-<variant>.hbm`。
- runtime 默认 `--model-path`：`resolve_selection` 选择的同一 `model/s100/bpu-siglip-base-patch16-224.hbm`。
- 显式 `--model-path` 必须与精确 manifest `--asset-id` 成对提供。

<a id="formats-checksums"></a>
## 格式与校验值

源 inventory 和发布 manifest 都将发布方 hash 记录为未知；不在制品之间复制 hash。

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| `s100/bpu-siglip-base-patch16-224.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-base-patch16-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-base-patch16-512.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-large-patch16-256.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-large-patch16-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch14-224.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch14-384.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `s100/bpu-siglip-so400m-patch16-256-i18n.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

发布 URL 是该 manifest 中的精确 `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-<variant>.hbm` 条目。本轮迁移未执行正式下载。

## 许可

准备脚本遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。源发布记录没有制品级模型许可证或权重版本，因此此处不虚构。
