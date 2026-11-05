[English](README.md) | 简体中文

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

从仓库根目录运行，通过 target 指定 `s100`、`s100p` 或 `s600`。

```bash
# cwd：仓库根目录；来源：上表精确 manifest URL
python3 samples/vision/dinov2/model/download.py --target s100
# 预期：samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm

# cwd：仓库根目录；另一个 target 和显式落盘目录
bash samples/vision/dinov2/model/download.sh s600 /tmp/dinov2-model
# 预期：/tmp/dinov2-model/nash-p/dinov2_vits14_224_int16_nashp.hbm
```

`download_model.sh` 是位置快捷入口，使用同样的必需 target 参数。历史三条 manifest 的 SHA-256 仍未知；新转换制品的精确摘要见下文。下载器打印下载文件的 SHA-256。I/O、选择或下载错误退出 2。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `download.py` | 解析一个精确 manifest 制品并下载。 | 脚本准备时是；已有 HBM 时否。 |
| `download.sh` | 显式 target/output-dir shell 脚本。 | 否。 |
| `download_model.sh` | 委托 `download.sh` 的 脚本。 | 否。 |
| `../test_data/dog.jpg`、`../test_data/bus.jpg` | 特征和 cosine 冒烟输入。 | 仅 demo 需要。 |

<a id="local-paths"></a>
## 本地路径

- 默认制品路径：按 target 分别为 `samples/vision/dinov2/model/nash-e/`、`nash-m/`、`nash-p/`。
- 不传 `--model-path` 时，runtime selection 会为具体 target 解析对应的精确 manifest 制品。
- 外部 `--model-path` 必须与精确 `--asset-id` 成对，例如 `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`。

<a id="formats-checksums"></a>
## 格式与校验值

历史三个制品的发布 SHA-256 仍记录为未知；下载观测 digest 用于跨主机识别同一份字节，发布者认证以发布摘要为准。

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| `nash-e/dinov2_vits14_224_int16_nashe.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-m/dinov2_vits14_224_int16_nashm.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |
| `nash-p/dinov2_vits14_224_int16_nashp.hbm` | HBM | `sha256: null (unknown)` | `docs/release/s/models.yaml` |

## 新转换制品（2026-11-02）

显式版本 `vits14_224_int16_20261102_o1` 当前提供 3 个分别编译的 O1 HBM，目标为 `s100`, `s100p`, `s600`。不传 `--variant` 或 `--asset-id` 时仍选择历史默认模型。新文件已通过浮点图对照、原装转换及 HBM 输入输出/架构静态检查；板端功能、精度、Python/C++ 性能均为 **not-run**，历史测量数据不适用于这些文件。

| Target | HBM | SHA-256 |
| --- | --- | --- |
| s100 | `dinov2_vits14_224_int16_nashe_20261102_o1.hbm` | `40d9654edb2b83b12928a4b23e8f482ec30c415d2389af5e64957d9921a9df1a` |
| s100p | `dinov2_vits14_224_int16_nashm_20261102_o1.hbm` | `05a94aae2cb7da14a6a0810c7c7e5e5bff604012b943dd66708e4cd9cc224cbe` |
| s600 | `dinov2_vits14_224_int16_nashp_20261102_o1.hbm` | `e7b720b39d5475d440d8631ba0ac008f230fa28e8a5b1d12d887a60a7cd2c549` |

```bash
python3 samples/vision/dinov2/model/download.py --target s100 --variant vits14_224_int16_20261102_o1
python3 samples/vision/dinov2/runtime/python/main.py --target s100 --variant vits14_224_int16_20261102_o1 --dry-run
```

此版本仅开放表中已完成的目标；尚未发布的目标会报错，不会回退。也可用完整的 `s:dinov2:nash-*/filename.hbm` 资产 ID 选择。输入仍是已归一化 RGB float32 FEATUREMAP `(1,3,224,224)`；实际公开输出为 float32，沿用元数据绑定的 `raw_f32` 路径；内部量化仍为 int16 配方，不添加 L2 归一化或 softmax。来源/权重版本、完整哈希、50 张真实图像校准配方、实际环境和验证记录见[转换记录](../conversion/converted.json)。

## 许可

模型由 Meta AI 发布的 Apache-2.0 DINOv2 权重量化而来。准备代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
