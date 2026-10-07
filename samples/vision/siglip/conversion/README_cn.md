[English](./README.md) | 简体中文

# 模型转换 — SigLIP

<a id="source-model"></a>
## 源模型

已发布制品是基于 Google 来源 HuggingFace 权重的 SigLIP 视觉编码器。源 README 和发布 manifest 没有记录精确上游权重版本、commit、导出版本或制品级许可证。模型家族论文见 [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)。本 sample 提供可部署 HBM 引用，不提供源 checkpoint 或可复现导出器。

<a id="toolchain-targets"></a>
## 工具链与目标

源资料只说明 Nash BPU 部署，没有 OE 发布版本、编译器构建版本、`march` 命令或按 target 的配置文件。通用 OE 资源：[OE 环境文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) 和[工具链下载](https://toolchain.d-robotics.cc/)。支持映射为 S100 → Nash-E、S100P → Nash-M；八个发布制品均引用 `s100/` 路径。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| s100 | Nash-E | unknown | 源中无配置 |
| s100p | Nash-M | unknown | 源中无配置 |

<a id="export"></a>
## 导出（ONNX）

本 sample 未发布导出脚本或可复现的源 checkpoint 流程，因此不提供 ONNX 命令、导出器版本或已验证的 ONNX shape。已知可部署输入契约为：`_input_0`、float32 RGB NCHW `(1,3,size,size)`、值域 `[-1,1]`。

<a id="calibration"></a>
## 校准

没有发布校准数据集、样本数、量化配置、校准脚本或校准版本。源资料仅说明视觉编码器经过量化和编译，无法从本目录还原。

<a id="compile"></a>
## 编译

没有发布可复现编译命令、编译器版本、校准参数或命名配方。结果是 [`../model/README_cn.md`](../model/README_cn.md) 所列预编译 `.hbm` 文件。

<a id="validation"></a>
## 转换后验证

每个发布的 HBM 打包两个固定子模型——`pooler_output`（图像全局嵌入向量）和 `last_hidden_state`（patch 级视觉特征），各自输出 `_output_0`。源在板端检查下载制品的方式是先看 model info，再用单线程 `perf` 分别测两个子模型：

```bash
# cwd：仓库根目录，板端执行，制品下载完成后
hrt_model_exec model_info --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name pooler_output --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm
```

对其余七个制品替换文件名，执行相同的三条命令。源记录评测表保留为参考记录；新转换以下述运行验证。验证应绑定打包的两个子模型，检查 `_input_0` metadata 及所选 `_output_0` shape/dtype，然后在 S100、S100P 上分别运行 [`../runtime/python/README_cn.md`](../runtime/python/README_cn.md) 的板端冒烟路径。


<a id="artifacts"></a>
## 产物

| 产物 | Target | 落盘路径 |
| --- | --- | --- |
| `bpu-siglip-base-patch16-224.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-base-patch16-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-base-patch16-512.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-large-patch16-256.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-large-patch16-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch14-224.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch14-384.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |
| `bpu-siglip-so400m-patch16-256-i18n.hbm` | s100, s100p | `samples/vision/siglip/model/s100/` |

<a id="known-gaps"></a>
## 缺失项

- 缺少精确源 checkpoint/版本和导出脚本。
- 缺少 OE 版本、编译器构建版本、`march` 配置、校准数据/配置和编译命令。
- 缺少权重/导出许可证 metadata 和制品 hash（model README 记为 `sha256: null (unknown)`）。
- 当前可复现边界是制品选择、输入/输出契约文档和使用已发布 HBM；本目录无法重新生成 HBM。

## 许可

转换文档和示例代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。源发布没有记录额外制品许可证，因此不作断言。
