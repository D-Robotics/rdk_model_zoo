[English](./README.md) | 简体中文

# DINOv2 ViT-S/14 视觉特征

<a id="overview"></a>
## 算法与来源

DINOv2 是一个生成全局图像特征和稠密 patch 特征的自监督 ViT 编码器。本 sample 将 ViT-S/14 以 int16 PTQ HBM 制品部署到 RDK S100、S100P、S600。上游实现和 Apache-2.0 模型制品来自 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)，转换固定使用 revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。

模型图包含 patch-14 stem、12 个 pre-LN transformer block、显式的 BPU 友好 attention 和最终归一化特征接口。sample 提供 `cls_feat` `(1,384)` 全局特征和 `patch_feat` `(1,256,384)` patch 特征。运行时依据绑定 metadata 将整数输出反量化为 owned float32 数组，不执行 softmax 或 L2 归一化。

<a id="support-matrix"></a>
## 支持与实测矩阵

唯一发布 variant 为 Nash-E、Nash-M、Nash-P 分别提供独立 HBM 制品。`supported-not-run` 表示已有本地契约和 fixture 覆盖，但本轮没有使用板卡。Python 为 supported-not-run；未提供 C++ runtime。

| Variant | x5 | s100 | s100p | s600 | Python | C++ |
| --- | --- | --- | --- | --- | --- | --- |
| `vits14-224-int16` | not-supported | supported-not-run | supported-not-run | supported-not-run | supported-not-run | not-supported |

板端验证证据：not-run。转换脚本来自源且已文档化，但本轮未执行。

<a id="prerequisites"></a>
## 环境前提

- 板端运行：RDK S100（Nash-E）、S100P（Nash-M）或 S600（Nash-P），板端镜像需提供 `hbm_runtime`。板端镜像和固件版本未核验。
- 主机契约检查：Python 3.14.7，以及 `requirements-host.txt` 中的 `numpy`、`opencv-python`、`PyYAML`。
- 转换环境：x86 Linux OE 3.7.0 镜像 `ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`；Torch 2.6 由镜像提供，追加 `onnx==1.19.0`、`onnxruntime==1.23.2`。本轮未执行转换。
- 板端推理前准备一个目标对应的 HBM；runtime 命令不会隐式下载模型。

<a id="quickstart"></a>
## 快速体验

从仓库根目录显式准备 S100 制品，再在板端运行 CLI。使用 `s100p` 或 `s600` 可选择对应的独立制品。

```bash
# cwd：仓库根目录；来源：docs/release/s/models.yaml 中的精确 URL
python3 samples/vision/dinov2/model/download.py --target s100
# 预期：samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm

# cwd：仓库根目录；输入：test_data/dog.jpg 和 test_data/bus.jpg
python3 samples/vision/dinov2/runtime/python/main.py --target s100 --output cls_feat
# 预期：打印 cls_feat JSON 摘要及第二张图 cosine_similarity；退出码 0
```

快捷脚本 `runtime/python/run.sh` 接收位置输出参数（`cls_feat` 或 `patch_feat`），后面可接命名参数；不会下载模型。兼容入口 `model/download_model.sh` 委托给显式 target 脚本，并要求 `s100`、`s100p` 或 `s600`。

<a id="expected-results"></a>
## 预期结果

CLI 打印包含 `output`、`shape`、`dtype`、`mean`、`std`、`min`、`max`、`l2_norm` 的 JSON 摘要。默认第二张图存在时，还打印 `second_image` 和 `cosine_similarity`；缺失时报告 `skipped_missing`。`cls_feat` shape 为 `(1,384)`，`patch_feat` 为 `(1,256,384)`，均由 metadata 绑定的反量化后以 float32 返回。目标板实际运行前不声明具体数值。

<a id="directory"></a>
## 目录职责

```text
dinov2/
├── conversion/             # 固定 ONNX 导出、校准和 hb_compile 配方
│   └── onnx_export/        # 源模型导出和图重写
├── evaluator/              # 历史性能和 cosine 记录
├── model/                  # 基于 manifest 的目标 HBM 准备
├── runtime/python/         # binding、runner、特征 task、tensor I/O 和 CLI
├── test_data/              # dog.jpg、bus.jpg 输入
└── README.md               # 英文说明
```

<a id="entry-points"></a>
## 入口索引

- 模型准备：[`model/README_cn.md`](model/README_cn.md) —— 三个目标对应的 HBM 制品和精确 URL。
- Python 运行：[`runtime/python/README_cn.md`](runtime/python/README_cn.md) —— 预处理、双输出 task API 和 CLI。
- C++ 运行：未提供；C++ 为 `not-supported`。
- 模型转换：[`conversion/README_cn.md`](conversion/README_cn.md) —— 固定源、ONNX 导出、校准和编译命令。
- 模型评估：[`evaluator/README_cn.md`](evaluator/README_cn.md) —— 历史板端表格和 cosine 复现条件。

<a id="license"></a>
## 许可说明

DINOv2 源模型和 checkpoint 是 Meta AI 通过 [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) 发布的 Apache-2.0 制品。示例代码遵循仓库 [LICENSE](../../../LICENSE) 的 Apache-2.0。保留源贡献者署名：D-Robotics model zoo 团队。
