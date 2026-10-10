[English](README.md) | 简体中文

# 模型评估 — SigLIP 视觉特征

本指南说明图像特征比较，以及已发布的 S100/S100P 精度与性能测量。

<a id="dataset"></a>
## 数据集

`evaluate.py` 直接使用仓库自带图片：anchor 与 negative 为 `samples/vision/dinov2/test_data/dog.jpg` 和 `bus.jpg`，positive 为 `samples/vision/mobile_sam/test_data/dogs.jpg`，无需下载数据集。下方参考表由发布方在 ImageNet-1k validation（50,000 张）上测零样本分类、在 COCO2014 validation（5,000 张）上测 patch 特征一致性；源侧评估实现未发布，因此这些表是参考条件，不是本地可复现的基准。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── evaluate.py  # 基础图像语义关系检查
```

<a id="environment"></a>
## 环境

- 源记录测量：RDK S100 和 S100P，CPU/BPU 设置见下文。
- 对照流程：同一块板卡、板端 `hbm_runtime`、runtime 依赖（`numpy`、`opencv-python`、`PyYAML`）和本 sample 的 runtime。
- 板端及 runtime 版本：未固定。

<a id="command"></a>
## 评估命令

`evaluate.py` 分别使用 `pooler_output` 检查全局图像语义，使用 `last_hidden_state` 检查对齐 patch 的输入一致性。基准图片是小狗，独立正例图片是两只狗，负例图片是公交车。图片之间的关系应在推理前确定。

```bash
# S100 板端，仓库根目录；输出目录必须尚不存在。
python3 samples/vision/siglip/evaluator/evaluate.py \
  --target s100 \
  --asset-id s:siglip:s100/bpu-siglip-base-patch16-224.hbm \
  --model-path samples/vision/siglip/model/s100/bpu-siglip-base-patch16-224.hbm \
  --anchor samples/vision/dinov2/test_data/dog.jpg \
  --positive samples/vision/mobile_sam/test_data/dogs.jpg \
  --negative samples/vision/dinov2/test_data/bus.jpg \
  --output-dir outputs/siglip-relations
```

`pooler_output` 要求：重复输入 cosine ≥ 0.99999，轻度变换 cosine ≥ 0.95，正例 cosine 减负例 cosine > 0.05。`last_hidden_state` 保留 token 位置，要求展平后的重复输入 cosine ≥ 0.99999，轻度变换 cosine ≥ 0.95，轻度变换 cosine 减无关图片 cosine > 0.05。其 token 均值图像排序作为诊断数据保存。轻度变换为 `clip(BGR * 0.9 + 5, 0, 255).astype(uint8)`。`result.json` 记录阈值、逐角色检查、元数据与输入/模型哈希；逐角色 `.npz` 保存基准、重复、变换、正例和负例特征。所有角色通过时返回 0；关系检查失败或模型/数值错误时记录结果并返回 1。

runtime CLI 也提供单个 embedding（见 [runtime/python](../runtime/python/README_cn.md)）。同板对照可在相同图片和 HBM 上重复运行 runtime CLI，并比较两次的 JSON 输出与保存的 embedding。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| `pooler_output` 延迟 | 全局特征子模型的一次 BPU `perf` 测量。 | 单线程；输入分辨率和输出 shape 见表；使用下述板端 CPU/BPU 设置。 |
| `last_hidden_state` 延迟 | patch 特征子模型的一次 BPU `perf` 测量。 | 单线程；输入分辨率和输出 shape 见表；使用下述板端 CPU/BPU 设置。 |
| TOP1/TOP5 | 由全局嵌入得到的 ImageNet 零样本分类准确率。 | ImageNet-1k val，50,000 张；浮点和 BPU 路径都使用 RGB `(127,127,127)` letterbox。 |
| Cosine Similarity | patch 特征相对参照的平均/最小~最大及 1% low 相似度。 | COCO2014 val，5,000 张；相同 RGB letterbox。 |
| MSE | patch 特征相对参照的平均/最小~最大及 1% low 均方误差。 | COCO2014 val，5,000 张；相同 RGB letterbox。 |

源记录板端设置：

- S100：CPU `6 x A78AE @ 1.5GHz`，BPU `1 x Nash-E @ 1.0GHz`。
- S100P：CPU `6 x A78AE @ 2.0GHz`，BPU `1 x Nash-M @ 1.5GHz`。
- 源资料记录了 CPU policy 0/4 和 BPU `28108000.bpu` 的 performance governor 命令。

### 源 `pooler_output` 性能

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,1,768)` | `0.2 B / 0.09 B` | 26.8 ms | 18.8 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,1,768)` | `0.2 B / 0.09 B` | 46.7 ms | 32.3 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1,768)` | `0.2 B / 0.09 B` | 81.7 ms | 55.8 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 68.8 ms | 47.2 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 132.5 ms | 91.4 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 89.8 ms | 62.2 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 255.7 ms | 175.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,1,1152)` | `1.0 B / 0.43 B` | 89.6 ms | 61.9 ms |

### 源 `last_hidden_state` 性能

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,196,768)` | `0.2 B / 0.09 B` | 26.0 ms | 18.3 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,576,768)` | `0.2 B / 0.09 B` | 45.9 ms | 31.7 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1024,768)` | `0.2 B / 0.09 B` | 80.8 ms | 55.3 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,256,1024)` | `0.7 B / 0.32 B` | 67.6 ms | 46.5 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,576,1024)` | `0.7 B / 0.32 B` | 131.3 ms | 90.5 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,256,1152)` | `0.9 B / 0.43 B` | 88.6 ms | 61.4 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,729,1152)` | `0.9 B / 0.43 B` | 254.2 ms | 174.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,256,1152)` | `1.0 B / 0.43 B` | 88.3 ms | 61.1 ms |

<a id="outputs"></a>
## 输出

评估结果写入命令中的 `--output-dir`：`result.json`（schema `rdk-model-zoo/embedding-relations/v1`）记录 target、asset ID、模型/输入 SHA-256、轻度变换定义、各 role 的 metadata、判定标准、cosine 数值与逐项检查；同目录下 `pooler_output.npz` 与 `last_hidden_state.npz` 分别保存该 role 的五个特征数组（`anchor`、`repeat`、`mild`、`positive`、`negative`），JSON 中另有 `feature_shape`/`feature_dtype`。全部 role 通过返回 0；关系不成立或模型/数值错误记录后返回 1。

<a id="reference-results"></a>
## 参考结果

以下表格列出 S100 和 S100P 的 ImageNet-1k 零样本分类与 COCO2014 patch 特征一致性。

### 源 `pooler_output` 零样本分类

| Model Name | PyTorch TOP1 / TOP5 | BPU TOP1 / TOP5 |
|---|---|---|
| siglip-base-patch16-224 | 0.7123 / 0.9143 | 0.7118 / 0.9144 |
| siglip-base-patch16-384 | 0.7411 / 0.9318 | 0.7418 / 0.9319 |
| siglip-base-patch16-512 | 0.7490 / 0.9343 | 0.7482 / 0.9340 |
| siglip-large-patch16-256 | 0.7490 / 0.9238 | 0.7490 / 0.9242 |
| siglip-large-patch16-384 | 0.7584 / 0.9252 | 0.7595 / 0.9256 |
| siglip-so400m-patch14-224 | 0.7659 / 0.9361 | 0.7651 / 0.9357 |
| siglip-so400m-patch14-384 | 0.7872 / 0.9433 | 0.7893 / 0.9447 |
| siglip-so400m-patch16-256-i18n | 0.7678 / 0.9395 | 0.7668 / 0.9397 |

### 源 `last_hidden_state` 语义一致性

| Model Name | Cosine Similarity mean (min ~ max), 1% low | MSE mean (min ~ max), 1% low |
|---|---|---|
| siglip-base-patch16-224 | 0.991 (0.951 ~ 0.997), 0.980 | 0.087 (0.024 ~ 0.471), 0.039 |
| siglip-base-patch16-384 | 0.989 (0.960 ~ 0.997), 0.977 | 0.113 (0.029 ~ 0.409), 0.050 |
| siglip-base-patch16-512 | 0.987 (0.956 ~ 0.995), 0.974 | 0.142 (0.045 ~ 0.507), 0.067 |
| siglip-large-patch16-256 | 0.990 (0.933 ~ 0.997), 0.974 | 0.069 (0.018 ~ 0.497), 0.024 |
| siglip-large-patch16-384 | 0.985 (0.900 ~ 0.995), 0.965 | 0.111 (0.034 ~ 0.775), 0.048 |
| siglip-so400m-patch14-224 | 0.984 (0.850 ~ 0.995), 0.961 | 0.104 (0.028 ~ 1.038), 0.041 |
| siglip-so400m-patch14-384 | 0.980 (0.859 ~ 0.993), 0.957 | 0.140 (0.040 ~ 1.093), 0.059 |
| siglip-so400m-patch16-256-i18n | 0.984 (0.878 ~ 0.996), 0.959 | 0.082 (0.018 ~ 0.570), 0.030 |

<a id="boundaries"></a>
## 适用范围

- `evaluate.py` 检查两个打包子模型上的指定图像关系。
- 四张表是源记录，本身不标识当前制品字节或 runtime 版本。
- 本 sample 仅评估视觉特征编码器，不覆盖文本编码器、文本 tokenizer、图文分数、校准配方或 C++ 评估器。

## 许可

评估文档和对照 helper 遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。保留源贡献者署名：Cauchy @吴超。
