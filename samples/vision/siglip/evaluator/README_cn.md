[English](./README.md) | 简体中文

# 模型评估 — SigLIP 视觉特征

本文所有数值表均为 S 平台 sample 和发布 benchmark 中的源记录，用于保留来源和可比性。

<a id="dataset"></a>
## 数据集

源记录 `pooler_output` 零样本分类使用 ImageNet-1k validation（50,000 张）。源记录 `last_hidden_state` 语义一致性使用 COCO2014 validation（5,000 张）。源资料没有发布准备脚本、精确压缩包版本、目录结构或评估实现。

```text
# cwd：仓库根目录
# 准备：未提供；不要从本文推断下载命令。
# 预期源数据布局：由评估负责人提供的 ImageNet-1k val 和 COCO2014 val
```

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

- 源记录测量：RDK S100 和 S100P，CPU/BPU 设置见下文。
- 对照流程：同一块板卡、板端 `hbm_runtime`、runtime 依赖（`numpy`、`opencv-python`、`PyYAML`）和本 sample 的 runtime。
- 板端及 runtime 版本：未固定。

<a id="command"></a>
## 评估命令

仓库没有评估脚本；runtime CLI 即功能入口（见 [runtime/python](../runtime/python/README_cn.md)）。同板对照可在相同图片和 HBM 上重复运行 runtime CLI，并比较两次的 JSON 输出与保存的 embedding。

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

对照流程将完整 raw 数组写入唯一的 `evaluator-output/siglip-raw-<UTC 微秒 run id>/legacy.npy` 和 `unified.npy`，不会用缩减摘要替代数组；数组即对照依据，未来评估可在旁边补充 JSON 记录。必须先相等 shape 和 dtype；整数 raw 必须完全相等，浮点 raw 允许 `rtol=0`、`atol=1e-5`，且断言必须通过。

<a id="reference-results"></a>
## 参考结果

以下两张源数据表保留全部行和列。来源：S 平台 evaluator README，并由 S 发布 benchmark 记录佐证。

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

- 本目录没有评估实现或数据集准备脚本；功能检查使用板端 runtime CLI。
- 四张表是源记录，本身不标识当前制品字节或 runtime 版本。
- 本 sample 仅评估视觉特征编码器，不覆盖文本编码器、文本 tokenizer、图文分数、校准配方或 C++ 评估器。

## 许可

评估文档和对照 helper 遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。保留源贡献者署名：Cauchy @吴超。
