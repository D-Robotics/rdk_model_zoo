[English](README.md) | 简体中文

# EfficientSAM 评估器

<a id="dataset"></a>
## 数据集

将固定的 `test_data/dogs.jpg` 分别送入固定版本的参考实现与本 Sample 运行时，在同一目标板卡比较输入张量、原始输出、掩码选择和 IoU。保持图片、模型对与调度配置一致。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── compare.py  # Python 脚本
```

<a id="environment"></a>
## 环境

在仓库根目录使用目标板卡上的 Python 3.10+ 运行，并安装匹配运行时。评估器不会下载模型。选定的模型对必须已存在于清单推导路径，或者同时提供自定义 encoder/decoder 路径及各自精确清单 asset ID。板端 SDK/系统版本以目标板卡实际环境为准，运行时请一并记录。

<a id="command"></a>
## 命令

输出目录必须是新目录，已有目录会被拒绝。命令先检查请求的目标板卡，再使用相同图像、模型对、优先级和调度设置执行参考实现与 Sample 两侧：

```bash
# cwd：仓库根目录；前置：目标 runtime 和已准备模型对
python3 samples/vision/efficient_sam/evaluator/compare.py \
  --target x5 \
  --output-dir /absolute/new-evidence-dir
```

S 系列使用 `--target s100|s100p|s600`。可选的 `--test-img`、`--encoder-model-path`、`--decoder-model-path`、配对的 `--encoder-asset-id`/`--decoder-asset-id`、`--priority`（默认 `0`）和 `--bpu-cores` 会传给两侧。省略 `--bpu-cores` 时 S 使用 `[0]`；X5 不支持显式 core 选择。退出码 `0` 表示全部检查通过，`1` 表示执行完成但有比较失败，`2` 表示目标板卡检查、参数、模型、图像或执行失败。

两侧执行使用相同的按模型名调度参数。`comparison.json` 保存模型名称、原生调用参数及每次调度调用的结果；比较运行结果时请核对实际生效的配置。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--target` | 必填 | `x5`、`s100`、`s100p` 或 `s600`；会检查实际执行目标板卡 |
| `--output-dir` | 必填 | 新的绝对或相对证据目录；已有目录会被拒绝 |
| `--test-img` | `samples/vision/efficient_sam/test_data/dogs.jpg` | 固定 BGR 示例输入 |
| `--encoder-model-path`、`--decoder-model-path` | 清单推导路径 | 自定义路径，各自必须配精确 asset ID |
| `--encoder-asset-id`、`--decoder-asset-id` | `null` | 自定义路径使用的限定清单 identity |
| `--priority` | `0` | `0..255` 的整数调度优先级 |
| `--bpu-cores` | `null` | S 系列 core index；省略为 `[0]`，X5 拒绝显式 core |

<a id="metrics"></a>
## 指标

比较要求输入形状、数据类型和数值完全相同；raw 输出形状、数据类型完全相同且数值绝对误差 `1e-5`、相对误差 `0`；bool 掩码和选中掩码 index 完全相同；IoU 绝对误差 `1e-6`；低分辨率 float32 掩码误差 `1e-5`。没有 tie 或 changed-mask 豁免。

<a id="outputs"></a>
## 输出

新目录包含 `comparison.json` 以及两侧的 NumPy 数组：encoder/decoder 输入、raw 输出、选中的 bool 掩码和低分辨率掩码。JSON 记录目标板卡、完整模型身份及实际摘要、图像 hash/shape/dtype、元数据、调度、参考实现、命令上下文、代码摘要、逐项判定和容差。执行或比较失败时，已写入的证据仍会保留。

<a id="reference-results"></a>
## 参考结果

下表列出各板卡上编码器与解码器的参考性能，测量条件见下文：

| 板卡 | 模型阶段 | 线程数 | 延迟（ms） | 吞吐（FPS） |
|---|---|---:|---:|---:|
| X5 | 编码器 | 1 | 1451.073 | 0.689135 |
| X5 | 编码器 | 8 | 1974.671 | 3.965380 |
| X5 | 解码器 | 1 | 86.532 | 11.553175 |
| X5 | 解码器 | 8 | 155.994 | 50.565231 |
| S100 | 编码器 | 1 / 2 | 11.78 / 22.93 | 84.75 / 86.95 |
| S100 | 解码器 | 1 / 2 | 3.25 / 5.94 | 306.31 / 334.44 |
| S100P | 编码器 | 1 / 2 | 9.36 / 18.20 | 106.69 / 109.52 |
| S100P | 解码器 | 1 / 2 | 2.49 / 4.47 | 399.96 / 445.74 |
| S600 | 编码器 | 1 / 12 | 6.72 / 19.58 | 148.60 / 598.44 |
| S600 | 解码器 | 1 / 12 | 1.50 / 4.08 | 662.55 / 2831.10 |

### X5 量化余弦（源记录）

`rdk_x5` 样本 README 的 Validation 记录了 RDK X5 上下列制品的 cosine。原文没有写对照张量或图片数量。

| 制品 | 范围 | cosine |
| --- | --- | --- |
| `efficient_sam_vitt_encoder_512x512_default_none.bin` | encoder final | 0.968013 |
| `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | fixed-prompt `low_res_masks` | 0.965641 |
| `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | fixed-prompt `iou_predictions` | 0.997313 |

运行完成后读取输出目录中的 `comparison.json` 和 NumPy 数组，比较输入、原始输出与掩码结果。

### 单模型性能测量

使用匹配板卡开发套件提供的 `hrt_model_exec perf` 分别测量编码器和解码器。先按模型指南准备模型对，再将下方 `TARGET` 设为实际板卡身份。

```bash
# Bash; cwd: repository root; run only on the matching prepared board
cd samples/vision/efficient_sam/evaluator
TARGET=s100
CORE_ARGS=()
case "$TARGET" in
  x5)
    ENCODER=../model/efficient_sam_vitt_encoder_512x512_default_none.bin
    DECODER=../model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin
    THREADS=8 ;;
  s100|s100p|s600)
    case "$TARGET" in
      s100) MARCH=nash-e; SUFFIX=nashe; THREADS=2 ;;
      s100p) MARCH=nash-m; SUFFIX=nashm; THREADS=2 ;;
      s600) MARCH=nash-p; SUFFIX=nashp; THREADS=12; CORE_ARGS=(--core_id 1,2,3,4) ;;
    esac
    ENCODER=../model/$MARCH/efficient_sam_vitt_encoder_512x512_$SUFFIX.hbm
    DECODER=../model/$MARCH/efficient_sam_vitt_decoder_512_$SUFFIX.hbm ;;
  *) exit 2 ;;
esac
for STAGE_MODEL in "$ENCODER" "$DECODER"; do
  hrt_model_exec perf --model_file "$STAGE_MODEL" --thread_num 1
  hrt_model_exec perf --model_file "$STAGE_MODEL" --thread_num "$THREADS" "${CORE_ARGS[@]}"
done
```

X5 参考记录使用工具默认 200 帧；S 源未记录帧数，复测时请记录帧数与工具/SDK 版本。S100/S100P 多线程为 2，S600 为 12，且 S600 多线程显式使用 `--core_id 1,2,3,4`。该工具的 core ID 参数不可照搬为 Python 运行时的 `--bpu-cores`。复测时保存完整命令、工具/系统版本、设备身份和输出；不要只保存汇总 FPS。

### 测量口径（参考记录）

源 S 表中的附加模型信息如下；参数量与 FLOPs 来自原 FP32 模型，不是量化模型大小。FLOPs 按 `2×MACs` 口径记录。类别数均为 `-`（类别无关），CPU 前后处理延迟均未记录。

| Stage | 输入规模 | Params (M) | FLOPs (G) |
| --- | --- | ---: | ---: |
| 编码器 | RGB 512×512 | 6.16 | 22.19 |
| 解码器 | 256×32×32 嵌入特征 | 4.06 | 0.98 |

源 S 说明的 BPU 延迟为任务提交到完成，包含缓存预热；流式测量复用预分配内存，不计分配/释放。输入是 float32 张量，不是 NV12。两个阶段顺序执行，完整流水线延迟还包括 CPU 预处理、掩码缩放等开销，不能将单阶段 FPS 当作完整示例的吞吐。以上为参考记录条件；本实现的性能按同一口径在目标板上实测。

<a id="boundaries"></a>
## 适用范围

`compare.py` 比较固定图片在参考实现与 Sample 运行时中的张量和掩码一致性。性能计时使用上方单模型命令；精度评估使用带标注的数据集。
