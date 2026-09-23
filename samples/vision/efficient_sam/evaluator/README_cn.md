[English](README.md) | 简体中文

# EfficientSAM 迁移评估器

<a id="dataset"></a>
## 数据集

本评估器将固定的 `test_data/dogs.jpg` 图像分别送入同一 target 上的 source legacy 入口和统一 runtime，比较迁移一致性。它只衡量该固定图像的一致性，不是数据集精度或性能基准。

<a id="environment"></a>
## 环境

在仓库根目录使用目标板卡上的 Python 3.10+ 运行，并安装匹配 runtime。评估器不会下载模型。选定的模型对必须已存在于 manifest 推导路径，或者同时提供自定义 encoder/decoder 路径及各自精确 manifest asset ID。板端 SDK/系统版本未知；此次迁移板端为 not-run。

<a id="command"></a>
## 命令

输出目录必须是新目录，已有目录会被拒绝。命令先 gate 请求的 target，再使用相同图像、模型对、优先级和调度设置执行 legacy 与统一两侧：

```bash
# cwd：仓库根目录；前置：目标 runtime 和已准备模型对
python3 samples/vision/efficient_sam/evaluator/compare.py \
  --target x5 \
  --output-dir /absolute/new-evidence-dir
```

S 系列使用 `--target s100|s100p|s600`。可选的 `--test-img`、`--encoder-model-path`、`--decoder-model-path`、配对的 `--encoder-asset-id`/`--decoder-asset-id`、`--priority`（默认 `0`）和 `--bpu-cores` 会传给两侧。省略 `--bpu-cores` 时 S 使用 `[0]`；X5 不支持显式 core 选择。退出码 `0` 表示全部检查通过，`1` 表示执行完成但有比较失败，`2` 表示 target gate、参数、模型、图像或执行失败。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--target` | 必填 | `x5`、`s100`、`s100p` 或 `s600`；会 gate 实际执行 target |
| `--output-dir` | 必填 | 新的绝对或相对证据目录；已有目录会被拒绝 |
| `--test-img` | `samples/vision/efficient_sam/test_data/dogs.jpg` | 固定 BGR fixture |
| `--encoder-model-path`、`--decoder-model-path` | manifest 推导路径 | 自定义路径，各自必须配精确 asset ID |
| `--encoder-asset-id`、`--decoder-asset-id` | `null` | 自定义路径使用的限定 manifest identity |
| `--priority` | `0` | `0..255` 的整数调度优先级 |
| `--bpu-cores` | `null` | S 系列 core index；省略为 `[0]`，X5 拒绝显式 core |

<a id="metrics"></a>
## 指标

比较要求输入 shape、dtype 和数值完全相同；raw 输出 shape、dtype 完全相同且数值绝对误差 `1e-5`、相对误差 `0`；bool mask 和选中 mask index 完全相同；IoU 绝对误差 `1e-6`；低分辨率 float32 mask 误差 `1e-5`。没有 tie 或 changed-mask 豁免。

<a id="outputs"></a>
## 输出

新目录包含 `comparison.json` 以及两侧的 NumPy 数组：encoder/decoder 输入、raw 输出、选中的 bool mask 和低分辨率 mask。JSON 记录 target、完整模型身份及实际 hash、图像 hash/shape/dtype、metadata、调度、source reference、命令上下文、代码 hash、逐项判定和容差。执行或比较失败时，已写入的证据仍会保留。

<a id="reference-results"></a>
## 参考结果

下表是从固定 source evaluator README 转录的历史测量值，仅作背景；它们不是统一代码树的结果，也不证明当前板端支持：

| Source target | Stage | Threads | 历史 latency (ms) | 历史 FPS |
|---|---|---:|---:|---:|
| X5 | encoder | 1 | 1451.073 | 0.689135 |
| X5 | encoder | 8 | 1974.671 | 3.965380 |
| X5 | decoder | 1 | 86.532 | 11.553175 |
| X5 | decoder | 8 | 155.994 | 50.565231 |
| S100 | encoder | 1 / 2 | 11.78 / 22.93 | 84.75 / 86.95 |
| S100 | decoder | 1 / 2 | 3.25 / 5.94 | 306.31 / 334.44 |
| S100P | encoder | 1 / 2 | 9.36 / 18.20 | 106.69 / 109.52 |
| S100P | decoder | 1 / 2 | 2.49 / 4.47 | 399.96 / 445.74 |
| S600 | encoder | 1 / 12 | 6.72 / 19.58 | 148.60 / 598.44 |
| S600 | decoder | 1 / 12 | 1.50 / 4.08 | 662.55 / 2831.10 |

每次指定 target 和已准备模型对的本地或板端成功执行，其完整输出目录就是该次参考证据；请保留目录供复核。

### 保留的单模型性能测试入口

`compare.py` 只检查迁移一致性。原样例另有 encoder、decoder 分别测量的 `hrt_model_exec perf` 能力，以下保留该流程；本次未执行。工具来自匹配板卡的开发套件，不通过 pip 安装。先按 model README 准备对应模型；将 `TARGET` 改为实际板卡，不能跨板复用文件。以下 shell 自身不校验硬件身份，执行前应按 runtime README 核对身份。

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

X5 源记录使用工具默认 200 帧；S 源未记录帧数，当前工具默认值/SDK 版本均未验证。S100/S100P 多线程为 2，S600 为 12，且 S600 多线程显式使用 `--core_id 1,2,3,4`。该工具的 core ID 参数不可照搬为 Python runtime 的 `--bpu-cores`。复测时保存完整命令、工具/系统版本、设备身份和输出；不要只保存汇总 FPS。

### 历史记录的测量口径

源 S 表中的附加模型信息如下；参数量与 FLOPs 来自原 FP32 模型，不是本次计算或量化模型大小。FLOPs 按 `2×MACs` 口径记录。类别数均为 `-`（类别无关），CPU 前后处理延迟均未记录。

| Stage | 输入规模 | Params (M) | FLOPs (G) |
| --- | --- | ---: | ---: |
| encoder | RGB 512×512 | 6.16 | 22.19 |
| decoder | 256×32×32 embedding | 4.06 | 0.98 |

源 S 说明的 BPU 延迟为任务提交到完成，包含缓存预热；流式测量复用预分配内存，不计分配/释放。输入是 float32 张量，不是 NV12。两个阶段顺序执行，完整流水线延迟还包括 CPU 预处理、掩码缩放等开销，不能将单阶段 FPS 当作完整 sample 的吞吐。这些均为历史条件，不是统一实现已实测的性能保证。

<a id="boundaries"></a>
## 边界

`compare.py` 不会下载制品、测试完整数据集、测量延迟、认证精度，也不会证明未显式 gate 和执行的 target。它使用相同 runtime 调用路径，对固定 source legacy 实现和统一实现进行比较。
