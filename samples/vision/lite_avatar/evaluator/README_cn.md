[English](README.md) | 简体中文

# LiteAvatar 评估

<a id="dataset"></a>
## 数据

LiteAvatar 是生成式流水线而非分类器：质量通过把渲染输出与参考渲染——已
保存的金标渲染，或同一渲染器在完全相同输入下的另一次运行——逐帧对比来
衡量。对比序列来自同一输入 WAV（如捆绑的 `asr_example.wav`，30 fps 共
166 帧）与同一数字人数据目录，因此任何差异都能把被比较的运行时与参考
区分开。不存在标注数据集；参考渲染即基准。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md           # 英文说明
├── README_cn.md        # 中文说明
└── evaluate_frames.py  # 带门槛的逐帧 PSNR/MAE 对比
```

<a id="environment"></a>
## 环境

主机侧对比需要示例的宿主机依赖（[requirements-host.txt](../requirements-host.txt)）：
帧解码与运算使用 NumPy 和 OpenCV；无需板端 SDK。帧可以来自 `.png`/`.jpg`
目录，参考也接受经 OpenCV 解码的视频文件。

<a id="command"></a>
## 命令

在仓库根目录把渲染帧目录与参考对比（成功：退出码 0；门槛未达：退出码 1；
输入错误：退出码 2）：

```bash
# 输入：渲染帧 + 参考帧（或一个参考视频）
# 输出：evaluation.json 及标准输出上的运行汇总
python3 samples/vision/lite_avatar/evaluator/evaluate_frames.py \
    --actual /path/to/rendered_frames \
    --reference /path/to/reference_frames \
    --gate 40.0 \
    --output /path/to/avatar-evaluation/evaluation.json
```

`--actual` 与 `--reference` 接受 `.png`/`.jpg` 帧目录（按排序顺序配对），
`--reference` 也接受视频文件。两侧以锁步方式流式读取——只保留逐帧指标，
从不同时持有两条完整序列——并在每条退出路径上释放流。

帧数语义：默认全量对比在帧数不一致时**直接失败**（退出码 1）——截断的
交付不可能通过。`--limit N` 选择明确标注的前缀检查，只比较各自的前
N 帧，且**要求两侧都至少有 N 帧**（任一侧更短即失败；检查绝不静默地
比较少于标注数量的帧）。`--limit` 必须为正、`--gate` 必须有限，否则为
用法错误（退出码 2）。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 逐帧 PSNR | 对 uint8 像素值计算 `10*log10(255^2 / MSE)`，逐帧给出 | 几何一致，两序列均为 uint8 BGR |
| 完全一致帧的 PSNR | `null`（不存在有限 PSNR），计入 `perfect_frames` | 逐比特相同的帧 |
| 平均 / 最小 PSNR | 仅对存在真实 MSE 的帧取均值 / 最小值（全部一致时为 `null`；全一致的运行通过任意有限门槛） | 聚合用原始值计算，仅在输出时舍入 |
| 逐帧 / 平均 MAE | uint8 像素值上的平均绝对差（0–255 尺度；`0` 为完全相同，`255` 为最大差异） | 条件同上 |
| 帧数检查 | 默认全量对比要求两侧帧数相等（不一致 → FAIL，退出码 1）；`--limit N` 要求每侧 ≥ N 帧 | 属于结论的一部分，不是独立错误 |

<a id="outputs"></a>
## 输出

命令打印运行汇总（比较帧数、完全一致帧数、平均 PSNR、平均 MAE、帧数
相等结论），并写入 JSON 记录：每帧的 PSNR 与 MAE、汇总及对比路径（完全
一致帧的 `psnr_db` 为 `null`——绝不是截断值）。退出码 0 为通过、1 为
对比或门槛失败（帧数不一致、前缀短于 `--limit`、门槛未达）、2 为用法 /
输入错误（缺失或无法解码的帧会给出确切文件名、几何不一致、非有限
`--gate`）。

<a id="reference-results"></a>
## 参考结果

本脚本没有固定的参考数值：结果完全取决于所选的参考渲染、编码器与产物集，
每次对比都会生成自己的基线 JSON 记录。自行编译产物集的产物级验证数值
（如解码器对比 FP32 ONNX 的张量检查，它在解码器归一化 float 输出尺度上
计算——既非本脚本产生，也不在其 uint8 帧尺度上）记录在
[转换文档](../conversion/README_cn.md#validation)。

<a id="boundaries"></a>
## 评估边界

该指标只比较渲染像素，不评判口型同步质量、时序平滑度或音画对齐——这些请
直接观看输出视频；数值门槛是示例运行时复现参考渲染的回归检查。与金标参考
对比时，请保持产物集、数字人数据、音频与编码器固定：差异即可归因于被比较
的运行时而非输入。端到端视频检查（可解码性、几何、时长、混音）属于示例的
上板验证，不在本脚本范围内。
