# 精度验证

**简体中文** | [English](./README.md)

用于验证量化精度和板端 tensor 对齐的工具与流程。

<a id="dataset"></a>
## 数据与输入

PC 对照使用转换教程中的 COCO 图像、文本验证 prompt、原始浮点权重及同目标 BC。
板端 golden 校验使用 `$GEMMA4_HOME/golden_mask_kv/<prompt_id>/prefill_chunk_0/` 下五份文件：
`input_ids.int64.bin`、`position_ids.int32.bin`、`inputs_embeds.f32.bin`、`full_mask.f32.bin`、`sliding_mask.f32.bin`。
这套内部 golden 数据不随公开模型归档提供。四张示例图片用于冒烟测试的定性演示。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

PC 命令在已按 [转换说明](../conversion/README_cn.md) 准备的 OE-LLM conda 环境执行；板端命令需要
目标匹配的 HBM、embedding、SDK 及全部原生可执行文件。下面每个带 `cd samples/...` 的代码块均从仓库根目录开始。

<a id="command"></a>
## PC 端（BC / 浮点对比）

在 OE-LLM conda 环境中：

```bash
cd samples/llm/gemma4-e2b/conversion
conda activate oellm
export TARGET_SOC=s600  # 需要时替换为 s100 或 s100p

# Vision BC cosine similarity
python -m leap_llm.apis.verifier_cli \
    --model_name gemma4-e2b-vision \
    --model_dir ./gemma4-e2b \
    --quant_vlm_model_path ./output/gemma4_e2b_vision_${TARGET_SOC}/gemma4-e2b_vit_ptq.bc \
    --input_image_path ./calibration_data/images/coco_00_000000000802.jpg

# Text BC 快速验证
python -u scripts/verify/quick_text_verify.py --target-soc "$TARGET_SOC"
# 输出：output/e2b_text_verify_quick_${TARGET_SOC}.json
```

脚本位于 [conversion/scripts/verify/](../conversion/scripts/verify/)。
如需在板端进行 HBM 对比，必须显式提供板端地址：

```bash
BOARD_IP=<board-ip> TARGET_SOC="$TARGET_SOC" \
  bash scripts/verify/run_remote_hbm_verify.sh
```

## 板端（golden mask / KV 对齐）

`golden_mask_kv/` 为可选的内部校验数据，不包含在公开模型服务器中。

需编译 **全部** runtime 目标（不只 `main`）：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 --build
```

然后运行 golden 校验：

```bash
export GEMMA4_HOME=~/gemma4_e2b
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 golden_verify --prompt_id prompt_0
```

预期输出：`ALL PASSED`（input_ids、mask、inputs_embeds 全部对齐）。

## VLM 冒烟测试

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
export GEMMA4_HOME=~/gemma4_e2b
./run.sh --target s600
# /image ../../test_data/image1.jpg
# 你看到什么？
```

预期结果见 [QUANTIZATION_TUTORIAL_zh.md §9.4](../conversion/QUANTIZATION_TUTORIAL_zh.md)。

<a id="metrics"></a>
## 指标口径

PC Text 快检记录每条 prompt 的 logits cosine 和 `mean_cosine`。
Golden 校验器检查 prefill 构造：`input_ids`/`position_ids` 整数全等，
`inputs_embeds` 最大绝对误差 ≤1e-3，`full_mask`/`sliding_mask` 最大误差为 0。
使用 PC BC 对比在自己的 prompt 集上测量数据集准确率。

<a id="outputs"></a>
## 输出与判读

PC Text 结果为 `conversion/output/e2b_text_verify_quick_<target>.json`，含 `results` 与 `mean_cosine`。
Golden 结果逐项打印 OK/FAIL、误差及最终 `ALL PASSED`/`SOME FAILED`；全部通过返回 0，不匹配或异常返回 1。
运行 golden 校验前请准备五份输入张量。交互示例将回答流式写到终端。

<a id="reference-results"></a>
## 参考测量

源 README 与完整教程保留 S100P 演示、约 6.9 tok/s 文本截图和 S600 源回归说明。
这些数值与 golden 文档中的预期输出均来自 S 源发布的记录。
板端对比按上述运行时命令进行。

<a id="boundaries"></a>
## 适用范围

使用 PC BC 对比在自己的 prompt 集上测量数据集准确率，并用板端 golden 命令对比目标端张量构造。
