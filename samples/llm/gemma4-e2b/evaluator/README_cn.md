# 精度验证

**简体中文** | [English](./README.md)

用于验证量化精度和板端 tensor 对齐的工具与流程。

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
BOARD_IP=<板端IP> TARGET_SOC="$TARGET_SOC" \
  bash scripts/verify/run_remote_hbm_verify.sh
```

## 板端（golden mask / KV 对齐）

`golden_mask_kv/` 为可选的内部校验数据，不包含在公开模型服务器中。

需编译 **全部** runtime 目标（不只 `main`）：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

然后运行 golden 校验：

```bash
export GEMMA4_HOME=~/gemma4_e2b
cd runtime/cpp/build
./gemma4_golden_verify --prompt prompt_0
```

预期输出：`ALL PASSED`（input_ids、mask、inputs_embeds 全部对齐）。

## VLM 冒烟测试

```bash
cd runtime/cpp/build
export GEMMA4_HOME=~/gemma4_e2b
./main
# /image ../../test_data/image1.jpg
# 你看到什么？
```

> **说明：** 主对话入口已按 Model Zoo 规范改名为 `main`，`./gemma4_chat` 不再存在。

预期结果见 [QUANTIZATION_TUTORIAL_zh.md §9.4](../conversion/QUANTIZATION_TUTORIAL_zh.md)。
