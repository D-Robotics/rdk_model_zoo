# Evaluator

[简体中文](./README_cn.md) | **English**

Tools and workflows to verify quantized accuracy and on-board tensor alignment.

## PC-side (BC / float comparison)

From your OE-LLM conda environment:

```bash
# Vision BC cosine similarity
python -u leap_llm/apis/verifier_cli.py \
    --model_name gemma4-e2b-vision \
    --model_dir ./gemma4-e2b \
    --quant_vlm_model_path ./output/gemma4_e2b_vision/gemma4-e2b_vit_ptq.bc \
    --input_image_path ./calibration_data/images/coco_00_000000000802.jpg

# Text BC quick check
python -u conversion/scripts/verify/quick_text_verify.py
```

Scripts live under [conversion/scripts/verify/](../conversion/scripts/verify/).

## Board-side (golden mask / KV)

After building `runtime/cpp`:

```bash
export GEMMA4_HOME=~/gemma4_e2b
cd runtime/cpp/build
./gemma4_golden_verify --prompt prompt_0
```

Expected: `ALL PASSED` for input_ids, masks, and inputs_embeds.

## VLM smoke test

```bash
cd runtime/cpp/build
export GEMMA4_HOME=~/gemma4_e2b
./gemma4_chat
# /image ../../test_data/image1.jpg
# What do you see?
```

See [docs/QUANTIZATION_TUTORIAL.md §9.4](../docs/QUANTIZATION_TUTORIAL.md) for expected output.
