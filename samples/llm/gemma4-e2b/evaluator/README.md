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

`golden_mask_kv/` is optional internal verification data and is not included in
the public model archive.

Build **all** runtime targets (not only `main`):

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

Then run the golden verifier:

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
./main
# /image ../../test_data/image1.jpg
# What do you see?
```

> **Note:** The primary chat entry was renamed to `main` (Model Zoo convention). `./gemma4_chat` no longer exists.

See [QUANTIZATION_TUTORIAL.md §9.4](../conversion/QUANTIZATION_TUTORIAL.md) for expected output.
