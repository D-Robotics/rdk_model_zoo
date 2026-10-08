# Evaluator

[简体中文](./README_cn.md) | **English**

Tools and workflows to verify quantized accuracy and on-board tensor alignment.

<a id="dataset"></a>
## Data and inputs

PC comparisons use the conversion tutorial's COCO image, text verification prompts, original float weights and matching-target BC.
Board golden verification needs five files under `$GEMMA4_HOME/golden_mask_kv/<prompt_id>/prefill_chunk_0/`:
`input_ids.int64.bin`, `position_ids.int32.bin`, `inputs_embeds.f32.bin`, `full_mask.f32.bin` and `sliding_mask.f32.bin`.
This internal golden dataset is not included in the public model archive. The four demo images are qualitative examples for the smoke test.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Run PC commands in the OE-LLM conda environment prepared by the [conversion guide](../conversion/README.md).
Board commands require matching-target HBMs, embeddings, SDK and all native executables.
Each code block containing `cd samples/...` below starts from the repository root.

<a id="command"></a>
## PC-side (BC / float comparison)

From your OE-LLM conda environment:

```bash
cd samples/llm/gemma4-e2b/conversion
conda activate oellm
export TARGET_SOC=s600  # replace with s100 or s100p when needed

# Vision BC cosine similarity
python -m leap_llm.apis.verifier_cli \
    --model_name gemma4-e2b-vision \
    --model_dir ./gemma4-e2b \
    --quant_vlm_model_path ./output/gemma4_e2b_vision_${TARGET_SOC}/gemma4-e2b_vit_ptq.bc \
    --input_image_path ./calibration_data/images/coco_00_000000000802.jpg

# Text BC quick check
python -u scripts/verify/quick_text_verify.py --target-soc "$TARGET_SOC"
# Output: output/e2b_text_verify_quick_${TARGET_SOC}.json
```

Scripts live under [conversion/scripts/verify/](../conversion/scripts/verify/).
For on-board HBM comparison, provide the board address explicitly:

```bash
BOARD_IP=<board-ip> TARGET_SOC="$TARGET_SOC" \
  bash scripts/verify/run_remote_hbm_verify.sh
```

## Board-side (golden mask / KV)

`golden_mask_kv/` is optional internal verification data and is not included in
the public model archive.

Build **all** runtime targets (not only `main`):

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 --build
```

Then run the golden verifier:

```bash
export GEMMA4_HOME=~/gemma4_e2b
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 golden_verify --prompt_id prompt_0
```

Expected: `ALL PASSED` for input_ids, masks, and inputs_embeds.

## VLM smoke test

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
export GEMMA4_HOME=~/gemma4_e2b
./run.sh --target s600
# /image ../../test_data/image1.jpg
# What do you see?
```

See [QUANTIZATION_TUTORIAL.md §9.4](../conversion/QUANTIZATION_TUTORIAL.md) for expected output.

<a id="metrics"></a>
## Metric definitions

The PC text quick check records per-prompt logits cosine and `mean_cosine`.
The golden verifier checks prefill construction: exact integer `input_ids` and
`position_ids`, `inputs_embeds` maximum absolute error ≤1e-3, and zero maximum
error for `full_mask` and `sliding_mask`. Use the PC BC comparison over your
prompt set to measure dataset-level accuracy.

<a id="outputs"></a>
## Outputs and interpretation

PC text results go to `conversion/output/e2b_text_verify_quick_<target>.json`, containing `results` and `mean_cosine`.
Golden verification prints per-input OK/FAIL, errors and final `ALL PASSED`/`SOME FAILED`; success returns 0, mismatches or exceptions return 1.
Prepare all five golden tensors before running the golden verifier. Interactive examples stream answers to the terminal.

<a id="reference-results"></a>
## Reference measurements

The source README and full tutorial retain the S100P demonstrations, approximately 6.9 tok/s text screenshot and S600 source regression notes.
They are records from the S source release; the documented golden expected output likewise comes from that record.
Board comparisons run through the runtime commands above.

<a id="boundaries"></a>
## Boundaries

Run the PC BC comparison over your prompt set for dataset-level accuracy, and
use the board golden command to compare target-side tensor construction.
