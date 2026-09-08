[English](README.md) | [简体中文](README_cn.md)

# S600 conversion

Use the extracted OpenExplorer LLM 2.0.0-beta1 package as the source of truth. Read `docs/01_环境搭建说明.md` and `docs/02_SkillShare初始化与使用.md` first. The SDK root must contain llm_compression/ and package/. Stop if build_env.sh or any of the five host wheels (compiler, march, hbm_infer, profiler, PyTorch plugin) are missing. Run environment commands from the SDK root; never add llm_compression itself to PYTHONPATH. Reuse an existing valid Python 3.10 environment rather than recreating it.

```bash
export OELLM_SDK_ROOT=/path/to/OpenExplorer_LLM
export SAMPLE=/path/to/rdk_model_zoo/samples/llm/minicpm5-2b
export WORK=/data/minicpm5
cd "$OELLM_SDK_ROOT"
ls llm_compression/build_env.sh package/host/*.whl
conda create -n oellm python=3.10 -y
conda activate oellm
python -c 'import sys; assert sys.version_info[:2] == (3,10)'
sha256sum llm_compression/deps_version.conf > /tmp/minicpm5-deps.sha256
bash llm_compression/build_env.sh
sha256sum -c /tmp/minicpm5-deps.sha256
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/llm_compression/lightcompress${PYTHONPATH:+:$PYTHONPATH}"
```

Perform the complete verification in section 7 of the bundled environment guide: torch operations, plugins, compiler, hbm-infer and actual CUDA computation, not imports alone. If .skillshare exists, install the documented CLI (tested v0.20.21), add claude/cursor/opencode targets when absent, then run `skillshare sync && skillshare status`. Use the bundled drobotics skills for quantization. Keep deps_version.conf unchanged; mirror selection must not change pinned package versions.

Verified host packages: torch2.8.0+cu128, horizon-plugin-pytorch3.3.5+cu128.torch280, profiler3.3.5, HBDK4 compiler/march4.11.7a2.dev202607040356+dcaae33.develop, hbm-infer3.15.3. Calibration uses CUDA; compilation is CPU work and the launcher hides CUDA for that stage. Allow substantial intermediate disk space beyond the SDK installation; compilation intermediates can exceed tens of GB.

## Reproduce

```bash
mkdir -p "$WORK"
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM5-2B', revision='0e9c66dce9fedde5ba8663bbcdd54b6810bb929a', local_dir='$WORK/MiniCPM5-2B')"
DATA_ROOT="$WORK/datasets" bash "$SAMPLE/conversion/download_data.sh"
python "$SAMPLE/conversion/create_config.py" --model-path "$WORK/MiniCPM5-2B"   --data-root "$WORK/datasets" --output-dir "$WORK/output"
python "$SAMPLE/conversion/main.py" torch_eval --config_path "$WORK/output/s600.yaml"
python "$SAMPLE/conversion/main.py" calib --config_path "$WORK/output/s600.yaml"
python "$SAMPLE/conversion/main.py" torch_eval --config_path "$WORK/output/s600.fake-quant.yaml"
python "$SAMPLE/conversion/main.py" compile --config_path "$WORK/output/s600.yaml"
python "$SAMPLE/conversion/prepare_tokenizer.py" "$WORK/MiniCPM5-2B" "$WORK/tokenizer"
python "$SAMPLE/conversion/package_s600.py"   "$WORK/output/hbm/MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm"   "$WORK/tokenizer" "$WORK/deployment"
```

TRAIN and TEST are downloaded separately with fixed hashes. The TRAIN parquet is renamed test-train-source.parquet solely because the SDK discovers test-*.parquet; its content is still TRAIN. Calibration uses 50 segments of 256 tokens; complete evaluation uses 140 segments of 2048. The fake-quant configuration explicitly loads calibration/lm_calibration.pth.tar and the launcher refuses missing checkpoints. No evaluation-step truncation is set.

The adapter validates bias-free Llama, restores RoPE theta=5000000 after Transformers config normalization, and tokenizes the official chat template once. Compilation uses Nash-p, four cores, opt2, jobs8, 256-token prefill, 4096 KV capacity, FP16 embedding and all prefill logits. Prepare_tokenizer fixes non-thinking mode and checks token equality; package_s600 adds that template to HBM metadata without changing the source checkpoint. A new compilation may have a different binary digest and must be evaluated before deployment; published metrics apply to the provided archive.

See [board evaluation](../evaluator/README.md).
