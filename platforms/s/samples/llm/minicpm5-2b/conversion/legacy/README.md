[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P quantization and compilation

Use D-Robotics LLM S100 **1.0.0 SDK**, Python 3.10 and leap_llm. This is separate from the parent S600 OELLM 2.0/lightcompress workflow. Obtain the [SDK](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_SDK.tar.gz) and [manual](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_Doc.zip). Install requirements/compiler/leap_llm according to the bundled manual; do not edit version configuration or mix 2.0 wheels.

## Adapter

Source: [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), revision `0e9c66dce9fedde5ba8663bbcdd54b6810bb929a`. Keep its original config, tokenizer and safetensors. It is a bias-free Llama model with 42 layers, hidden size 2048, 16 Q heads, 2 KV heads, head dimension 128, RoPE theta 5000000 and vocabulary 130560.

`legacy_adapter.py` reuses the SDK DeepSeek Llama-compatible implementation, removes Q/K/V biases, strictly loads all original weights and enables preserve_precision/W8. It does not convert the architecture into DeepSeek. `main.py` registers an external adapter with the SDK calibration/export/compile CLI without modifying the SDK. The verified shape is chunk256/cache4096; other shapes require compilation and validation.

## Reproduce

The build host used Ubuntu 22.04 and 192 GiB RAM. Compilation consumes CPU/RAM; GPU can accelerate calibration only in a compatible environment. The original PyTorch 2.6/cu124 does not support the tested RTX 5090, so the commands use CPU. Do not arbitrarily upgrade pinned dependencies. Allow over 100 GB storage for weights and two targets' BC/HBO/HBM intermediates; actual peaks depend on concurrency.

```bash
# In the SDK 1.0.0 Python 3.10 environment; from conversion/legacy
export MODEL_DIR=/path/to/MiniCPM5-2B
export DATA_ROOT="$PWD/datasets"
bash ../download_data.sh
python prepare_calibration.py   --train "$DATA_ROOT/wikitext2-calibration-train/test-train-source.parquet"   --model-dir "$MODEL_DIR" --output ./calibration-train.json
python prepare_tokenizer.py "$MODEL_DIR" ./deployment-tokenizer
python main.py --model_name minicpm5-2b --march nash-e   --input_model_path "$MODEL_DIR" --output_model_path ./output/s100   --calib_text_path ./calibration-train.json --device cpu   --w_bits 8 --chunk_size 256 --cache_len 4096
# Compile S100P independently from the same calibrated raw BC:
python retarget.py   --source-hbm ./output/s100/minicpm5-2b_chunk_256_cache_4096_q8.hbm   --output-hbm ./output/s100p/minicpm5-2b_chunk_256_cache_4096_q8.hbm   --march nash-m --jobs 8
```

Calibration verifies a pinned WikiText2 TRAIN SHA256, joins text with two newlines, encodes without extra special tokens and decodes the first 50 consecutive 256-token chunks into JSON text records. The filename `test-train-source.parquet` accommodates SDK file discovery; its contents are TRAIN. TEST is excluded. All 50 records were checked against the calibration used for the distributed artifacts.

`retarget.py` independently converts original `.prefill.bc` and `.decode.bc` for the selected march, removes quantization IO, compiles and links. Never feed Nash-e converted BC into Nash-m conversion. Sequential targets reduce peak memory and allow board testing while compiling the second target.

## Package and verify

Deploy only HBM, deployment tokenizer, LICENSE, modification notice and checksums. Exclude SDK, BC/HBO and caches. Public HBM names are `minicpm5-2b_ctx4096_s100.hbm` and `minicpm5-2b_ctx4096_s100p.hbm`. Tokenizer preparation changes only BPE serialization, the equivalent basic non-thinking template and primary chat EOS 130073; vocabulary and merge order are preserved.

See [models](../../model/README.md) and [basic board validation](../../evaluator/README.md). S600 PPL is not evidence for either legacy HBM; only single-turn English/Chinese and EOS are verified.
