[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P 量化与编译

本目录针对 D-Robotics LLM S100 **1.0.0 SDK**（Python 3.10、leap_llm），与上层 S600 的 OELLM 2.0/lightcompress 流程独立。下载 [SDK](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_SDK.tar.gz) 和 [用户手册](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm_s100/1.0.0/D-Robotics_LLM_S100_1.0.0_Doc.zip)，严格按包内文档安装 requirements、编译器和 leap_llm，不能修改版本配置或混装 2.0 的 wheel。

## 模型与适配

原始 [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) revision `0e9c66dce9fedde5ba8663bbcdd54b6810bb929a`。保留原始 config、tokenizer 和 safetensors。模型为 bias-free Llama：42 层、hidden 2048、Q heads 16、KV heads 2、head_dim 128、RoPE theta 5000000、词表 130560。

`legacy_adapter.py` 复用 SDK DeepSeek 的 Llama 兼容实现，移除 Q/K/V bias 并 strict load 全部原始权重，启用 preserve_precision 和 W8。不是把模型变成 DeepSeek。`main.py` 注册外部模型，复用原 SDK 的校准、导出、编译入口，不改 SDK 源码。已测 chunk=256/cache=4096；更改配置需重新编译和验证。

## 复现

开发环境使用 Ubuntu 22.04、192 GiB 内存。编译主要消耗 CPU 和 RAM，GPU 仅用于兼容环境下的校准；原 SDK PyTorch 2.6/cu124 不支持本次 RTX 5090，下面明确使用 CPU。不要通过随意升级依赖修复兼容性。建议预留 100 GB 以上存储，包含原权重、两目标 BC/HBO/HBM 中间产物；具体峰值取决于并行度。

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

校准数据是已固定 SHA256 的 WikiText2 TRAIN，将文本用两个换行拼接，无额外 special tokens 编码，取前 50 个连续 256-token 片段并解码成 JSON text。文件名 `test-train-source.parquet` 是沿用 SDK 文件发现规则，内容仍是 TRAIN。TEST 不用于校准。该生成方式已与产物编译所用的 50 条记录逐项核对。

`retarget.py` 从原始 `.prefill.bc` / `.decode.bc` 分别针对新 march 转换、移除量化 IO、编译并链接；不能把 Nash-e 转换后的 BC 当作 Nash-m 输入。可先部署验证一块板，再使用这些原始 BC 编译另一块，以控制内存峰值。

## 打包与验证

运行文件仅需 HBM、`deployment-tokenizer/`、LICENSE、修改说明和校验清单；不分发 SDK、`.bc`、`.hbo` 或编译缓存。公开包将 HBM 重命名为 `minicpm5-2b_ctx4096_s100.hbm` / `minicpm5-2b_ctx4096_s100p.hbm`。tokenizer 准备仅转换 BPE merges 序列化、使用等价的纯文本非思考模板，并选择已有 chat EOS 130073，不改词表和合并顺序。

参见 [模型下载](../../model/README_cn.md) 与 [基础板端验证](../../evaluator/README_cn.md)。S600 的 PPL 不能视为这两个 HBM 的精度结果；目前只完成单轮中英文和 EOS 验证。
