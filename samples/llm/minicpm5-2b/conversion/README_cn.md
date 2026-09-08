[English](README.md) | [简体中文](README_cn.md)

# S600 模型转换

以解压后的 OpenExplorer LLM 2.0.0-beta1 为准，先读包内 docs/01_环境搭建说明.md 和 docs/02_SkillShare初始化与使用.md。软件包根目录必须同时包含 llm_compression/ 与 package/；缺少 build_env.sh 或 compiler、march、hbm_infer、profiler、PyTorch plugin 五类 host whl 时停止。环境命令始终从软件包根目录执行，不将 llm_compression 本身加入 PYTHONPATH。已有可用 Python3.10 环境可复用，无须重新创建。

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

按包内环境文档第7节进行完整验证，覆盖 torch 实际运算、plugins、compiler、hbm-infer 和 CUDA 实际计算，不能只检查 import。若有 .skillshare，按包内说明安装CLI（本次0.20.21），补充尚不存在的 claude/cursor/opencode target，再执行 skillshare sync 和 skillshare status；量化优先使用包内 drobotics skills。保持 deps_version.conf 不变，切换镜像不能改变固定依赖版本。

已验证版本：torch2.8.0+cu128，horizon-plugin-pytorch3.3.5+cu128.torch280，profiler3.3.5，HBDK4 compiler/march4.11.7a2.dev202607040356+dcaae33.develop，hbm-infer3.15.3。校准使用CUDA；编译是CPU任务，入口会在编译阶段隐藏CUDA。编译中间文件可能超过数十GB，应在SDK安装空间之外预留足够磁盘。

## 复现步骤

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

TRAIN 和 TEST 分开下载并固定校验和。TRAIN 仅因SDK按 test-*.parquet 搜索而改名 test-train-source.parquet，内容仍来自TRAIN。校准使用50段256 token，完整评估使用140段2048 token。假量化配置显式加载 calibration/lm_calibration.pth.tar，入口会拒绝缺失checkpoint，不允许回退为浮点评估；未设置评估步数截断。

适配器验证无bias的Llama配置，恢复Transformers配置归一化后的RoPE theta=5000000，并通过官方chat template一次完成分词。编译为Nash-p四核、opt2、jobs8、256-token prefill、4096 KV容量、FP16 embedding，并保留全部prefill logits。prepare_tokenizer固定非thinking模式并检查token一致性；package_s600将模板加入HBM元数据，不改动原始checkpoint。重新编译可能产生不同二进制摘要，必须重新评估；公开性能和精度数据只对应本次下载包。

继续阅读[板端评估](../evaluator/README_cn.md)。
