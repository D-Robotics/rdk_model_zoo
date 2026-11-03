# 模型准备

[English](README.md) · [Sample](../README_cn.md)

`artifacts.json` 是本次 SDK 2.0 S600 文本部署文件的绑定清单。语言分支使用固定
revision 的 QAT-unquantized 原始权重，经 SDK AutoClip/GPTQ、HPP 校准及 HBDK
编译生成。文件名中的 `_w8_` 来自 SDK 通用导出命名；实际投影权重是 W4/W8
混合精度，以清单和转换配置为准。

## 下载 Archive 模型包

[S600 部署权重包](https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz)压缩后 4,764,575,933 字节，包含清单中的完整八个模型文件、配置模板、模型卡和许可证。
从仓库根目录执行下载、校验和路径配置：

```bash
curl -fL --retry 3 -o gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s600/gemma/gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz
printf '%s  %s\n' '619696547bb487c11d27a81e3bbdcf5358a93fbffefda239d6329900a57bd2d9' 'gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz' | sha256sum -c -
mkdir -p /data/gemma4-e2b-sdk2
tar -xzf gemma4-e2b-w4w8-s600-chunk512-ctx4096.tar.gz --strip-components=1 -C /data/gemma4-e2b-sdk2
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir /data/gemma4-e2b-sdk2 \
  --output-dir /data/gemma4-e2b-sdk2
```

## 包内结构

```text
gemma4-e2b-w4w8-s600-chunk512-ctx4096/
├── model_language_chunk_512_cache_4096_w8_nash-p_corenum_4_4.hbm
├── fake_quant_model_embed_tokens.bin
├── fake_quant_model_embed_tokens_per_layer.bin
├── gemma-4-E2B-it_vision_384x384_w8_nash-p_corenum_4.hbm
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── oellm_config.template.json
├── model-manifest.json
├── SHA256SUMS
├── SOURCE_MODEL_README.md
├── README.md
└── LICENSE
```

权重包共 13 个文件。`prepare.py` 按实际目录生成 `oellm_config.json`；官方 runtime 运行库和 Sample 源码单独提供。

## 准备已有制品

工具链运行目录：

```text
/opt/D-Robotics_LLM_2.0.0_S600_S100_SDK/adapt_runs/gemma4_e2b_s600_20261103_ede84003/
  artifacts/hpp/hbm/              语言 HBM、两张 embedding、tokenizer 与模板
  artifacts/runtime_companions/   SDK 官方视觉 HBM
```

在该服务器上可将两处文件整理为一个可搬运目录：

```bash
python3 samples/llm/gemma4/model/prepare.py \
  --source-dir "$RUN_DIR/artifacts/hpp/hbm" \
  --companion-dir "$RUN_DIR/artifacts/runtime_companions" \
  --output-dir /data/gemma4-e2b-sdk2
```

`RUN_DIR` 指向上述运行目录。通过已有文件传输方式将准备好的目录放到 S600。
若板端绝对路径改变，使用同一目录作为 `--source-dir` 和 `--output-dir` 再执行一次，
即可校验文件并更新配置中的绝对路径。

也可以在板端分别放置语言制品与官方视觉文件，再执行 Sample 快速开始中的准备命令。
准备操作按文件大小和 SHA256 检查，使用临时目录后整体安装，不覆盖已有不匹配文件。

## SDK 视觉配套文件

GEMMA4 runtime 在初始化阶段加载视觉模型。本 Sample 只提交文本请求，但仍须提供
SDK 2.0.0 官方模型包中的以下文件：

```text
gemma-4-E2B-it_vision_384x384_w8_nash-p_corenum_4.hbm
```

其来源是 SDK `model/resolve_model_s600.md` 登记的
[Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip)。
仅使用该包的视觉配套 HBM；语言 HBM、embedding、tokenizer 和模板仍使用本次转换
清单中的文件。视觉文件来源、包内路径和 SHA256 单独登记，不作为视觉精度验证。

完整八个文件合计 7,298,660,276 字节。准备后另生成 `oellm_config.json` 和清单副本。
Archive 压缩包内保留这些文件校验值，下载大小与解压后文件大小分别记录。

只校验已准备目录，不加载 runtime：

```bash
python3 samples/llm/gemma4/model/prepare.py --check --output-dir /data/gemma4-e2b-sdk2
```
