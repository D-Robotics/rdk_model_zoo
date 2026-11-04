[English](README.md) | 简体中文

# 官方模型准备

[Sample](../README_cn.md)

<a id="artifacts"></a>
## 官方制品

| Profile | ZIP | Bytes |
| --- | --- | ---: |
| `e2b-w4` | [Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip) | 4,402,995,725 |
| `e2b-w8` | [Gemma4-E2B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w8.zip) | 6,547,749,326 |
| `e4b-w4` | [Gemma4-E4B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w4.zip) | 6,320,670,645 |
| `e4b-w8` | [Gemma4-E4B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w8.zip) | 9,762,116,686 |

<a id="directory"></a>
## 目录

`artifacts.json` 索引 `manifests/<profile>.json`；`prepare.py` 校验并生成部署配置。

<a id="preparation"></a>
## 准备模型

完整下载、解压、CRC 和 SHA256 已核对。原始模型 ZIP 保持官方字节，直接从 OSS 下载。
以 [Sample 快速体验](../README_cn.md#quickstart) 的命令准备 E2B W4。
每个配置使用自己的 `model/manifests/<profile>.json`，不能混用不同规格的 HBM、embedding 或 tokenizer。

`prepare.py --profile <profile> --source-dir <directory> --output-dir <directory>`
校验全部文件，生成绝对路径 `oellm_config.json` 和 `model-manifest.json`。
传入已解压目录作为相同的 source/output 可避免重复复制大文件。
<a id="accompanying-files"></a>
## 伴随文件

包中包含语言/视觉/音频 HBM、普通和 PLE embedding、tokenizer 及 chat template。
准备步骤包含官方视觉和音频模型；本 Sample 仅提供文本请求入口。
运行配置从同版本 SDK 的 `Gemma4_config` 取得，SDK Runtime 需单独准备。

<a id="local-paths"></a>
## 部署路径与检查

例如 E2B W4 位于 `/data/gemma4-official/Gemma4-E2B_w4`；生成的配置文件是 `oellm_config.json`。

```bash
python3 samples/llm/gemma4/model/prepare.py --profile e2b-w4 \
  --output-dir /data/gemma4-official/Gemma4-E2B_w4 --check
```

<a id="formats-checksums"></a>
## 格式与校验

每个 manifest 保存完整 ZIP 及八个原始文件的 SHA256、大小、HBM 编译器和目标信息。
`artifacts.json` 是四个配置的索引；文件校验不执行推理、不安装依赖、不访问网络。
官方 ZIP 没有附带来源 README、HF 完整 revision、许可证文件或 Runtime 配置。
模型来源、[许可证](https://ai.google.dev/gemma/docs/gemma_4_license) 和 SDK 发布线由独立清单保留。
