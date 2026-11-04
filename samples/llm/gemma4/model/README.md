English | [简体中文](README_cn.md)

# Official model preparation

[Sample](../README.md)

<a id="artifacts"></a>
## Official artifacts

| Profile | ZIP | Bytes |
| --- | --- | ---: |
| `e2b-w4` | [Gemma4-E2B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip) | 4,402,995,725 |
| `e2b-w8` | [Gemma4-E2B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w8.zip) | 6,547,749,326 |
| `e4b-w4` | [Gemma4-E4B_w4.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w4.zip) | 6,320,670,645 |
| `e4b-w8` | [Gemma4-E4B_w8.zip](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E4B_w8.zip) | 9,762,116,686 |

<a id="directory"></a>
## Directory

`artifacts.json` indexes `manifests/<profile>.json`; `prepare.py` verifies files and writes deployment configuration.

<a id="preparation"></a>
## Prepare models

Each ZIP was fully downloaded, extracted and checked with CRC and SHA256. Downloads use the unchanged vendor ZIP at OSS.
Follow the [sample quick start](../README.md#quickstart). Each profile owns its manifest, HBM, embeddings and tokenizer bindings.

`prepare.py --profile <profile> --source-dir <directory> --output-dir <directory>`
verifies every file and writes absolute paths to `oellm_config.json`, plus a model manifest.
Use the extracted directory as both source and output to avoid copying large files.
<a id="accompanying-files"></a>
## Accompanying files

The ZIP contains language/vision/audio HBMs, ordinary and PLE embeddings, tokenizer files and the chat template.
All official vision/audio companions are retained; the sample request interface accepts text.
Configuration comes from the SDK's `Gemma4_config`; Runtime libraries are obtained separately.

<a id="local-paths"></a>
## Deployment paths and checks

For E2B W4, use `/data/gemma4-official/Gemma4-E2B_w4`; the generated configuration is `oellm_config.json`.

```bash
python3 samples/llm/gemma4/model/prepare.py --profile e2b-w4 \
  --output-dir /data/gemma4-official/Gemma4-E2B_w4 --check
```

<a id="formats-checksums"></a>
## Formats and checksums

Each manifest records the ZIP and eight original member SHA256 digests, sizes, HBM compiler and target metadata.
`artifacts.json` indexes four profile manifests. File verification does not execute inference, install dependencies or access the network.
The vendor ZIP supplies no source README, full HF revision, license file or Runtime configuration.
The manifest independently preserves vendor release provenance and the [model license](https://ai.google.dev/gemma/docs/gemma_4_license).
