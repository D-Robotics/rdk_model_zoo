[English](README.md) | 简体中文

# Paraformer 模型包

[Python 集成](../runtime/python/README_cn.md)

活动 [S 清单](../../../../docs/release/s/models.yaml) 发布一个包含六个文件的 S100
模型包。本目录提供显式准备入口，推理不自动下载。当前没有 X5、S100P 或 S600
对应的 Paraformer 发布组合。

<a id="artifacts"></a>
## 发布制品

| `s100/` 下的文件 | 用途 | 来源 |
| --- | --- | --- |
| `paraformer_large_encoder_400x560_s100.hbm` | encoder | 活动清单 URL（远端 `encoder_int16.hbm`） |
| `paraformer_large_predictor_400x512_s100.hbm` | predictor | 活动清单 URL（远端 `predictor_int16.hbm`） |
| `paraformer_large_decoder_400x512_s100.hbm` | decoder | 活动清单 URL（远端 `decoder_int16.hbm`） |
| `tokens.json` | 8,404 项有序词表 | 活动清单 URL |
| `am.mvn` | 前端 CMVN 统计量 | 仓库内固定 S 源文件 |
| `paraformer_config.yaml` | 源前端／模型配置 | 仓库内固定 S 源文件 |

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download_model.sh  # Shell 脚本
└── paraformer_config.yaml  # 配置
```

<a id="preparation"></a>
## 预览与准备

需要 Python、NumPy 和 PyYAML。在主机准备文件不需要 SDK、板卡或 publisher 构建。
从仓库根目录预览全部来源与目标，不访问网络也不写文件：

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100 --dry-run
```

显式下载四个远端文件并复制两份本地前端文件：

```bash
bash samples/speech/paraformer/model/download_model.sh --target s100
```

默认输出根目录为当前 model 目录，六个文件都放入 `model/s100/`。
指定 `--output-dir /path/to/package` 后写入 `/path/to/package/s100/`。
`--target` 仅接受并默认使用 `s100`；`--dry-run` 输出六行，不创建任何目录或文件。
使用 shell 包装器时可以设置 `PYTHON` 为解释器路径，也可以直接用选定 Python
运行 `download.py`。`--help` 不需要 SDK 或模型文件。

已有文件不会被覆盖。远端文件使用共享下载器的临时文件与原子安装；本地文件在
复制前核对固定 SHA-256，并以不替换已有路径的方式安装。配置或词表不匹配时
退出码为 2，保留原文件。后续步骤失败可能留下前面已完整准备的文件；应先检查
错误并保留用户数据，纠正失败项后再执行。不能把部分准备结果视为可用模型包。

脚本逐项输出实测摘要，六项均完成后才输出最终成功行。活动清单没有 HBM 的
发布方哈希，实测摘要用于绑定所下载字节。词表与核定固定包不同会被拒绝。

<a id="accompanying-files"></a>
## 附属文件

`tokens.json` 是文本推理与 FP32/HMCT 评测的必需词表，仅有类别索引不能确定文本。
`am.mvn` 是音频前处理所需的 CMVN。仓库内 `paraformer_config.yaml` 记录源模型与
前端配置，准备模型包时核对；它不是额外的 HBM 推理模型。
C++ 入口读取 Python 前端生成的 NPY／清单，不重复执行音频前端或 CMVN。

<a id="local-paths"></a>
## 本地路径

三个 HBM 默认路径是 `samples/speech/paraformer/model/s100/` 下表中对应文件；
默认词表为 `samples/speech/paraformer/model/s100/tokens.json`。Python 音频准备
默认读取仓库内 `samples/speech/paraformer/model/am.mvn`，而非复制后的 `s100/am.mvn`，
两者使用同一源摘要检查。仓库内配置为 `samples/speech/paraformer/model/paraformer_config.yaml`。
下载时指定 `--output-dir` 不会修改运行时默认值：Python 需要同时提供三个外部模型
路径及匹配的 asset ID；C++ 启动器需要指定完整模型目录，具体见各运行时说明。

<a id="formats-checksums"></a>
## 固定来源与字节身份

`am.mvn`、`paraformer_config.yaml` 逐字节保留 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`。SHA-256 如下：

- `am.mvn`：`29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5`
- `paraformer_config.yaml`：`1d9057edeaba9e131cb98f26011606497cf3af187d8943525ddb5ee36c836b1b`
- 已下载 `tokens.json`：`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`

源前端使用 16 kHz、80 个 mel 频带、25 ms 窗长、10 ms 帧移，LFR 堆叠 7 帧、步长 6，
最多 400 帧，每帧 560 维。真实 CPU 前端的特征生成流程
详见 Python 说明。流程中的全零 context bias
保留源部署方式，不提供用户自定义热词功能。三模型的具体接口见
[物理张量契约](../runtime/python/README_cn.md)。INT16 名称描述编译配方，不能据此猜测 I/O 类型。

三个 `.hbm` 是 S100 编译制品，各自发布方 SHA-256 均为 `null (unknown)`。
`tokens.json` 是 UTF-8 JSON，`am.mvn` 是文本 CMVN，`paraformer_config.yaml` 是 YAML。
上述摘要用于核对本地附属文件。
