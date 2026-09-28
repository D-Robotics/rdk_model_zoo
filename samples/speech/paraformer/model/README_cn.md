# Paraformer 模型包

[English](README.md) · [Python 集成](../runtime/python/README_cn.md)

活动 [S 清单](../../../../docs/release/s/models.yaml) 发布一个包含六个文件的 S100
模型包。本目录提供显式准备入口，推理不自动下载。当前没有 X5、S100P 或 S600
对应的 Paraformer 发布组合。

| `s100/` 下的文件 | 用途 | 来源 |
| --- | --- | --- |
| `paraformer_large_encoder_400x560_s100.hbm` | encoder | 活动清单 URL（远端 `encoder_int16.hbm`） |
| `paraformer_large_predictor_400x512_s100.hbm` | predictor | 活动清单 URL（远端 `predictor_int16.hbm`） |
| `paraformer_large_decoder_400x512_s100.hbm` | decoder | 活动清单 URL（远端 `decoder_int16.hbm`） |
| `tokens.json` | 8,404 项有序词表 | 活动清单 URL |
| `am.mvn` | 前端 CMVN 统计量 | 仓库内固定 S 源文件 |
| `paraformer_config.yaml` | 源前端／模型配置 | 仓库内固定 S 源文件 |

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

脚本逐项输出实测摘要，六项均完成后才输出最终成功行。下载成功不等于 SDK 兼容
或推理通过。活动清单没有 HBM 的发布方哈希，实测摘要不能独立认证官方来源。
词表与本次迁移核定的固定包不同会被拒绝，但这个本地内容约束不等于发布方哈希。

## 固定来源与字节身份

`am.mvn`、`paraformer_config.yaml` 逐字节保留 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`。SHA-256 如下：

- `am.mvn`：`29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5`
- `paraformer_config.yaml`：`1d9057edeaba9e131cb98f26011606497cf3af187d8943525ddb5ee36c836b1b`
- 已下载 `tokens.json`：`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`

源前端使用 16 kHz、80 个 mel 频带、25 ms 窗长、10 ms 帧移，LFR 堆叠 7 帧、步长 6，
最多 400 帧，每帧 560 维。真实 CPU 前端已通过 7 组源实现对照，详见 Python 说明。流程中的全零 context bias
保留源部署方式，不提供用户自定义热词功能。三模型的具体接口见
[物理张量契约](../runtime/python/README_cn.md)。INT16 名称描述编译配方，不能据此猜测 I/O 类型。

## 当前验证范围

主机测试运行真实准备逻辑，只替换 HTTP 响应字节，检查六文件完整性、重跑复用、
已有文件保护及不写文件的预览。测试中的合成模型字节不是实际 HBM。发布词表已
真实下载，并验证 8,404 项内容不重复。本步骤未执行完整 HBM 包下载或真实 SDK 推理。
源辅助文件字节、预览和帮助命令的检查见
[绑定／模型包证据](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-binding-review.md)。
