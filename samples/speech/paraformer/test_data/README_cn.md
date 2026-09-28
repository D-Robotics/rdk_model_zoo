# Paraformer 输入音频

[English](README.md) · [前端说明](../runtime/python/README_cn.md)

两条单声道 16 kHz PCM16 WAV 与 `manifest.json` 逐字节保留 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`，用于小规模源实现对照，
不是数据集精度基准，也不代表新增板测已通过。

| 语音 ID | 样本数 | 参考文本 | 前端有效帧数 |
| --- | --- | --- | --- |
| BAC009S0724W0121 | 68496 | 广州市房地产中介协会分析 | 71 |
| BAC009S0724W0168 | 75137 | 新地王的诞生迅速搅热南沙土地市场 | 78 |

参考文本来自源标注，不是本次迁移的模型预测。有效帧数由运行说明中固定配置的
真实 FunASR 前端测得。两条输入均生成 float32 `[1,400,560]` 特征，补齐位置为零，
没有截断；在记录的主机环境下与源输出逐字节相同。本结果没有执行 HBM 模型。

## 目录与自定义输入

`manifest.json` 是包含 `utt_id` 和参考 `text` 的 JSON 对象列表，对应音频为
`audio/<utt_id>.wav`。ID 应唯一，应作为名称而非路径处理。参考文本与预测分开保存，
生成特征时不能改写此源清单。完整清单 CLI 和 C++ 特征准备桥接仍在迁移，
归档源 `run.sh` 的说明不能当作已存在的统一命令。

当前前端 API 接收从 16 kHz 单／多声道音频加载的有限 float32 样本，多声道取均值，
不重采样。LFR 超过 400 帧时仅保留前 400 帧并返回 `truncated=True`，这不是长音频
完整识别。依赖和可直接无板运行的完整特征示例见运行说明。

## 复现验证

激活文档规定的前端环境，从仓库根目录运行：

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-frontend/verify_real.py
```

该脚本对照固定 Git 源核验复制的 WAV，在 7 组输入上比较真实 FunASR 输出，并检查
CPU 随机状态恢复；生成证据 JSON 并输出日志，不生成模型转写，也不修改此清单。
[记录结果](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-frontend-review.md)
包含源音频与特征摘要、原始／有效帧数以及截断状态。源分支板测仍属于历史记录，
新增 SDK／板端推理、数据集 CER 与计时均未执行。
