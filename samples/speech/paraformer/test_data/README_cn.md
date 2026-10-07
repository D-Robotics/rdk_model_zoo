# Paraformer 输入音频

[English](README.md) · [前端说明](../runtime/python/README_cn.md)

两条单声道 16 kHz PCM16 WAV 与 `manifest.json` 逐字节保留 S 发布的
原始内容，作为特征生成与推理的小规模源对照输入。

| 语音 ID | 样本数 | 参考文本 | 前端有效帧数 |
| --- | --- | --- | --- |
| BAC009S0724W0121 | 68496 | 广州市房地产中介协会分析 | 71 |
| BAC009S0724W0168 | 75137 | 新地王的诞生迅速搅热南沙土地市场 | 78 |

参考文本来自源标注，不是模型预测。有效帧数由运行说明中固定配置的
真实 FunASR 前端测得。两条输入均生成 float32 `[1,400,560]` 特征，补齐位置为零，
没有截断。

## 目录与自定义输入

`manifest.json` 是包含 `utt_id` 和参考 `text` 的 JSON 对象列表，对应音频为
`audio/<utt_id>.wav`。ID 应唯一，应作为名称而非路径处理。参考文本与预测分开保存，
生成特征时不能改写此源清单。Python 清单 CLI 已提供独立清单与特征输出，见运行说明。
原生启动器读取 Python 准备的 `prepared-manifest.json` 和特征 NPY 而非音频，
流程见[原生说明](../runtime/cpp/README_cn.md#quickstart)。
S100 SDK 推理使用原生说明中的命名参数。

当前前端 API 接收从 16 kHz 单／多声道音频加载的有限 float32 样本，多声道取均值，
不重采样。LFR 超过 400 帧时仅保留前 400 帧并返回 `truncated=True`，这不是长音频
完整识别。依赖和可直接无板运行的完整特征示例见运行说明。

## 复现验证

激活文档规定的前端环境，从仓库根目录重新生成特征：

```bash
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

准备清单记录每份特征文件的 SHA-256、原始／有效帧数和截断状态；`feat_length`
分别为 71、78。该运行不改写此清单。SDK／板端推理、数据集 CER 与计时按
[运行说明](../runtime/python/README_cn.md)与[评测说明](../evaluator/README_cn.md)执行。
