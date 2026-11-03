# 精度与性能记录

[English](README.md) · [Sample](../README_cn.md)

`host_reference.json` 保存此次 SDK 主机量化参考。WikiText2 test 全量 142×2048：

| 路径 | 浮点 PPL | fakequant PPL | 相对变化 |
| --- | ---: | ---: | ---: |
| LightCompress | 41.21096802 | 43.29276657 | +5.05% |
| HPP 部署路径 | 41.2612 | 43.0350 | +4.30% |

两条路径分别配对比较。HPP 达到本次 ≤5% 门槛，未达到 ≤3% 目标。
主机 SDK 生成算术结果 `4` 和翻译结果 `你好`。以上均不是编译后 HBM 的板端结果。

`board_result.json` 的板端 PPL、TTFT、decode token/s、峰值内存、生成验证和环境
全部为空，状态为 `not-run`。此次接入不执行板端测试。

未来板测可使用 SDK 的 `examples/ppl_demo`，将模型配置指向准备好的
`oellm_config.json`，并按该 SDK 的 README 准备语料和评测参数。正式比较前需对齐
tokenizer、chat template、前缀、分段与标签掩码；保留全部有效 token 的 NLL 和计数。
文本生成可按 `test_data/prompts.json` 检查。性能记录需包含输入/输出 token 数、
并发、缓存状态、runtime、系统和硬件频率，不从主机 CUDA 时间换算板端性能。
