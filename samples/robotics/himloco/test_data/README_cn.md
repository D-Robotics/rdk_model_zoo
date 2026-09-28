# HIMLoco 离线观测

[English](README.md)

21 个 `obs_history/*.bin` 文件和 `runtime-input-manifest.json` 逐字节保留自 X5
提交 `ac115717197920355fc390bb04299b20e6436864`。每个文件存 270 个小端 float32
数值，共 1080 字节，无文件头；当前 45 维观测在前，之后为五帧历史。

清单记录它们是 `rollout_evaluation.pt`（1504 个样本）的源索引 0–20，源摘要为
`49f5459a5ff4d8003d9ee9d95c1104d158688017408a74bcd1506ff171cc01ab`，并附逐文件摘要。
原始 rollout 没有随本目录提供，其来源继承源记录，本轮没有重新生成或独立认证。
这些是留出的离线 runtime 输入，不是代表性校准集。

Python CLI 按文件名中的数字排序，拒绝重复索引，并根据旁边清单检查每个输入。
摘要来自实际用于推理的同一份字节。在 X5 上执行单文件可指定
`--input-path samples/robotics/himloco/test_data/obs_history/000000.bin`，完整命令见
[运行时说明](../runtime/python/README_cn.md#usage)。

这些输入不附带参考动作或机器人运动结论。主机测试中的模拟 SDK 输出明确是夹具，
不是模型预测；数值一致本身也不能证明观测构造正确或闭环行为有效。
