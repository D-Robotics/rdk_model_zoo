[English](README.md) | 简体中文

# HIMLoco 离线观测



## 目录结构

```text
test_data/
├── obs_history/  # obs_history 相关文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── runtime-input-manifest.json  # 结构化数据
```


21 个 `obs_history/*.bin` 文件和 `runtime-input-manifest.json` 逐字节保留自 X5
提交 `ac115717197920355fc390bb04299b20e6436864`。每个文件存 270 个小端 float32
数值，共 1080 字节，无文件头；当前 45 维观测在前，之后为五帧历史。

清单记录它们是 `rollout_evaluation.pt`（1504 个样本）的源索引 0–20，源摘要为
`49f5459a5ff4d8003d9ee9d95c1104d158688017408a74bcd1506ff171cc01ab`，并附逐文件摘要。
原始 rollout 没有随本目录提供，其来源继承源记录。
这些是留出的离线 runtime 输入，不是代表性校准集。

Python CLI 按文件名中的数字排序，拒绝重复索引，并根据旁边清单检查每个输入。
摘要来自实际用于推理的同一份字节。在 X5 上执行单文件可指定
`--input-path samples/robotics/himloco/test_data/obs_history/000000.bin`，完整命令见
[运行时说明](../runtime/python/README_cn.md#usage)。

按策略训练时的约定准备六帧观测历史。Runtime 返回十二维原始动作；控制器缩放方式见运行说明。
