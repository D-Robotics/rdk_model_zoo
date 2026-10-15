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


21 个 `obs_history/*.bin` 文件和 `runtime-input-manifest.json` 为离线示例观测。
每个文件存 270 个小端 float32
数值，共 1080 字节，无文件头；当前 45 维观测在前，之后为五帧历史。

清单记录各文件对应 `rollout_evaluation.pt`（1504 个样本）中的源索引 0–20；
原始 rollout 未随本目录提供。
这些文件仅用作离线 runtime 输入；标定请使用代表性观测数据。

Python CLI 按文件名中的数字排序，拒绝重复索引，并按旁边的清单校验每个输入。
在 X5 上运行单文件可指定
`--input-path samples/robotics/himloco/test_data/obs_history/000000.bin`，完整命令见
[运行时说明](../runtime/python/README_cn.md#usage)。

按策略训练时的约定准备六帧观测历史。Runtime 返回十二维原始动作；控制器缩放方式见运行说明。
