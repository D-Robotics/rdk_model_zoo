# HIMLoco 模型包

[English](README.md)

<a id="artifacts"></a>
## 制品

| 文件 | 格式 | 目标 | 作用 |
| --- | --- | --- | --- |
| `bayes-e/himloco_go2_bayese_1x270.bin` | BIN | 仅 X5 | 融合 estimator 与 actor；obs_history float32 [1,270] → actions float32 [1,12] |

URL 与摘要以[活动 X5 清单](../../../../docs/release/x5/models.yaml)为准。
S100／S100P／S600 没有匹配发布组合。此模型已经融合，不能将单独 encoder 或 policy
改名后替代。

<a id="preparation"></a>
## 准备

从仓库根目录执行，需要 Python、NumPy 和 PyYAML：

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5 --dry-run
bash samples/robotics/himloco/model/download_model.sh --target x5
```

预览输出身份、URL、目标路径与预期摘要，不下载、不写文件。第二条命令显式准备模型。
已有文件只检查、不覆盖；新下载使用临时文件，核对发布摘要后原子安装。摘要不符返回 2，
不会留下新的、看似完整的 BIN。纠正输入／路径后重试，不通过改名替代模型。
可用 `PYTHON` 指定 shell 包装器的解释器。

`--target` 仅接受并默认使用 `x5`。`--output-dir` 默认当前 model 目录，始终在其中
创建 `bayes-e/` 子目录，不改变运行时默认值。准备已发布制品不需要工具链或板端 SDK。
本轮迁移没有实际下载模型；主机检查覆盖预览和共享准备边界，不代表板端推理。

<a id="accompanying-files"></a>
## 附属文件

不需要词表、类别标签或外部归一化文件。调用者须提供正确构造的六帧观测历史。
[内置测试输入](../test_data/README_cn.md)记录源索引和 SHA-256，既不是校准数据，
也不是实时机器人状态估计器。

<a id="local-paths"></a>
## 本地路径

运行时默认：`samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin`。
使用其他位置时，向 Python 入口同时传入 `--model-path /absolute/path/model.bin` 和
`--asset-id x5:himloco:himloco_go2_bayese_1x270.bin`。发布摘要与板型检查仍生效，
外部路径不表示接受另一份策略模型。

<a id="formats-checksums"></a>
## 格式与校验

发布 BIN SHA-256：
`7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a`。
来源是活动清单，继承自固定 X5 Sample。准备成功证明文件与该摘要一致，不证明 SDK
兼容、动作精度或闭环稳定性；运行时还会核验物理张量元数据。
