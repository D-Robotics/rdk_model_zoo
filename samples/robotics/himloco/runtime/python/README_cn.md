[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Python Runtime

本示例加载融合 RDK X5 模型，在 BPU 上运行带源行号的离线策略输入，并写出 float32
action 和 JSON 证据报告。

## 环境依赖

- RDK X5，RDK OS 3.5.0 或更高版本；
- BSP 随附且与板端 `libdnn` 匹配的 `hbm_runtime`；
- Python 3.10 或更高版本和 NumPy。

所需 Runtime 通常由 BSP 提供，可使用以下命令检查：

```bash
cat /etc/version
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
```

不要安装 PyPI 上无关的同名 `hbm_runtime` 包。

## 目录结构

```text
python/
├── himloco.py       # HimLocoConfig 与可复用 Model Zoo 推理流水线
├── main.py          # 带源行号的命令行推理与报告生成
├── run.sh           # 模型下载和一键运行
├── README.md        # 英文说明
└── README_cn.md     # 中文说明
```

生成文件写入被忽略的 `runs/` 目录。

## 模型合同

| Tensor | 名称 | 类型 | 逻辑 shape |
| --- | --- | --- | --- |
| 输入 | `obs_history` | float32 | `[1,270]` |
| 输出 | `actions` | float32 | `[1,12]` |

batch 为 1 且元素数一致时可接受等价 Runtime 布局。名称、类型、shape、NaN/Inf 或
非 `.bin` 模型不符合合同时会在推理前报错。

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | RDK X5 Bayes-e 模型 | `../../model/bayes-e/himloco_go2_bayese_1x270.bin` |
| `--input-path` | 单个输入或输入目录 | `../../test_data/obs_history` |
| `--output-dir` | 新 action dump 目录 | `runs/<timestamp>/action_dumps` |
| `--report` | 新 JSON 证据报告 | `runs/<timestamp>/python-report.json` |
| `--warmup` | 不计时的 Runtime 调用次数 | `10` |
| `--priority` | 可选 DNN 优先级 [0,255] | Runtime 默认 |
| `--bpu-cores` | 可选 BPU 核编号 | Runtime 默认 |

## 快速运行

在 HIMLoco 示例目录执行以下命令；默认模型缺失时会自动下载，并运行随附的 21 个输入：

```bash
bash runtime/python/run.sh
```

使用外部数据时可显式指定参数：

```bash
bash runtime/python/run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/python_actions \
  --report /root/work/himloco/evaluation/python-report.json \
  --warmup 10
```

## 输出

- 每个带源行号的输入对应一个 48 字节 float32 action 文件；
- JSON 报告记录模型/输入/输出哈希、Runtime 元信息、调度状态、逐样本延迟和汇总。

action 文件是 evaluator 输入，不是机器人命令。使用
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py)
与参考结果比较。

## 性能

在 DNN Runtime 1.24.5 的 RDK X5 上，100 个输入和 10 次预热的同步
`HB_HBMRuntime.run` 平均延迟为 0.885 ms（顺序吞吐 1129.37 FPS）。计时不含
文件 I/O、校验和 action 写入。

## 代码接口

`himloco.py` 对外提供：

- `HimLocoConfig`：模型路径和调度默认值；
- `HimLoco.pre_process`、`forward`、`post_process` 和 `predict`；
- `HimLoco.__call__`：与 `predict` 等价的可调用接口。

按照 [源码文档生成说明](../../../../../docs/source_reference/README.md) 生成 API 文档。

## 注意事项

- 本示例只执行离线 tensor 推理，不向机器人发送命令；
- X5 必须使用与板端 `libdnn` 匹配的 BSP Runtime 包；
- 输出目录和报告路径不能包含已有运行产物。
