[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco Python Runtime

本 Runtime 加载融合 RDK X5 `.bin`、检查 tensor 合同，并在 BPU 上运行带源行号的离线
策略输入。它不构造实时机器人观测，也不向执行器发送命令。

## 环境要求

- RDK X5，RDK OS 3.5.0 或更高版本；
- BSP 随附、且与板端 `libdnn` 匹配的 `hbm_runtime`；
- Python 3 和 NumPy。

```bash
cat /etc/version
python3 -c "from hbm_runtime import HB_HBMRuntime; print(HB_HBMRuntime.version)"
```

不要安装 PyPI 上无关的同名 `hbm_runtime` 包。

## 模型合同

| Tensor | 名称 | 类型 | 逻辑 shape |
| --- | --- | --- | --- |
| 输入 | `obs_history` | float32 | `[1,270]` |
| 输出 | `actions` | float32 | `[1,12]` |

batch 为 1 且元素数一致时可接受等价 Runtime 布局。错误的名称、类型、shape、NaN/Inf
或非 `.bin` 模型会在推理前被拒绝。

## 准备输入

在主机上从独立 rollout 生成带源行号的输入：

```bash
python3 samples/robotics/himloco/evaluator/prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

将模型和 `runtime_inputs` 目录复制到开发板。

## 运行

```bash
cd /root/work/himloco/runtime/python
bash run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/python_actions \
  --report /root/work/himloco/evaluation/python-report.json \
  --warmup 10
```

默认由 Runtime 自动调度。仅在受控调度实验中使用 `--priority` 或 `--bpu-cores`。
执行 `python3 main.py --help` 查看完整参数。

## 输出

- 每个带源行号的输入对应一个 48 字节 float32 action 文件；
- JSON 报告记录模型/输入/输出哈希、Runtime 元信息、调度状态、逐样本延迟和汇总。

action 文件是 evaluator 的临时输入，不是仓库资产。应复制回主机，并使用
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py) 检查。

## 性能

在 DNN Runtime 1.24.5 的 RDK X5 上，100 个输入、10 次预热的同步
`HB_HBMRuntime.run` 平均延迟为 0.885 ms（顺序吞吐 1129.37 FPS）。计时不含文件 I/O、
输入校验和 dump 写入。

## 代码接口

`himloco.py` 提供可复用 `HimLoco` 类；`main.py` 实现输入发现、manifest 校验、批量执行、
action 序列化和 JSON 证据报告。

按照 [源码文档生成说明](../../../../../docs/source_reference/README.md) 生成并浏览 API 文档。
