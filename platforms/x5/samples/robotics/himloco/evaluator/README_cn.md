[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco 模型评测

本目录比较融合浮点模型格式，并使用独立原生 rollout 验证 X5 策略输出。

## 文件

- `compare_jit_onnx.py`：TorchScript 与浮点 ONNX 对比；
- `prepare_runtime_inputs.py`：原生 rollout 转带源行号的 X5 输入；
- `compare_action_dumps.py`：X5 float32 action 与 JIT/记录动作对比；
- `metrics.py`：通用数据加载、哈希和数值指标。

使用已经提供 PyTorch、ONNX 和 NumPy 的训练/导出环境：

```bash
python3 -c "import torch, onnx, numpy"
```

生成的输入、action dump 和报告必须写在仓库外，例如
`/work/himloco/evaluation`。

## 数据合同

独立评测 rollout 不得与 PTQ 校准数据重叠。推荐原生 `.pt` 结构：

```python
{
    "obs_history": torch.Tensor,  # [N,270]
    "actions": torch.Tensor,      # 可选 [N,12]
}
```

`obs_history` 应在完成部署等价的裁剪、缩放、关节排序和历史堆叠后采集。记录动作必须是
尚未缩放的策略输出。

## 1. 验证 TorchScript 和 ONNX

```bash
python3 compare_jit_onnx.py \
  --jit /work/himloco/policy.pt \
  --onnx /work/himloco/export/himloco_go2_op11.onnx \
  --data /work/himloco/rollout_evaluation.pt \
  --report /work/himloco/evaluation/jit-vs-onnx.json
```

默认门禁为逐样本最小 cosine `>= 0.999`、action 最大绝对误差 `<= 1e-4`。失败时仍会
写出报告，并返回非零退出码。

## 2. 准备 X5 输入

```bash
python3 prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

每个输入包含 270 个连续 float32。文件名数字部分保留原始 rollout 行号，确保板端输出
能够匹配正确参考。

## 3. 验证 X5 Action Dump

使用任一板端 Runtime 执行准备好的输入，将 action 复制回主机后与融合 JIT 对比：

```bash
python3 compare_action_dumps.py \
  --jit /work/himloco/policy.pt \
  --data /work/himloco/rollout_evaluation.pt \
  --candidate-dir /work/himloco/evaluation/x5_actions \
  --report /work/himloco/evaluation/x5-vs-jit.json \
  --min-cosine 0.99
```

若 `actions` 确定由同一个导出策略记录，可省略 `--jit`，并使用
`--action-key actions`。

本策略没有图像或其他可人工判断的输出，因此保留 action dump Runtime 功能。每个带
源行号的 48 字节文件在 X5 专用 Runtime 和主机 TorchScript evaluator 之间提供可移植
边界，可独立计算哈希、比较 Python/C++，也无需在板端安装 PyTorch。它们是临时验证
证据，不是提交文件。

报告包含 action MAE、RMSE、最大误差、全局/逐样本最小 cosine，以及使用 `0.25` rad
动作缩放换算的关节目标误差。

## RDK X5 性能数据

同一个 MIX 模型使用 100 个输入、10 次预热：

| Runtime | 计时范围 | Min | Mean | P50 | P95 | Max | 顺序 FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | `HB_HBMRuntime.run` | 0.818 ms | 0.885 ms | 0.880 ms | 0.942 ms | 0.975 ms | 1129.37 |
| C++ | `hbDNNInfer` + wait | 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

环境为 RDK X5、RDK OS 3.5.0-beta、DNN Runtime 1.24.5、HBRT 3.15.55。两者计时
范围不同且不含文件 I/O。发布正式性能指标前，应使用 `hrt_model_exec perf` 进行统一的
纯模型测试。

## 验收顺序

```text
JIT 与 ONNX
  -> Mapper 输出 cosine
  -> X5 action 与 JIT
  -> 闭环仿真
  -> 有安全约束的真机验证
```

离线数值一致是必要门禁，但不能证明控制安全。

## 许可证

评测代码遵循仓库许可证。策略和 rollout 数据仍受其对应条款约束。
