[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco 模型转换

本目录将融合 HIMLoco TorchScript 策略导出为静态 ONNX，并编译为 RDK X5 Bayes-e
`.bin` 模型。

## 目录结构

```text
conversion/
├── export_onnx.py          # 融合 TorchScript -> 静态 opset-11 ONNX
├── prepare_calibration.py  # 原生 rollout .pt -> Mapper float32 输入
├── mapper.py               # 生成 YAML、执行 Mapper 并记录证据
├── README.md
└── README_cn.md
```

`mapper.py` 根据经过校验的命令行参数生成完整 YAML，不再维护静态 YAML 模板。生成的
YAML 保存在每次 run 的 `config/` 中，作为实际生效的构建记录。

## 模型协议

| 项目 | 取值 |
| --- | --- |
| 输入 | `obs_history`，float32 `[1,270]` |
| 输出 | `actions`，float32 `[1,12]` |
| ONNX batch/opset | 固定 batch 1 / opset 11 |
| Runtime 输入 | DDR featuremap、NHWC、不预处理 |
| PTQ | MIX、float32 校准数据 |
| 编译 | latency 模式、单核、默认 O3 |

单帧 45 维观测顺序如下：

```text
velocity_commands(3)
base_ang_vel(3, scale=0.25)
projected_gravity(3)
joint_pos_rel(12)
joint_vel_rel(12, scale=0.05)
last_action(12)
```

270 维输入由当前观测和前 5 帧观测组成。裁剪、缩放、关节顺序和历史顺序必须与训练
策略输入边界完全一致。

## 环境

ONNX 导出在能够加载 `policy.pt` 的训练/导出环境中执行。已验证版本为 PyTorch 2.7.0、
ONNX 1.22.0 和 NumPy 1.26.0。

PTQ 在 x86 Linux 主机的 RDK X5 OpenExplorer 环境中执行。已验证工具链为
OpenExplorer v1.2.8 / Mapper 1.24.3。开始前检查依赖和工具：

```bash
python3 -c "import torch, onnx, numpy, yaml"
hb_mapper --version
hb_model_info --help
```

工具链参考：

- <https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/overview>
- <https://toolchain.d-robotics.cc/>

## 1. 导出融合 ONNX

在生成或能够加载融合 JIT 的环境中运行：

```bash
cd samples/robotics/himloco/conversion
python3 export_onnx.py \
  --jit /work/himloco/policy.pt \
  --output /work/himloco/export/himloco_go2_op11.onnx
```

脚本检查 `[1,270] -> [1,12]` 合同、执行 `onnx.checker`，并比较确定性
TorchScript/ONNX ReferenceEvaluator 输出。同时生成相邻的 `.export.json`，记录哈希、
版本、算子和数值指标；已有输出不会被覆盖。

## 2. 准备校准数据

在代表性仿真 rollout 中采集策略输入边界的准确 tensor。原生文件应包含：

```python
{"obs_history": torch.Tensor}  # [N,270]
```

生成 100 个确定性校准样本：

```bash
python3 prepare_calibration.py \
  --input /work/himloco/rollout_calibration.pt \
  --tensor-key obs_history \
  --num-samples 100 \
  --seed 20260820 \
  --output /work/himloco/calibration
```

每个 `.bin` 包含 270 个连续 float32（1080 字节）。`calibration-manifest.json` 记录
源文件哈希、选中行号、统计量和预处理合同。不能把序列化 `.pt` 直接改名为 `.bin`。

## 3. 执行 Checker 和 MIX PTQ

在 X5 OpenExplorer 环境中运行：

```bash
python3 mapper.py \
  --onnx /work/himloco/export/himloco_go2_op11.onnx \
  --calibration /work/himloco/calibration/obs_history \
  --calibration-type mix \
  --optimize-level O3 \
  --output /work/himloco/compile_mix_o3
```

脚本校验 calibration manifest、生成实际 YAML，并执行 `hb_mapper checker`、
`hb_mapper makertbin` 和 `hb_model_info`。输出路径不得已经存在。

```text
compile_mix_o3/
├── artifacts/
│   ├── himloco_go2_bayese_1x270.bin
│   └── himloco_go2_bayese_1x270_quantized_model.onnx
├── config/himloco_go2_bayese_1x270.yaml
├── reports/
│   ├── checker.log
│   ├── makertbin.log
│   ├── hb_model_info.log
│   └── compile-report.json
└── working/
```

## 验证结果

已验证 MIX/O3 构建结果：

| 指标 | 结果 |
| --- | ---: |
| 输出 cosine | 0.999606 |
| 编译器估计延迟 | 63.0 us |
| 编译器估计吞吐 | 15870.5 FPS |
| 编译器估计 DDR 流量 | 261776 bytes |

这些是 Mapper 指标，不是独立任务精度或完整 Runtime 延迟。使用 `hb_model_info` 确认
`.bin` march 后，按照 [评测说明](../evaluator/README_cn.md) 完成独立数值和板端输出验证。

## 注意事项

- 使用代表性 rollout 状态；随机高斯输入不能作为发布级校准替代；
- 校准与独立评测 rollout 不得重叠；
- 生成模型、校准数据、日志和报告必须存放在仓库外；
- Mapper 未报告输出相似度或 `hb_model_info` 未确认 `bayes-e` 时不得发布模型。

## 许可证

本目录工具遵循仓库许可证。训练代码、策略和重新分发产物仍受其上游条款约束。
