[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco C++ Runtime

本 Runtime 使用 RDK X5 BPU SDK C API 加载融合 `.bin`、检查 tensor 合同，并运行带源
行号的离线策略输入。它不构造实时机器人观测，也不向执行器发送命令。

## 环境要求

- 提供 DNN Runtime 头文件和动态库的 RDK X5；
- CMake 3.10 或更高版本；
- 支持 C++17 的编译器。

BSP 通常在 `/usr/include` 和 `/usr/lib` 提供 `dnn/hb_dnn.h`、`dnn/hb_sys.h`、
`libdnn.so` 和 `libhbrt_bayes_aarch64.so`。

## 模型合同

| Tensor | 名称 | 类型 | Runtime shape |
| --- | --- | --- | --- |
| 输入 | `obs_history` | float32 | `[1,270,1,1]` |
| 输出 | `actions` | float32 | `[1,12,1,1]` |

实现会查询全部 tensor 属性，按 `alignedByteSize` 分配内存；CPU 写入后 CLEAN 输入 cache，
BPU 写入后 INVALIDATE 输出 cache，并根据模型报告的 aligned shape 提取有效输出。

## 准备输入

在主机上从独立 rollout 生成带源行号的输入：

```bash
python3 samples/robotics/himloco/evaluator/prepare_runtime_inputs.py \
  --data /work/himloco/rollout_evaluation.pt \
  --output /work/himloco/evaluation/runtime_inputs \
  --num-samples 0
```

每个 `.bin` 输入必须包含 270 个 float32（1080 字节）。将模型和输入目录复制到开发板。

## 编译运行

```bash
cd /root/work/himloco/runtime/cpp
bash run.sh \
  --model-path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input-path /root/work/himloco/test_data/obs_history \
  --output-dir /root/work/himloco/evaluation/cpp_actions \
  --report /root/work/himloco/evaluation/cpp-report.json \
  --warmup 10
```

`run.sh` 在 `build/` 下配置 Release 构建、编译 `himloco_cpp` 并运行。仅在受控调度
实验中添加 `--priority 0..255`。X5 只有一个 BPU 核，因此不提供多核绑定参数。

## 输出

- 每个带源行号的输入对应一个 48 字节 float32 action 文件；
- JSON 报告记录板端/DNN 版本、valid/aligned shape、逐样本延迟、p50/p95 和顺序吞吐。

输出是 evaluator 的临时证据，必须放在仓库外。使用
[`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py)
与 JIT 参考对比。

## 性能

在 DNN Runtime 1.24.5 的 RDK X5 上，100 个输入、10 次预热的结果：

| Min | Mean | P50 | P95 | Max | 顺序 FPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

延迟只包含 `hbDNNInfer` 和 `hbDNNWaitTaskDone`，不含文件 I/O 和 cache 维护；其计时
范围与 Python Runtime 不同。

## 代码接口

`inc/himloco.hpp` 声明可复用 `HimLoco` 接口；`src/himloco.cpp` 实现模型资源、tensor
内存、cache 同步、推理和对齐输出提取；`src/main.cpp` 实现带源行号的 CLI 和证据报告。

按照 [源码文档生成说明](../../../../../docs/source_reference/README.md) 生成并浏览 API 文档。
