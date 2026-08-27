[English](./README.md) | [简体中文](./README_cn.md)

# HIMLoco C++ Runtime

本示例通过 RDK X5 BPU SDK 加载融合模型、运行带源行号的离线输入，并写出 float32
action 和 JSON 报告。

## 环境依赖

- 提供 DNN Runtime 头文件和动态库的 RDK X5；
- CMake 3.10 或更高版本和支持 C++17 的编译器；
- gflags 头文件和动态库。

RDK OS 可访问 apt 时，使用以下命令安装构建依赖：

```bash
sudo apt install cmake g++ libgflags-dev
```

BSP 提供 `dnn/hb_dnn.h`、`dnn/hb_sys.h`、`libdnn.so` 和
`libhbrt_bayes_aarch64.so`。

## 目录结构

```text
cpp/
├── inc/himloco.hpp   # Config、metadata 和可复用公开接口
├── src/himloco.cc    # X5 模型资源与标准推理流水线
├── src/main.cpp      # gflags 命令行推理与 JSON 报告
├── CMakeLists.txt    # C++17 构建配置
├── run.sh            # 模型下载、Release 构建和一键运行
├── README.md         # 英文说明
└── README_cn.md      # 中文说明
```

构建产物和生成文件分别在被忽略的 `build/` 和 `runs/` 下。

## 模型合同

| Tensor | 名称 | 类型 | Runtime shape |
| --- | --- | --- | --- |
| 输入 | `obs_history` | float32 | `[1,270,1,1]` |
| 输出 | `actions` | float32 | `[1,12,1,1]` |

实现会查询 tensor 属性、按 `alignedByteSize` 分配内存、CLEAN 输入 cache、INVALIDATE
输出 cache，并按 aligned shape 提取有效输出。

## 编译工程

```bash
cd runtime/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

`run.sh` 会自动完成上述步骤。

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model_path` | RDK X5 Bayes-e 模型 | `../../model/bayes-e/himloco_go2_bayese_1x270.bin` |
| `--input_path` | 单个输入或输入目录 | `../../test_data/obs_history` |
| `--output_dir` | 新 action dump 目录 | `runs/<timestamp>/action_dumps` |
| `--report` | 新 JSON 证据报告 | `runs/<timestamp>/cpp-report.json` |
| `--warmup` | 不计时的 Runtime 调用次数 | `10` |
| `--priority` | DNN 优先级 [0,255] 或 -1 | `-1` |

## 快速运行

在 HIMLoco 示例目录执行以下命令；默认模型缺失时会自动下载、完成构建并运行 21 个输入：

```bash
bash runtime/cpp/run.sh
```

使用外部数据时可显式指定参数：

```bash
bash runtime/cpp/run.sh \
  --model_path /root/work/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin \
  --input_path /root/work/himloco/test_data/obs_history \
  --output_dir /root/work/himloco/evaluation/cpp_actions \
  --report /root/work/himloco/evaluation/cpp-report.json \
  --warmup 10
```

## 输出

- 每个带源行号的输入对应一个 48 字节 float32 action 文件；
- JSON 报告记录板端/DNN 版本、tensor metadata、逐样本延迟、p50/p95 和顺序吞吐。

使用 [`../../evaluator/compare_action_dumps.py`](../../evaluator/compare_action_dumps.py)
与 JIT 参考比较。

## 性能

在 DNN Runtime 1.24.5 的 RDK X5 上，100 个输入和 10 次预热的结果：

| Min | Mean | P50 | P95 | Max | 顺序 FPS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.332 ms | 0.350 ms | 0.349 ms | 0.364 ms | 0.396 ms | 2853.09 |

延迟只包含 `hbDNNInfer` 和 `hbDNNWaitTaskDone`，不含文件 I/O 和 cache 维护。

## 代码接口

`HimLoco` 构造函数不执行重资源操作。先调用 `init`，再使用
`pre_process`、`infer`、`post_process`，或组合接口 `predict`。
所有公开流水线方法成功返回 0、失败返回 -1；失败后通过 `last_error()` 获取原因。

按照 [源码文档生成说明](../../../../../docs/source_reference/README.md) 生成 API 文档。

## 注意事项

- 本示例只执行离线 tensor 推理，不向机器人发送命令；
- 单个 `HimLoco` 实例不是线程安全的；
- X5 只有一个 BPU 核，因此 C++ 示例仅暴露优先级，不提供核绑定参数。
