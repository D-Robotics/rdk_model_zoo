# HIMLoco Python 策略阶段

[English](README.md)

统一 Python 入口已提供准确的 X5 模型选择、懒加载 SDK、源索引输入校验、预热、
独立动作导出和失败报告。主机集成测试使用明确的 SDK 替身，真实板端执行仍未运行。
源运行时说明 (historical `../../../../../platforms/x5/samples/robotics/himloco/runtime/python/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)
保留历史板测证据，不代表统一入口重新完成板测。

<a id="environment"></a>
## 环境

核心仅依赖 Python 和 NumPy，不导入板端 SDK、Torch、机器人中间件或转换工具链。
以下命令均从仓库根目录执行，使用已安装 NumPy 的仓库主机 Python 环境。
实际 X5 推理需要与板端库匹配的 BSP `hbm_runtime`，核心不会安装或替代此依赖。

<a id="usage"></a>
## 使用方式

构造 `HimLocoTask(runner)`，runner 接收物理映射
`{"obs_history": float32[1,270]}`，返回 `{"actions": float32[1,12]}`。
SDK 适配器负责目标板、制品身份和实际模型元数据。核心不下载模型、不打开设备。
一般调用 `predict(observation)`，也可按下方示例分别执行三个阶段。

```bash
# Repository root; these commands do not load SDKs or download.
python samples/robotics/himloco/runtime/python/main.py --list-models
python samples/robotics/himloco/runtime/python/main.py --target x5 --dry-run

# Explicit model preparation, followed by offline inference on X5.
bash samples/robotics/himloco/model/download_model.sh --target x5
python samples/robotics/himloco/runtime/python/main.py --target x5 \
  --input-path samples/robotics/himloco/test_data/obs_history \
  --output-dir outputs/himloco
```
`run.sh` 转发相同参数，可用 `PYTHON` 指定解释器。每次运行使用新输出目录。
外部模型路径必须同时指定 `--asset-id x5:himloco:himloco_go2_bayese_1x270.bin`，
仍校验同一发布 SHA-256。目标板不符或模型缺失／摘要不符，在创建 SDK 前失败。
使用 BSP 提供的 runtime，不安装 PyPI 上无关的同名 hbm_runtime 包。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `auto` | 执行时检测本机；list 的 auto 映射 x5；dry-run 需显式 x5 |
| `--list-models` | `false` | 列出准确发布信息，不加载 SDK、不联网、不写文件 |
| `--dry-run` | `false` | 仅预览选择，不验证真实运行时元数据 |
| `--asset-id` | `null` | 外部模型路径须指定准确发布身份 |
| `--model-path` | `null` | 默认使用 Sample 下 model/bayes-e/himloco_go2_bayese_1x270.bin |
| `--input-path` | `samples/robotics/himloco/test_data/obs_history` | 数字命名 BIN 或目录；实际默认值为 Sample 内绝对路径 |
| `--output-dir` | `outputs/himloco` | 相对工作目录的新动作输出目录 |
| `--report` | `null` | 默认 output-dir/report.json；另指定的文件也须不存在 |
| `--warmup` | `10` | 使用首条输入预热的非负次数，不计入输出样本 |
| `--priority` | `null` | 可选整数 0–255，传入 SDK |
| `--bpu-cores` | `null` | 可选非空列表，非负 SDK 核心索引 |

| API 输入 | 契约 |
| --- | --- |
| `runner` | 必填可调用对象，无默认值，不隐式创建 SDK |
| `observation` | 恰好 270 个有限实数，按输入顺序展平并转为 float32 |
| `pre_process(...).tensors` | 独立存储、连续的 float32 `obs_history` `[1,270]` |
| `forward(tensors)` | 准确命名的物理输入；返回含原始动作及本次耗时的 `RawOutputs` |
| `post_process(raw)` | 必须传入 `RawOutputs`，不依赖实例中的“上次结果” |

扁平 `[270]`、批次 `[1,270]`、历史 `[6,45]` 均保持源实现展平规则。
复数、布尔、字符串／对象、非有限值及 float32 溢出会拒绝。输入原始传感器数据
不会自动构造训练时的观测量。

<a id="results"></a>
## 结果

`HimLocoResult.actions` 为独立存储的 float32 `[1,12]`，不裁剪、不激活、
不重排、不反量化、不缩放。`latency_ms` 仅测同步 runner 调用，不含核心的复制、
前处理和后处理，也不是纯加速器耗时。每次原始输出携带自身时间，先运行 A、再运行 B、
最后处理 A，不会取错成 B 的耗时。SDK 异常直接传播，不生成备用动作。

源部署在模型边界之外应用 `default_joint_position + 0.25 * actions`。
本核心只返回动作数组，不发送机器人控制指令。

CLI 成功返回 0，生成按源索引命名的 `000000.bin` 等小端 float32 文件，每个 48 字节，
以及 `completed` JSON 报告。报告包括模型／输入／输出摘要、清单来源、运行时元数据、
请求的调度参数、完成的预热次数、UTC 时间，以及最小／平均／p50／p95／最大 runner 耗时。
输出创建后发生异常返回 2，保留 `failed` 报告、当前源索引和已完成文件，不对部分结果
提供汇总耗时。前置检查失败不创建输出目录，已有结果不复用、不覆盖。强制终止可能
留下 `running` 报告，应按未完成处理。

输入必须是数字命名的 BIN，每个恰好 1080 字节，按数值源索引排序；重复索引拒绝。
若旁边存在 `../runtime-input-manifest.json`，须满足固定输入契约，并匹配选中文件的
索引与摘要。没有清单时来源明确记录为 null，不伪造。摘要来自实际读入推理的同一份
字节。不生成文本转写，也不执行控制器动作。

<a id="integration-example"></a>
## 可执行集成示例

```bash
python - <<'PYCODE'
import numpy as np
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

def fixture_runner(tensors):
    assert tensors["obs_history"].shape == (1, 270)
    return {"actions": np.arange(12, dtype=np.float32).reshape(1, 12)}

task = HimLocoTask(fixture_runner)
observation = np.zeros((6, 45), dtype=np.float32)
prepared = task.pre_process(observation)
raw = task.forward(prepared.tensors)
explicit = task.post_process(raw)
result = task.predict(observation)
np.testing.assert_array_equal(explicit.actions, result.actions)
print(result.actions.shape, result.actions.tolist())
PYCODE
```

预期输出形状 `(1, 12)`，动作值为 0 到 11。这些来自测试夹具，而非学习策略。
核心不保存历史或每次调用的上下文，调用者须提供全部六帧观测。若 SDK runner 使用
共享缓冲区，应串行调用；结果独立存储不表示设备运行时具备线程安全性。

<a id="stage-io"></a>
## 阶段语义与源对齐

每帧 45 维依次为速度指令（3）、角速度（3，源缩放 0.25）、投影重力（3）、
相对关节位置（12）、相对关节速度（12，源缩放 0.05）、上次动作（12）。
当前帧在前，之后是五帧历史。核心不会重复执行缩放、积累历史或选择关节顺序。

`pre_process` 打包并独立持有特征，`forward` 只调用一次模型，`post_process`
校验并独立持有动作，`predict` 串联三步。已使用源清单摘要核对全部 21 个归档观测文件，
前处理与源实现一致；后处理仅以模拟动作输出对照。未测试板端／模型精度、控制稳定性
或机器人运动。源实现可变的“上次耗时”字段和可能共享缓冲区的结果，改为每次调用
独立的记录与结果存储。

<a id="troubleshooting"></a>
## 故障处理

观测数量不符时，应按训练策略构建完整历史，不盲目补零或截断。输出名称／形状／类型
不符时，应核对所绑定模型的接口，核心不会静默强转不兼容模型的输出。
手动构造 `RawOutputs` 时，耗时须为有限非负数。原生运行见 [C++ 说明](../cpp/README_cn.md)，
动作文件对照见[评测说明](../../evaluator/README_cn.md)。主机测试不代表 SDK／板端兼容性验证。
