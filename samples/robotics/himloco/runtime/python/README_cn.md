# HIMLoco Python 策略阶段

[English](README.md)

本目录目前提供 `policy.py` 离线策略核心。统一 SDK 适配器和 CLI 尚在迁移，下方示例
注入明确的模拟输出，不执行发布模型。完整[源运行时说明](../../../../../platforms/x5/samples/robotics/himloco/runtime/python/README.md)
作为迁移依据保留，不表示其中入口已在本目录实现。

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

<a id="parameters"></a>
## 参数

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
手动构造 `RawOutputs` 时，耗时须为有限非负数。统一 CLI、模型准备、原生运行时及
完整 Sample 文档仍在迁移，不能将当前核心状态当作整套 Sample 已验收。
