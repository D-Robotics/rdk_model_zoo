[English](README.md) | 简体中文

# HIMLoco Python 策略阶段

<a id="overview"></a>
## Python 推理

在 X5 上使用六帧观测历史离线运行 HIMLoco 策略。`HimLocoTask.from_model` 加载 Runtime；`predict` 返回十二维原始策略动作。`cli.py` 读取输入并保存动作文件与报告。

<a id="directory"></a>
## 目录结构

```text
python/
├── cli.py  # 参数、模型选择与结果展示
├── input_io.py  # 输入文件与数据记录
├── main.py  # 命令行入口：构造模型并调用 predict
├── model_binding.py  # 模型选择与物理张量契约
├── policy.py  # 模型阶段与预测
└── run.sh  # 定位 Python 入口并转发参数
```

<a id="environment"></a>
## 环境

核心仅依赖 Python 和 NumPy，不导入板端 SDK、Torch、机器人中间件或转换工具链。
以下命令均从仓库根目录执行，使用已安装 NumPy 的仓库主机 Python 环境。
实际 X5 推理需要与板端库匹配的 BSP `hbm_runtime`，核心不会安装或替代此依赖。

<a id="usage"></a>
## 使用方式

准备模型后，通过 `HimLocoTask.from_model(selection)` 初始化 Runtime。模型输入为 `{"obs_history": float32[1,270]}`，输出为 `{"actions": float32[1,12]}`。一般调用 `predict(observation)`，也可按下方示例分别执行三个阶段。

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
| `preprocess(...).tensors` | 独立存储、连续的 float32 `obs_history` `[1,270]` |
| `infer(tensors)` | 准确命名的物理输入；返回含原始动作及本次耗时的 `RawOutputs` |
| `postprocess(raw)` | 必须传入 `RawOutputs`，不依赖实例中的“上次结果” |

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
## 集成示例

先准备 X5 模型，从仓库根目录执行。示例读取第一份观测历史，返回形状为 `(1,12)` 的策略动作。

```python
from pathlib import Path
import numpy as np
from samples.robotics.himloco.runtime.python.model_binding import resolve_selection
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

selection = resolve_selection("x5")
task = HimLocoTask.from_model(selection)
task.set_scheduling_params(priority=0, bpu_cores=[0])
input_dir = Path("samples/robotics/himloco/test_data/obs_history")
input_path = min(input_dir.glob("*.bin"), key=lambda path: int(path.stem))
observation = np.fromfile(input_path, dtype="<f4").reshape(6, 45)
result = task.predict(observation)
print(result.actions.shape, result.actions.tolist())

# Optional access to intermediate stages.
prepared = task.preprocess(observation)
raw = task.infer(prepared.tensors)
staged_result = task.postprocess(raw)
```

<a id="stage-io"></a>
## 阶段语义与源对齐

每帧 45 维依次为速度指令（3）、角速度（3，源缩放 0.25）、投影重力（3）、
相对关节位置（12）、相对关节速度（12，源缩放 0.05）、上次动作（12）。
当前帧在前，之后是五帧历史。核心不会重复执行缩放、积累历史或选择关节顺序。

`preprocess` 独立持有打包特征，`infer` 调用一次模型，`postprocess` 校验并独立持有动作数组，`predict` 串联三步。`pre_process`、`forward`、`post_process` 委托给相同的阶段实现。结果与计时记录按每次调用独立保存。

<a id="troubleshooting"></a>
## 故障处理

观测数量不符时，应按训练策略构建完整历史，不盲目补零或截断。输出名称／形状／类型
不符时，应核对所绑定模型的接口，核心不会静默强转不兼容模型的输出。
手动构造 `RawOutputs` 时，耗时须为有限非负数。原生运行见 [C++ 说明](../cpp/README_cn.md)，
动作文件对照见[评测说明](../../evaluator/README_cn.md)。
