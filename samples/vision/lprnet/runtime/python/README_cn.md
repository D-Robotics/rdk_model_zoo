# LPRNet Python runtime

<a id="environment"></a>
## 环境

在带有 `hbm_runtime` 的 RDK X5 系统镜像上使用 Python 3 和 NumPy。runtime 只有在选择、模型文件和板卡检查通过后才导入板端 SDK；`--help`、`--list-models`、`--dry-run` 不加载 SDK。编译模型必须暴露一个 float32 输入 `(1,3,24,94)` 和一个 float32 输出 `(1,68,18)`。

<a id="usage"></a>
## 使用

显式准备好 `model/lpr.bin` 后，在仓库根目录运行：

```bash
python3 -m samples.vision.lprnet.runtime.python.main --target x5
```

成功判断为退出码 `0` 且打印含 `plate` 的 JSON。无额外参数时输入默认绝对路径为 `samples/vision/lprnet/test_data/test_input.dat`。`bash samples/vision/lprnet/runtime/python/run.sh --target x5` 等价。

<a id="parameters"></a>
## 参数

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--target` | `auto` | `auto` 解析唯一发布目标 X5；S 目标拒绝 |
| `--asset-id` | `null` | 精确 `x5:lprnet:lpr.bin`，外部模型路径必需 |
| `--model-path` | `null` | 已存在模型路径；不下载 |
| `--test-bin` | `samples/vision/lprnet/test_data/test_input.dat` | 预打包 float32 输入 |
| `--priority` | `5` | runtime 调度优先级 |
| `--bpu-cores` | `[0]` | 一个或多个 BPU 核索引 |
| `--list-models` | `false` | 不加载 SDK，打印 manifest 制品 |
| `--dry-run` | `false` | 不加载 SDK 或模型，打印 binding 契约 |

`--list-models` 与 `--dry-run` 互斥。用户或 runtime 错误返回 `2`。

<a id="results"></a>
## 结果

CLI 打印 `target`、完整 `asset_id` 和 `plate`。`LPRNetTask.post_process` 返回 Python `str`，对 18 个时间步做 argmax、连续重复删除和 blank 索引 `67` 删除。raw logits 保持 float32，不做 softmax。

<a id="integration-example"></a>
## 集成示例

模型文件和随源输入存在后，下面示例定义所有变量并显式执行与 `predict` 相同的三阶段：

```python
from pathlib import Path
from samples.vision.lprnet.runtime.python.model_binding import resolve_selection
from samples.vision.lprnet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lprnet.runtime.python.lprnet import LPRNetTask

target = "x5"
asset_id = "x5:lprnet:lpr.bin"
model_path = Path("samples/vision/lprnet/model/lpr.bin")
test_bin = Path("samples/vision/lprnet/test_data/test_input.dat")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=5, bpu_cores=[0])
task = LPRNetTask(runner, binding)
prepared = task.pre_process(test_bin)
raw_logits = task.forward(prepared.tensors)
plate = task.post_process(raw_logits)
assert plate == task.predict(test_bin)
print(plate)
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process(test_bin)` 精确读取 `1*3*24*94` 个 float32 值，返回含 `tensors, context` 的 `PreparedInput`，只有一个 NCHW tensor，不做图像变换。
- `forward(tensors)` 校验绑定的名称、shape、dtype，调用选定模型并返回 owned raw float32 `(1,68,18)` 数组。
- `post_process(raw)` 只执行源 CTC 风格解码并返回 `str`。
- `predict(test_bin)` 串联三个阶段；context 是本次输入路径，不写入可被下一次调用覆盖的 task 字段。

<a id="troubleshooting"></a>
## 故障排查

- 没有 `--asset-id x5:lprnet:lpr.bin` 的模型路径会被拒绝，避免按文件名猜协议。
- 缺失或尺寸错误的 `.dat` 会在 SDK 执行前报错。
- 输入/输出名称、shape 或 dtype 与 metadata 不符时 binding 失败；不会用 runtime cast 掩盖不匹配。
