# LPRNet Python runtime

<a id="overview"></a>
## Python 推理

使用 LPRNet 从已准备的 float32 输入张量解码车牌文本。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── lprnet.py  # 模型初始化与推理阶段
├── cli.py  # 参数、模型选择与结果交付
├── main.py  # 命令行入口：构造模型并调用 predict
└── run.sh  # 运行示例
```

从 [main.py](main.py) 开始：入口构造 `LPRNetRecognizer` 并调用 `predict`。[lprnet.py](lprnet.py) 实现模型初始化及推理阶段；[cli.py](cli.py) 负责参数、模型选择和结果交付。模型初始化会加载 Runtime，应用可复用同一个实例执行多次预测。

<a id="environment"></a>
## 环境

在带有 `hbm_runtime` 的 RDK X5 镜像上使用 Python 3 和 NumPy。通过 `--help`、`--list-models` 和 `--dry-run` 查看参数及模型选择。发布模型 `lpr.bin` 接收 float32 `(1,3,24,94)` 输入，输出 float32 `(1,68,18,1)` logits。任务 API 也接受 `(1,68,18)` logits，各输出必须与其声明的元数据一致。

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

CLI 打印 `target`、完整 `asset_id` 和 `plate`。`LPRNetRecognizer.postprocess` 返回 Python `str`，先只移除绑定布局的单元素轴——发布版制品为 `(1,68,18,1)`——得到源 `(68,18)` CTC 载荷，再对 18 个时间步做 argmax、连续重复删除和 blank 索引 `67` 删除。raw logits 保持 float32，不做 softmax。

<a id="integration-example"></a>
## 集成示例

模型文件和随源输入存在后，下面示例定义所有变量并显式执行与 `predict` 相同的三阶段：

```python
from pathlib import Path
from samples.vision.lprnet.runtime.python.cli import resolve_selection
from samples.vision.lprnet.runtime.python.lprnet import LPRNetRecognizer

target = "x5"
asset_id = "x5:lprnet:lpr.bin"
model_path = Path("samples/vision/lprnet/model/lpr.bin")
test_bin = Path("samples/vision/lprnet/test_data/test_input.dat")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
task = LPRNetRecognizer(selection)
prepared = task.preprocess(test_bin)
raw_logits = task.infer(prepared.tensors)
plate = task.postprocess(raw_logits)
assert plate == task.predict(test_bin)
print(plate)
```

<a id="stage-io"></a>
## 三阶段 I/O

- `preprocess(test_bin)` 精确读取 `1*3*24*94` 个 float32 值，返回含 `tensors, context` 的 `PreparedInput`，只有一个 NCHW tensor，不做图像变换。
- `infer(tensors)` 校验绑定的名称、shape、dtype，调用选定模型并返回 owned raw float32 数组，shape 为绑定的完整 native shape——发布版 `lpr.bin` 为 `(1,68,18,1)`；每次调用都必须与绑定 shape 严格一致。
- `postprocess(raw)` 只移除绑定 native logits 的单元素轴（绝不 reshape 或重排轴），然后执行源 CTC 风格解码并返回 `str`。
- `predict(test_bin)` 串联三个阶段；context 是该次输入路径，不写入可被下一次调用覆盖的 task 字段。
- 既有的 `pre_process`、`forward`、`post_process` 名称保留为 `preprocess`、`infer`、`postprocess` 的可导入薄别名——同一实现，两个名字。

<a id="troubleshooting"></a>
## 故障排查

- 没有 `--asset-id x5:lprnet:lpr.bin` 的模型路径会被拒绝，避免按文件名猜协议。
- 缺失或尺寸错误的 `.dat` 会在 SDK 执行前报错。
- 输入/输出名称、shape 或 dtype 与 metadata 不符时 binding 失败；不会用 runtime cast 掩盖不匹配。
- 输出 metadata shape 不在 `(1,68,18,1)`/`(1,68,18)` 之内——例如 `(1,18,68,1)` 或 `(1,68,18,2)`——绑定失败；运行时输出的 shape 与绑定 shape 漂移时在解码前报错。
