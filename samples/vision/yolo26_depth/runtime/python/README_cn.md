# YOLO26 Depth Python 推理

<a id="environment"></a>
## 环境

使用目标板兼容的 Python、`hbm_runtime`、NumPy、OpenCV 和 PyYAML。
仅真正推理时才加载 SDK。主机无需 SDK 即可列出制品、执行 `--dry-run`；
这些操作不会下载模型，也不证明模型能在板上运行。
参阅[模型准备](../../model/README_cn.md)和[样例概览](../../README_cn.md)。

<a id="usage"></a>
## 使用

从仓库根目录执行：

```bash
python samples/vision/yolo26_depth/runtime/python/main.py --list-models
python samples/vision/yolo26_depth/runtime/python/main.py --target s600 --variant l --dry-run
python samples/vision/yolo26_depth/model/download.py --target x5 --variant n
python samples/vision/yolo26_depth/runtime/python/main.py --target x5 --variant n --output outputs/depth-x5-n
```

最后一条需要 X5 和已准备的模型。`run.sh` 会先切到仓库根目录，因此用户传入的
相对路径以仓库根为基准；直接调用 Python 时，以当前目录为基准。默认模型和
内置图片路径以样例目录定位。输出目录必须尚不存在。

`x5/s100/s100p/s600` 各有 `n/s/m/l/x` 五种变体。省略变体默认 `n`，但精确
asset ID 可以推导其他变体。target、variant、asset ID 冲突会报错。
生产加载器在 SDK 加载前检查板身份；选择 target 并不意味着可以跨板运行模型。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `auto` | `auto/x5/s100/s100p/s600`；默认 `auto` 检测身份 |
| `--variant` | `null` | `n/s/m/l/x`；默认 `n`，或从精确 asset ID 推导 |
| `--asset-id` | `null` | 精确清单引用；通过 `--list-models` 获取 |
| `--model-path`、`--model` | `null` | 外部制品路径；必须同时指定精确 asset ID |
| `--converted-model` | `false` | 明确使用自行转换制品；必须指定模型路径和契约 asset ID |
| `--test-img`、`--input` | `samples/vision/yolo26_depth/test_data/bus.jpg` | 可解码的 BGR 图片；默认内置 `test_data/bus.jpg` |
| `--output` | `outputs/yolo26_depth` | 新输出目录；默认 `outputs/yolo26_depth` |
| `--warmup` | `3` | 非负预热 forward 次数；默认 3 |
| `--priority` | `null` | 0–255；S 默认 0，X5 未指定时保留 SDK 默认值 |
| `--bpu-cores` | `null` | 一个或多个非负核编号；S 默认 `[0]`，X5 保留 SDK 默认值 |
| `--list-models` | `false` | 列出清单制品，不加载 SDK |
| `--dry-run` | `false` | 解析选择和输出边界，不推理、不下载 |

列出与 dry-run 模式互斥。外部已发布 X5 模型仍需匹配发布者摘要。自行编译模型
必须显式使用 `--converted-model`：asset ID 只选择张量契约，不认证这些字节。
目标和元数据检查仍然有效；报告写入 `artifact_origin=user-converted`、
`asset_id=null`、契约引用及实际文件摘要。参阅[自转换示例](../../model/README_cn.md)。
源清单未提供 S 的发布者摘要，本地计算的摘要不能证明发布者身份。
可用 BPU 核编号由实际 SDK 决定。

<a id="results"></a>
## 结果

成功时写入 `log_depth.npy`（192×192 float32）、`depth_native.npy`
（原图高×宽 float32）、`depth.png`、`overlay.png` 和 `report.json`。
Lite 模型另写 `raw_logit.npy`（192×192 float32）。深度为相对量，不是米。
颜色沿用源实现的 2%/98% 分位范围和反向 TURBO，不代表精度指标。

报告记录模型/图片摘要、模型选择、运行时元数据和 forward 耗时。耗时包含 runner
校验与复制，不是纯 BPU 延迟，也不包含前后处理。未知运行时版本如实记为
`unknown`。主机测试使用受控运行时替身；板端精度、真实 SDK 执行及数据集指标
仍为 `not-run`。历史数字见[评测说明](../../evaluator/README_cn.md)。

<a id="integration-example"></a>
## 集成示例

准备模型后，在目标板执行。此示例不保存文件、不预热、不计时：

```python
import cv2
from samples.vision.yolo26_depth.runtime.python.model_binding import resolve_selection
from samples.vision.yolo26_depth.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import Yolo26DepthTask

selection = resolve_selection("x5", variant="n")
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = Yolo26DepthTask(runner, binding)
image = cv2.imread("samples/vision/yolo26_depth/test_data/bus.jpg")
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
result = task.post_process(raw, prepared.context)
# 等价调用：result = task.predict(image)
print(result.depth_native.shape)
```

`PreparedInput` 携带张量和不可变的单次调用几何上下文。`DepthResult` 携带
`log_depth`、`depth_native`、可选 `raw_logit` 和对应上下文。每帧必须保留匹配的
上下文，task 不保存“上一帧变换”。注入 runner 是主机测试接口，不是板测证明。
CLI 单独设置调度参数；应用可在支持时调用
`runner.set_scheduling_params(priority=0, bpu_cores=[0])`。
归档源 API 及其内嵌计时改为三个阶段和 `predict`；计时、图片 IO、渲染归调用方。

<a id="stage-io"></a>
## 阶段契约

| 阶段 | 契约 |
| --- | --- |
| `pre_process(image)` | 非空 BGR uint8 HWC → `PreparedInput` |
| `forward(tensors)` | 按名称传入物理张量 → 未解码的单个 float32 SDK 输出 |
| `post_process(raw, context)` | 已绑定输出及匹配几何 → 拥有独立存储的相对深度数组 |
| `predict(image)` | 顺序执行上述三个阶段一次 |

X5 全部变体及 S 的 `n/s/m` 使用 768×768 INTER_LINEAR letterbox，填充值 114，
再转换为**单个 884736 字节的扁平 NV12 uint8 数组**，不是独立 Y/UV 输入。
192×192 输出已经是校准后的 log depth。后处理执行 exp、放大至 768、裁去填充、
恢复原图尺寸。几何使用 Python ties-to-even 舍入；缩放后某一边变成零会显式拒绝。

S 的 `l/x` 使用 INTER_LINEAR 直接拉伸、BGR→RGB、`/255`，得到 float32 NCHW
`[1,3,768,768]`。输出是原始 logits：截断到 `[-4,5]`，乘 scale 1，加 bias
`-0.2498779296875`（`l`）或 `-0.316650390625`（`x`），再 exp 并直接恢复尺寸。
exp 和尺寸恢复在 CPU 执行。源文字声称它们在图内，与其导出/运行代码矛盾；
根 README 链接的源审计记录了此问题。

元数据必须描述一个模型、一个输入和一个 float32 输出，输出形状为
`[1,192,192,1]` 或 `[1,1,192,192]`。形状/类型错误、NaN/Inf、上下文不匹配、
exp 溢出均报错。不要对已校准的 log depth 再套 lite 校准。task 隔离几何状态
不等于 SDK runner 线程安全；除非 SDK 明确支持，否则共享 runner 的调用应串行。

<a id="troubleshooting"></a>
## 排查

- **板身份未知/不匹配：** 在所选支持目标上运行。dry-run 仅检查选择，不能证明兼容。
- **SDK 缺失：** 按平台说明安装匹配运行时；只有 NumPy/OpenCV 无法执行模型。
- **摘要不符：** 重新获取精确发布制品。主动转换应使用显式转换模式，不能把损坏下载
  改标为自转换来绕过检查。
- **元数据不符：** 检查输入输出契约，改文件名不能修复布局或 raw/log 边界。
- **输出已存在：** 换新目录，防止残留文件混入本次结果。
- **图片无效/溢出：** 提供非空可解码 BGR 图片，并检查模型输出；无效值不会被静默裁剪。
