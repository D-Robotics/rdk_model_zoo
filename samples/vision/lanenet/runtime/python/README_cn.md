[English](README.md) | [简体中文](README_cn.md)

# LaneNet Python 推理

<a id="environment"></a>
## 环境

实际推理需要 S100、匹配的 `hbm_runtime` Python 运行库、NumPy 和 OpenCV。主机检查入口（`--list-models`、`--dry-run`）不导入板端 SDK、不下载模型。主机测试显式注入 SDK 夹具，不能证明某个板卡镜像或 SDK 版本受支持。运行时版本可获取时写入报告，否则记为 `unknown`。

通过[显式模型下载器](../../model/README_cn.md)准备 HBM。运行入口不安装依赖。以下命令均从仓库根目录执行；Shell 包装入口也会切换到该目录。

<a id="usage"></a>
## 使用方法

只检查选择结果，不执行推理：

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --dry-run
```

在 S100 上使用新输出目录执行一次：

```bash
bash samples/vision/lanenet/runtime/python/run.sh --target s100 --output outputs/lanenet_python
```

使用已准备好的外部模型与图片：

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/models/lanenet256x512.hbm --test-img /data/road.jpg --output outputs/lanenet_external
```

外部模型路径必须同时提供资产 ID。这只声明预期契约；缺少发布方校验和时，不能认证文件来源。不支持的目标显式失败，不静默选择 S100。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `auto` | 解析为唯一已发布的 S100 契约；实际执行另检查物理身份 |
| `--asset-id` | `null` | 自动推断精确发布身份；外部模型路径必填 |
| `--model-path` | `null` | 解析至 sample 的 `model/s100/lanenet256x512.hbm` |
| `--test-img` | `samples/vision/lanenet/test_data/lane.jpg` | OpenCV BGR 输入图片 |
| `--output` | `outputs/lanenet` | 新结果目录 |
| `--instance-save-path` | `null` | 额外嵌入显示图路径；保留源参数 |
| `--binary-save-path` | `null` | 额外二值显示图路径；保留源参数 |
| `--priority` | `0` | SDK 优先级，校验 0..255 |
| `--bpu-cores` | `[0]` | 一个或多个非负 SDK 核 ID；实际支持取决于运行库 |
| `--list-models` | `false` | 只列清单，不执行 |
| `--dry-run` | `false` | 只打印解析的身份与路径，不执行 |

两个检查模式互斥。额外显示文件必须不存在、互不相同，不能覆盖正式结果。不隐式预热或计时。保留源 Python 的优先级 0、核 0 调度；原生入口使用源 UCP 默认调度。

<a id="results"></a>
## 结果

| 文件 | 含义 |
| --- | --- |
| `raw_outputs.npz` | 全部具名输出，保留精确类型与形状；`report.json.raw_tensor_keys` 将 SDK 名称映射到 `output_N` 归档键 |
| `embedding.npy` | 独立 float32 `[3,256,512]` 嵌入，数值不变 |
| `binary.npy` | 独立 uint8 `[256,512]` 标签，仅允许 0/1 |
| `instance_pred.png` | 裁剪、舍入后的嵌入显示，不是聚类车道 ID |
| `binary_pred.png` | 用 0/255 显示标签 |
| `report.json` | 模型/输入摘要、真实元数据、调度、处理边界与可选显示路径 |

不能直接把 NPZ 键当作语义名称，应使用报告中的映射。任务未消费的其他实际输出也保留原始值。源文案提到的第三个输出没有名称，本实现不虚构其身份。

显示时将嵌入裁剪至 [0,1]，乘 255，按最近值、半数取偶舍入，并保留通道顺序。源 Python 直接相乘后转 uint8，导致截断或越界回绕；该显示变更不影响 `embedding.npy`。二值标签超出 0/1 时拒绝处理，不将其画成看似合理的掩码。结果保持 256×512，不插值回原图、不拟合车道。

<a id="integration-example"></a>
## 应用集成

在 S100 准备好默认模型后，从仓库根目录执行。使用与 CLI 相同的任务类和 runner，不绘图、不写结果文件：

```python
import cv2
from samples.vision.lanenet.runtime.python.model_binding import resolve_selection
from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lanenet.runtime.python.lanenet import LaneNetTask

selection = resolve_selection("s100")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = LaneNetTask(runner, binding)
image = cv2.imread("samples/vision/lanenet/test_data/lane.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise ValueError("Cannot decode input image")
result = task.predict(image)
assert result.embedding.shape == (3, 256, 512)
assert result.binary.shape == (256, 512)
```

需要比较原始输出时，分开调用 `pre_process`、`forward`、`post_process`，保留 `forward` 返回的映射。绘图与文件读写放在应用层。未建立应用层同步策略前，不应在并发调用间共享可变 SDK runner。

<a id="stage-io"></a>
## 阶段 IO 与绑定

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `pre_process` | 非空三通道 BGR uint8 HWC | 从绑定输入名到连续 float32 NCHW `[1,3,256,512]` 的映射 |
| `forward` | 预处理映射 | 从运行库存储复制出的全部实际具名原始数组 |
| `post_process` | 与元数据匹配的原始映射 | `LaneResult`：CHW 浮点嵌入和 HW uint8 二值标签 |
| `predict` | BGR 图像 | 组合上述三阶段 |

前处理保持源算术：BGR→RGB，INTER_AREA 拉伸到宽 512/高 256，/255，均值 `[0.485,0.456,0.406]`，标准差 `[0.229,0.224,0.225]`，转 CHW 并加 batch。校准数据复用同一纯图像函数。不增加 letterbox、sigmoid、softmax、argmax 或聚类。

绑定要求单模型、单 float32 `[1,3,256,512]` 输入、float32 `[1,3,256,512]` 的 `instance_seg_logits`，以及 int64 `[1,1,256,512]` 或 `[1,256,512]` 的 `binary_seg_pred`。检查名称、维数、类型、有限值与全部声明的输出形状。辅助输出允许 float16/float32/int8/uint8/int16/int32/int64，形状须固定且各维为正。原始数值保留，任务返回所选结果的独立副本。Python 辅助类型范围比当前原生实现宽（原生无 float16），跨语言比较前必须检查真实元数据。

<a id="troubleshooting"></a>
## 排障与验证边界

- 模型缺失：按模型说明显式准备；`--dry-run` 不验证文件内容。
- 身份拒绝：确认物理板卡，S100P/S600 不等于 S100。
- 元数据不符：保留实际名称/形状/类型；未确认语义时不能通过改名绕过检查。
- 图像无效或输出路径已存在：使用可解码图像与新目录。部分 IO 失败可能留下不完整目录，复用结果前先检查错误。
- 颜色不符合预期：单独检查原始嵌入，本实现没有实例聚类。

主机测试覆盖源前处理、绑定、原始数据所有权、标签、显示，以及使用伪 SDK 的真实 CLI。板端推理、真实 SDK 兼容性、数据集精度和延迟仍为 **not-run**。声明精度或等价前请阅读[评估边界](../../evaluator/README_cn.md)。
