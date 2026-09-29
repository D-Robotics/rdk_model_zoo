[English](README.md) | [简体中文](README_cn.md)

# Python 运行时

<a id="environment"></a>
## 环境

需要兼容 S100 的 `hbm_runtime`、Python、NumPy、OpenCV、PyYAML。列出和 dry-run
可在无板端 SDK 的主机执行。源包装脚本自动安装 NumPy1.26.4、OpenCV4.11.0.86、
Torch2.3.1；这些是历史固定版本，不代表本轮兼容声明。当前无自动安装，源仅用于
尺寸恢复的 Torch 依赖已移除。参见[模型准备](../../model/README_cn.md)。

<a id="usage"></a>
## 使用

从仓库根目录执行：

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --list-models
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 --dry-run
bash samples/vision/depth_anything_v2/model/download.sh --target s100
bash samples/vision/depth_anything_v2/runtime/python/run.sh --target s100 \
  --test-img samples/vision/depth_anything_v2/test_data/furseal.jpg \
  --output outputs/depth-anything-default
```

只有最后两条准备/执行模型。真实推理需要 S100。shell 先切到仓库根，直接 Python
调用则从当前目录解析用户路径。默认图片/模型以样例定位。输出目录和额外图片路径
均必须不存在。额外彩色图片不能占用五个固定输出文件的路径。

源可选 letterbox 能力现在通过明确参数使用：

```bash
python -m samples.vision.depth_anything_v2.runtime.python.main --target s100 \
  --resize-type 1 --output outputs/depth-anything-letterbox \
  --img-save-path outputs/depth-anything-letterbox-result.jpg
```

与源后处理不同，现在先裁去 letterbox 填充再恢复原图。参考/候选比较必须使用相同
模式。

<a id="parameters"></a>
## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `auto` | auto 选择唯一 S100 制品；x5/s100p/s600 无制品并报错 |
| `--asset-id` | `null` | 精确 `s:depth_anything_v2:s100/depth_any.hbm` |
| `--model-path` | `null` | 外部副本；需精确 asset ID |
| `--test-img` | `samples/vision/depth_anything_v2/test_data/furseal.jpg` | 解码为 BGR uint8 的图片 |
| `--output` | `outputs/depth_anything_v2` | 新输出目录 |
| `--img-save-path` | `null` | 可选额外源风格彩色图片；新路径 |
| `--resize-type` | `0` | 0 最近邻拉伸；1 线性 letterbox、灰127填充 |
| `--priority` | `0` | 0–255 整数；传入 SDK 调度 |
| `--bpu-cores` | `[0]` | 非负核编号；可用核由 SDK 决定 |
| `--list-models` | `false` | 列出匹配清单，不加载 SDK |
| `--dry-run` | `false` | 解析选择，不下载、不推理 |

路径默认 `null` 表示从样例推导 `model/s100/depth_any.hbm`。列出和 dry-run 互斥。
目标选择不能绕过实际身份。调度保留源 priority0/core[0]，无隐式预热或计时。
外部路径选择的是契约，不认证发布者字节；清单未提供预期摘要。

<a id="results"></a>
## 结果

| 文件 | 含义 |
| --- | --- |
| `raw_depth.npy` | 未解码 float32 `[1,518,686]` 输出 |
| `depth_native.npy` | 恢复至原图 H×W 的 float32 相对深度 |
| `depth_gray.png` | 逐图 min/max 显示归一化的 uint8 |
| `depth_color.png` | INFERNO 显示图片 |
| `report.json` | 模型/输入摘要、目标、元数据、运行时版本、归一化和调度 |

报告不测延迟。未知运行时版本/发布者摘要保持未知。有限负值如实保留，不把它们
静默解释为米制距离。恒定图变为零灰度，但零值着色是 colormap 最低颜色，不一定
是黑色。无效/NaN/Inf 拒绝处理；拒绝已有输出，避免混入残留文件。

<a id="integration-example"></a>
## 集成示例

准备模型后在 S100 执行，图片 IO 留在 task 外：

```python
import cv2
from samples.vision.depth_anything_v2.runtime.python.model_binding import resolve_selection
from samples.vision.depth_anything_v2.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.depth_anything_v2.runtime.python.depth_anything_v2 import DepthAnythingV2Task

runner = RuntimeModelRunner(resolve_selection("s100"))
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = DepthAnythingV2Task(runner, binding, resize_type=0)
image = cv2.imread("samples/vision/depth_anything_v2/test_data/furseal.jpg")
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
result = task.post_process(raw, prepared.context)
# 等价调用：result = task.predict(image)
print(result.depth_native.shape)
```

`PreparedInput` 携带按名称组织的物理张量和不可变 `ImageContext`。
`DepthResult` 包含独立存储的浮点 `depth_native` 与其上下文。每帧保持匹配上下文，
task 不保存上一帧尺寸。注入 runner 用于主机测试，不证明硬件执行；API 不保证共享
runner 并发安全。

源 `DepthAnythingV2Config`/`DepthAnythingV2` 返回显示 uint8，task 还提供调度和
`__call__`。新 API 明确分离 runner，返回浮点深度。需要旧显示形式时调用
`visualization.normalize_depth(result.depth_native)` 或 `colorize_depth(...)`。
归档源仍可使用，不声称新接口静默兼容旧 API。

<a id="stage-io"></a>
## 阶段契约

| 阶段 | 输入 → 输出 |
| --- | --- |
| `pre_process` | 非空 BGR uint8 HWC → 含 float32 `[1,3,518,686]` 的 `PreparedInput` |
| `forward` | 命名输入映射 → 独立原始 float32 `[1,518,686]`，不做激活 |
| `post_process` | 原始张量 + 匹配上下文 → 原图尺寸浮点 `DepthResult` |
| `predict` | 三阶段执行一次，无计时、渲染、IO |

默认输入缩放为 INTER_NEAREST，保留源 helper 的实际行为。BGR→RGB 后，每像素三
通道使用 `(rgb - mean(rgb)) / sqrt(var(rgb) + 1e-5)`，再转置并转换 float32。
这**不是 ImageNet 均值/标准差，也不是 `/255`**，源注释与实际代码不符。源对 uint8
以 float64 统计计算，当前保留此顺序。

Letterbox 使用向下取整尺寸、INTER_LINEAR、127填充，灰色填充归一化为零；缩放边
变成零会显式拒绝。后处理裁去可选填充，用 OpenCV INTER_LINEAR 恢复原图。默认
拉伸恢复与源 Torch `align_corners=False` 具有相同半像素线性几何，但舍入/浮点
累积可能不同，不声称逐位相等。主机解析仿射平面测试检查几何，实际 HBM/Torch
一致性未测。

元数据必须恰好包含一个模型/输入/输出，形状和 float32 类型符合声明。图内 int16
量化不是 IO 类型声明。形状/类型错误、非有限值或几何上下文不匹配会报错；不要对
不兼容输出 reshape 以绕过验证。

<a id="troubleshooting"></a>
## 排查

- **S100P 或其他目标拒绝：** 仅 S100 有制品；源脚本无条件回退 S100 已移除，不要
  覆盖板身份。
- **SDK/模型缺失：** 显式准备匹配运行时和模型；主机 dry-run 不是推理测试。
- **元数据不符：** 检查实际制品/运行时版本，不能根据图内量化文字放行整数输出。
- **颜色不同：** 先比较浮点数组和前处理模式，逐图显示归一化会隐藏尺度/偏移差异。
- **恒定图：** 零灰度是明确行为，但声称准确前需调查输入/模型；不推导数据集验收。
- **输出已存在：** 换新路径，不混入部分写入或历史结果。
