# MobileNetV4 Python 运行

`main.py` 是 canonical 的用户命令。它从发布 Manifest 解析唯一的模型引用，
核验检测到的板卡，懒加载 `hbm_runtime`，执行一条 `ClassificationTask` 流程。
模型准备是显式操作；本运行时不下载模型、不安装软件包。

<a id="environment"></a>
## 环境

在目标板卡的 Python 环境运行，需要与板卡匹配的 `hbm_runtime`、NumPy 和
OpenCV-Python；读取 Manifest 需要 PyYAML。`hbm_runtime` 只存在于板卡镜像
中并被懒加载——`--help`、`--list-models`、`--dry-run` 与主机 unittest 套件
均不需要它。主机侧测试依赖见 sample 的 `requirements-host.txt`。

<a id="usage"></a>
## 用法

cwd：仓库根目录。默认命令（除模式标志外零参数，在没有板卡与制品时不可执行，
因此最小的免 SDK 调用是列清单模式）：

```bash
# 成功判据：打印全部发布引用，退出码 0，不加载 SDK
python3 samples/vision/mobilenetv4/runtime/python/main.py --list-models --target auto
```

在已准备制品的 X5 板卡上的完整运行：

```bash
# 前置：bash samples/vision/mobilenetv4/model/download.sh x5 --variant small
# 成功判据：退出码 0 且打印 Top-5 列表
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

S100/S600 替换为对应的 `s:` 引用与 `s100/`/`s600/` 制品路径；标签文件两侧
共用。`--dry-run --target x5` 在无板卡访问、无模型加载、无下载的情况下解析
选择。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--variant` | choice | null | 模型变体（省略时默认 `small`；发布组合见 `--list-models`） |
| `--target` | choice | auto | 执行目标：`auto`、`x5`、`s100`、`s100p`、`s600`；执行目标必须与检测到的硬件一致 |
| `--asset-id` | string | null | 完整的 `group:sample:filename` Manifest 引用 |
| `--model-path` | string | null | 已存在的 `.bin`/`.hbm`；必须与 `--asset-id` 配对；省略时默认取解析引用对应的 `model/` 位置 |
| `--test-img` | string | samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG | BGR 输入图像 |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | 逐行一个类别的 ImageNet 标签 |
| `--top-k` | int | 5 | 打印的结果数量 |
| `--topk` | int | 5 | `--top-k` 的旧拼写 |
| `--resize-type` | int | null | `0` 直接缩放或 `1` letterbox（BGR 127 填充）；默认跟随绑定的源实现 |
| `--priority` | int | 0 | 运行时调度优先级（0-255） |
| `--bpu-cores` | int 列表 | [0] | 运行时 BPU 核编号 |
| `--img-save-path` | string | null | 可选的标注结果图输出路径 |
| `--list-models` | flag | false | 无板卡访问列出 Manifest 支持的引用 |
| `--dry-run` | flag | false | 不加载模型或 SDK 地解析/检查选择 |

上面的默认值由 Q3 检查器对照 `build_parser()` 做机器校验。

<a id="results"></a>
## 结果

命令打印稳定的 Top-K（`ClassificationResult(class_ids, scores, labels)`），
仅当给出 `--img-save-path` 时写出标注图像。X5 收到 packed NV12 的 canonical
一维 uint8 缓冲（`H*W*3/2` 字节；224x224 即 75,264 字节，与源实现的
`(1,336,224,1)` 视图字节一致）；S100/S600 收到 Y `(1,224,224,1)` 与 UV
`(1,112,112,2)` uint8 数组（medium 变体为 256x256：Y `(1,256,256,1)`，UV `(1,128,128,2)`）。
发布制品返回原始 logits；任务在 Top-K 前施加数值稳定的 softmax（两个平台一致）。
输出 shape 遵循 rank 规则：任何能 squeeze 到 `(1000,)` 的单例批次/空间拼写
均可绑定（发布制品声明 `raw_f32` 变换；量化制品需要显式 `dequant` 契约）。

<a id="integration-example"></a>
## 集成示例

前置：制品已准备（见 [model/README_cn.md](../../model/README_cn.md)）且
OpenCV-Python 可导入。示例中的每个输入变量都有定义：

```python
import cv2

from samples.vision.mobilenetv4.runtime.python.classification import ClassificationTask
from samples.vision.mobilenetv4.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.mobilenetv4.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "x5",
    asset_id="x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin",
    model_path="samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

三阶段也可以显式驱动：`prepared = task.pre_process(image)`、
`outputs = task.forward(prepared.tensors)`、`result = task.post_process(outputs)`
——`predict` 精确串联这三个阶段（由阶段契约测试验证）。

<a id="stage-io"></a>
## 阶段 I/O

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `pre_process` | 一张任意尺寸的 BGR `uint8` 图 | `PreparedInput.tensors`（目标形状的 NV12 张量）+ `PreparedInput.transform`（本次调用的冻结缩放上下文） |
| `forward` | `prepared.tensors` | 原始输出字典（X5 F32 `[1,1000,1,1]`；S F32 `[1,1000]`）——与 runner 输出逐位一致，无解码 |
| `post_process` | 原始输出（分类不消耗几何上下文） | `ClassificationResult(class_ids, scores, labels)`，按声明的分数策略做稳定降序 Top-K |
| `predict` | BGR `uint8` 图 | 串联三阶段，返回同一 `ClassificationResult` |

<a id="troubleshooting"></a>
## 故障排查

| 症状 | 检查 |
| --- | --- |
| `Cannot identify this board` | 先用显式 target 做 dry-run，再只在匹配的板卡上执行；显式 target 不是硬件证据。 |
| `model_path requires --asset-id` | 从 `--list-models` 复制完整引用；不要用裸文件名。 |
| `No published ... asset`（S100P） | Manifest 没有 s100p 资产行；在匹配板卡上使用 S100/S600 制品。 |
| 输入 shape 或 dtype 不匹配 | 核对制品引用与运行时 metadata；不要互换 X5 packed 与 S split 制品。 |
| 输出与旧实现不一致 | 先固定相同制品、图像、缩放方式、Top-K 比较 raw 输出，再考虑分数语义。 |

主机检查（仓库根目录）：
`python3 -m unittest discover -s samples/vision/mobilenetv4/tests -v`。
