# Ultralytics YOLO 模型转换

[English](README.md)

本目录把 Ultralytics 浮点模型导出为 ONNX，再编译为共用 Python Sample
使用的目标 BPU 制品。导出和编译在主机执行，不能在板端运行。此次合并的
代表范围是 YOLOv8/YOLO11 的 DFL 检测和 YOLO26 的直接 LTRB 检测。下文也列出已有分类、分割、姿态和 OBB 导出入口；存在脚本不等于已完成各任务端到端转换验证。

<a id="source-model"></a>

## 源模型与复现身份

输入为与所选任务匹配的 Ultralytics PyTorch `.pt` 权重。`/models/*.pt` 是用户需预先准备的本地路径，不是仓库附带资产。导出时记录权重 SHA-256、训练/导出环境版本与模型类别/输入尺寸；本仓库没有给所有家族固定同一组 Ultralytics、PyTorch、ONNX 版本，也没有提供所有权重的发布摘要。自训练权重须保留训练配置和类别顺序。已发布 `.bin/.hbm` 的 [模型清单](../model/README_cn.md) 不证明某个同名 `.pt` 就是其原始权重。

<a id="toolchain-targets"></a>
## 准备两个主机环境

第一步使用 Ultralytics 训练/导出环境，需提前准备 `ultralytics`、PyTorch
以及该权重所需的 ONNX 导出依赖。请先确认本地权重路径再调用脚本。脚本
直接调用 `ultralytics.YOLO`；Ultralytics 对已知的裸模型名可能自动下载，
因此裸名称不能证明使用了本地权重。本仓库不把这种隐式下载记录为来源或
验证；需要可复现时应传入已经存在的绝对 `.pt` 路径。可以在本目录运行，
也可以给权重和输出传绝对路径。

第二步使用对应的 D-Robotics OpenExplore 工具链环境：

| 目标 | 工具检查 | 制品 | 标定文件 | 编译配置中的输入约定 |
| --- | --- | --- | --- | --- |
| X5 | `hb_mapper --version` | `.bin` | 原始 float32 `.rgbchw` | `bayes-e`，运行时 `nv12` |
| S100 | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-e`，运行时 `nv12` |
| S100P | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-m`，运行时 `nv12` |
| S600 | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-p`，运行时 `nv12` |

Mapper 会检查工具链是否已加载、ONNX 是否只有一个静态
`tensor(float)` 四维输入，以及标定目录中是否有 JPG/JPEG/PNG。缺依赖时
只报错，不会静默安装。工具链环境和板端运行时镜像是两套环境，不要把
`hb_mapper` 或 `hb_compile` 安装步骤放入板端部署流程。

X5 可在 x86 Linux 主机使用本仓库此前记录的 OE 1.2.8 CPU 镜像。若发布
版本不同，以 D-Robotics 提供的镜像和 tag 为准；镜像来源为
[X5 工具链包](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz)：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

S 系列请从 [RDK S OE 工具链文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
或[工具链下载页](https://toolchain.d-robotics.cc/)选择与 S100/S100P 或 S600
匹配的镜像，再加载和挂载：

```bash
docker load -i ai_toolchain_ubuntu_22_s100_<release>.tar
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <loaded-s100-or-s600-image>:<tag> /bin/bash
```

可参考 [S100/S100P 工具链页](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=4.0.5&p=RDK+S100)
和 [S600 工具链页](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=5.1.0&p=RDK+S600)。
`-v /workspace` 会把权重、ONNX、标定图片和输出目录暴露给容器；进入容器
后在 `/workspace` 运行下方 Mapper 命令。

<a id="export"></a>
## 导出 ONNX

通用 Ultralytics 检测器通过 `--platform` 选择已核定的 opset 默认值：X5
使用 opset 11，S 系列使用 opset 19。`--opset` 与历史拼写 `--optse` 等价，
显式传入的值优先。

```bash
# YOLOv8/YOLO11 风格 DFL 检测器，导出给 X5。
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --platform x5 --pt /models/yolo11n.pt --require-local --opset 11

# 同一模型系列，导出给 S600。
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --platform s600 --pt /models/yolo11n.pt --require-local --opset 19
```

通用导出器通过 `ultralytics.YOLO` 加载权重，在模型头应用 Sample 的 BPU
monkey patch，然后调用 `model.export(format='onnx', simplify=False,
opset=...)`。Ultralytics 通常把 ONNX 写在权重旁边；启动 Mapper 前请确认
实际路径，Mapper 不会猜测或替换相似文件。

YOLO26 检测器使用单独的图导出器，因为它输出三个 NHWC 分类张量和三个
四通道直接 LTRB 张量。默认值为 X5 `opset=11, simplify=1`，S 系列
`opset=19, simplify=0`。

```bash
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task detect --platform x5 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx

python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task detect --platform s600 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx
```

也可以直接运行 `yolo26/export_yolo26_detect_bpu.py`，参数为 `--weights`、
`--output`、`--imgsz`、`--platform`、`--opset`、`--simplify` 和可选的
`--require-local`。导出失败会
返回错误，不会在文档中把一个未生成的路径当成制品。

所有 Python 示例从仓库根目录运行。通用导出器根据权重中的 head 类型应用 patch；`--task` 不能将检测权重变成分类/分割权重。YOLO26 由 `--family yolo26 --task ...` 分发到任务专用脚本，默认分类输入 224，其他任务 640。以下只展示现有导出配方，不宣称本轮实际运行了导出：

```bash
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task cls --platform x5 \
  --pt /models/yolo26n-cls.pt --imgsz 224 --output /models/yolo26n_cls_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task seg --platform x5 \
  --pt /models/yolo26n-seg.pt --imgsz 640 --output /models/yolo26n_seg_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task pose --platform x5 \
  --pt /models/yolo26n-pose.pt --imgsz 640 --output /models/yolo26n_pose_bpu.onnx
python samples/vision/ultralytics_yolo/conversion/export_monkey_patch.py \
  --family yolo26 --task obb --platform x5 \
  --pt /models/yolo26n-obb.pt --imgsz 640 --output /models/yolo26n_obb_bpu.onnx
```

YOLO26 的 cls/seg/pose/obb 导出器尚不支持 `--require-local`；请在运行前自行确认上述绝对路径存在，不能把该参数传给它们。

将生成的 ONNX 交给下面的 mapper，YOLO26 添加 `--family yolo26`。预期输入为静态 batch-one float32 NCHW；检测 DFL 与直接 LTRB 输出不可互换。分割附带 mask 系数/prototype，姿态附带关键点，OBB 附带角度；分类输出 logits，运行端进行 Softmax。检查对应导出脚本的输出说明和运行时绑定，不要仅按输出数量判断兼容性。

<a id="dataflow"></a>
## DFL 系列数据流：图的输出与运行时解码

源转换指南（X5：rdk_x5 @ac11571；S：rdk_s
@380e1a2bf42041af54be6f34935e50197cfadff9——两者分别与存档的 `platforms/x5`、
`platforms/s` 副本逐字节一致）用下面的插图解释 DFL 系列部署流程，此处作为
源材料保留。它们描述 DFL 检测协议（YOLOv5u/v8/v9/v10/11/12/13）及其分割、
姿态扩展。YOLO26 检测没有对应插图，其协议差异在本节末尾单独说明。

### 目标检测（DFL）

![](./imgs/ultralytics_yolo_detect_dataflow.png)

在标准处理流程中，会完整计算全部 8400 个 Bounding Box（bbox）的
scores、categories 和 xyxy 坐标，用于结合 GT 计算 loss。但部署阶段只需要
保留满足分数阈值的 bbox，因此没有必要对全部 8400 个 bbox 做完整计算。

这里的优化主要利用 Sigmoid 函数的单调性，在计算前先进行筛选。DFL 和特征
解码阶段同样采用"先筛选，再计算"的思路，从而节省大量计算量并降低推理
耗时。

- **分类部分：ReduceMax 操作**

ReduceMax 用于在指定维度上取最大值。在 YOLO 检测头中，该操作在 8400 个
Grid Cell 的 80 个类别分数中找最大值，操作维度为 C 维。该操作输出最大
值本身，而不是最大值对应的类别索引。

Sigmoid 函数具有单调性，因此 80 个分数在 Sigmoid 前后的相对大小关系
不变：

$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$

$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$

因此，模型输出最大值的位置就是最终 score 最大值的位置；对该输出值做
Sigmoid 才得到浮点模型的最大类别 score：
$\operatorname{Sigmoid}(\max \mathrm{logits}) = \max \operatorname{Sigmoid}(\mathrm{logits})$。
argmax 的先后排序在 Sigmoid 前后一致；但输出值本身仍是 logit，还不是
概率。

- **分类部分：Threshold(TopK) 操作**

Threshold(TopK) 用于筛选满足阈值要求的 Grid Cell，操作对象是 8400 个
Grid Cell，对 H/W 维度进行筛选；实现中可能将 H/W 展平，这只是便于实现
和表达，本质上没有区别。设某个 Grid Cell 某一类别的原始分数为 $x$，经过
Sigmoid 后的值为 $y$，阈值为 $C$，则该分数满足要求的充要条件为：

$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$

进一步可得：

$$x > -ln\left(\frac{1}{C}-1\right)$$

该操作得到满足阈值的 Grid Cell 索引及其对应最大值。最大值经过 Sigmoid
后，即为该 Grid Cell 的类别 score。

- **分类部分：GatherElements 和 ArgMax 操作**

利用 Threshold(TopK) 得到的索引，GatherElements 取出满足要求的 Grid
Cell，ArgMax 判断 80 个类别中最大值所在的类别，从而得到每个合格 Grid
Cell 的类别。

- **Bounding Box 部分：GatherElements 操作**

利用同样的 Grid Cell 索引，GatherElements 取出对应 bbox 信息，得到形状为
`1×64×k×1` 的 bbox 特征。

- **Bounding Box 部分：DFL（SoftMax + Conv）**

每个 Grid Cell 使用 4 个数描述 bbox 位置。DFL 结构会对某条边相对 Grid
Cell（anchor）位置的 offset 给出 16 个估计值。对这 16 个估计值执行
SoftMax，再通过卷积计算期望值。这是 Anchor Free 的核心设计：每个 Grid
Cell 只负责预测一个 Bounding Box。以某条边的 offset 为例，设 16 个估计值
为 $l_p$，其中 $p=0,1,...,15$，offset 的计算公式为：

$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

- **Bounding Box 部分：Decode（dist2bbox / ltrb2xyxy）**

该操作将每个 Bounding Box 的 ltrb 描述解码为 xyxy 描述。ltrb 表示左、
上、右、下四条边相对 Grid Cell 中心的距离：

![](./imgs/ltrb2xyxy.jpg)

设输入尺寸 $Size=640$，bbox 预测分支第 $i$ 个特征图（$i=1, 2, 3$）对应的
下采样倍数为 $Stride(i)$。YOLOv8-Detect 中 $Stride(1)=8$、$Stride(2)=16$、
$Stride(3)=32$，对应特征图尺寸 $n_i = Size/Stride(i)$，即 $n_1 = 80$、
$n_2 = 40$、$n_3 = 20$，合计 $n_1^2+n_2^2+n_3^2=8400$ 个 Grid Cell。对
第 $i$ 层第 $x$ 列、第 $y$ 行的 Grid Cell（$x$ 沿水平方向计数，$y$ 沿
竖直方向计数；$x,y \in [0, n_i)\cap Z$，$Z$ 为整数集），ltrb 到 xyxy 的
转化关系为：

$$x_1 = (x+0.5-l)\times{Stride(i)},\quad y_1 = (y+0.5-t)\times{Stride(i)}$$

$$x_2 = (x+0.5+r)\times{Stride(i)},\quad y_2 = (y+0.5+b)\times{Stride(i)}$$

最终的检测结果包括类别（id）、分数（score）和位置（xyxy）。

**这些阶段在本 sample 中的执行位置。** 插图是源指南对 DFL 流程的数据流
视角。本仓库导出的图停在插图顶部绘制的每层 stride 消息上——NHWC 分类
logits（80 类模型在 stride 8/16/32 分别为 `1×80×80×80`、`1×40×40×80`、
`1×20×20×80`；自定义类别数改变 80）和 DFL box logits（`...×64`）——而
ReduceMax / 阈值筛选 / gather / ArgMax、DFL SoftMax 加期望 bin、
dist2bbox 等阶段的等价计算，以及在所选绑定需要时进行的按类别 NMS，都由
维护的 Python 后处理（`runtime/python/decode.py` 中的 `decode_dfl`；协议见
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md)）完成。S 系列 YOLOv10
绑定是维护的 NMS-free 例外：它复用同样的解码阶段并固定 `nms='none'`
（见[运行时 README](../runtime/python/README.md)）。运行时只接受
已完成反量化的浮点输出：整数张量或 SCALE 量化元数据在绑定期直接报错，
维护代码不实现任何手动输出反量化。因此"反量化完成后"之类的历史表述
描述的是运行时 SDK 已经提供的结果，不是需要读者手写的步骤。

### 实例分割（DFL 系列任务）

![](./imgs/ultralytics_yolo_seg_dataflow.png)

实例分割基于目标检测流程扩展而来。检测分支筛选出满足要求的 bbox 后，
两次 GatherElements 取出该 Grid Cell 的 32 个 mask 系数，再与 proto
分支输出做线性组合（加权求和，图中画为 MatMul；proto 为 `1×160×160×32`，
即 stride 4），生成实例 mask。因此检测部分的 ReduceMax、
Threshold(TopK)、GatherElements、DFL 和 Decode 优化仍然适用。维护运行时
在浮点 head 上的后处理中执行同样的系数–proto 组合（`segmentation_decode`）。

### 姿态估计（DFL 系列任务）

![](./imgs/ultralytics_yolo_pose_dataflow.png)

> **历史插图勘误（图中标签已过时）。** 保留的源插图把 pose 消息标为每
> Grid Cell `×57`、类别分支保留 80 通道，同时又按 3×17 reshape。当前维护
> 的绑定与此不同：单类别通道（单人体 pose 模型）加每个 Grid Cell
> `3 × 17 = 51` 个关键点通道（17 个 COCO 关键点）。插图按源材料原样保留，
> 未做修改；请将其中的 pose/类别通道标签读作历史信息，而非当前契约。

Ultralytics YOLO Pose 的关键点基于目标检测结果。COCO keypoint 定义如下：

```python
COCO_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}
```

Pose 模型的目标检测部分与 Detect 模型一致，由 pose head 额外增加一个每
Grid Cell 的特征图。在已发布的 17 关键点 COCO 契约下，维护绑定要求每个
Grid Cell 提供 1 个类别通道加 `3 × 17 = 51` 个关键点通道：每个关键点包含
相对该层下采样倍率的 x、y 坐标以及一个可见性 score（`pose_decode.py` 绑定
`cls` 为 1 通道、`kpts` 为 `3 × nkpt`；已发布绑定固定 `nkpt = 17`，仅改变
形状声明不构成受支持的变体）。检测分支确定某个 Grid Cell 合格后，DFL 解码
按 `(raw_xy × 2 + anchor − 0.5) × stride` 计算关键点在模型输入坐标系中的
位置，其中 `anchor` 是该 Grid Cell 的半整数中心（`runtime/python/
rdk_yolo_utils/postprocess.py` 的 `decode_kpts`）；随后 `inverse_points`
连同 `inverse_boxes` 把模型输入的 letterbox 几何还原回原图，Sigmoid 把
关键点可见性 logits 转成 score（`pose_decode.py`）。作为对照，YOLO26
直接 LTRB pose 分支使用 `(raw_xy + anchor) × stride`，没有 DFL 的 ×2
形式。维护运行时在后处理中解码这些 head（`pose_decode`）。

### YOLO26 直接 LTRB 的差异

上面两张框图只描述 **DFL 协议**，不描述 YOLO26。YOLO26 检测导出三个
NHWC 分类张量加三个**直接四通道 LTRB** 张量（`[1,Hs,Ws,4]`，不是
`[1,Hs,Ws,64]`）：四个通道已经是 cell 单位的左/上/右/下距离，DFL 的
SoftMax 加 16-bin 期望阶段在该协议中不存在。`ltrb2xyxy.jpg` 插图属于
DFL 协议的解码。YOLO26 解码器在 raw-logit 空间取最大类别，只对选中的
类别做 Sigmoid，用同样的 cell 中心网格几何转换四个距离，再应用共用的
按类别 NMS（`decode_ltrb`）。DFL 与直接 LTRB 制品不可互换，见上文导出
一节和 [`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md)。

<a id="calibration"></a>
## 标定数据准备

在标定目录放入 20–50 张有代表性的输入图片。共用流程会把 BGR 转为 RGB，
缩放到静态 ONNX 宽高，HWC 转 NCHW，并写入 float32。X5 写入未除 255 的
RGB 张量 `*.rgbchw`；S 写入除以 255 后的 NumPy `*.npy`。这是编译器标定
格式；运行时的 NV12 打包仍由目标板输入绑定负责。

<a id="compile"></a>
## 编译

从仓库根目录运行统一入口（历史平台入口会补上相同的 platform）：

```bash
# X5 -> hb_mapper makertbin、bayes-e、.bin
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform x5 \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100 -> hb_compile、nash-e、.hbm
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s100 --march nash-e \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100P/S600 分别使用 nash-m/nash-p。
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s600 --march nash-p \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled
```

<a id="validation"></a>
## 转换后验证

Mapper 成功后，把对应目标的制品复制到板端，用完整路径运行共用运行时。
例如上面的 S600 命令生成 `/models/compiled/yolo11n_nashp_640x640_nv12.hbm`：

```bash
# 在 S600 板卡上从仓库根目录运行
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/compiled/yolo11n_nashp_640x640_nv12.hbm \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolo11n-s600.jpg
```

X5 使用 `--platform x5` 和 `.bin` 文件名；其他 Nash 目标使用对应的
`--platform s100`/`s100p` 与 `.hbm` 文件名。编译成功不能跳过运行时的
输入/输出绑定，板端仍会在推理前检查制品元数据及 DFL 或直接 LTRB 契约。

通用导出器及 YOLO26 检测导出器可传 `--require-local` 要求权重已存在；省略该参数则保留 Ultralytics
解析已知裸模型名的历史能力。

通用 YOLO 和 YOLO26 mapper 现在调用同一个 `conversion/workflow.py`。其中
`mapper_x5.py`、`yolo26/mapper_x5.py` 选择 X5 配置；`mapper_s.py`、
`yolo26/mapper_s.py` 选择 Nash 配置并保留 `--march`。共用流程仍明确保留
以下目标差异：

* X5 执行 `hb_mapper makertbin --config config.yaml --model-type onnx`，
  使用 `bayes-e` 和原始 RGB/NCHW 标定；保留 Softmax int8 优化，
  `--quantized int16` 时追加 `set_all_nodes_int16`。
* S 执行 `hb_compile --config config.yaml`，使用选择的 Nash march、归一化
  NumPy 标定，并加入 `input_no_padding`、`output_no_padding`；
  `--quantized int16` 时加入 S 的 `quant_config` 模型/输出类型设置。

配置写在 `--ws` 指定目录下本次创建的唯一子目录中（`--ws` 是工作区父
目录）。成功后默认删除该子目录，`--save-cache` 会保留它。编译失败会保留
子目录中的配置、标定文件和日志以便排查。已有同名制品必须显式传
`--overwrite` 才会替换；不会删除用户提供的工作区父目录、输入模型、标定
图片目录或输出目录。

<a id="artifacts"></a>
## 输出名称和路径

当 ONNX 文件名为 `yolo11n`、输入为 640×640 时，默认输出目录是 ONNX 所在
目录，名称如下：

```text
X5:    yolo11n_bayese_640x640_nv12.bin
S100:  yolo11n_nashe_640x640_nv12.hbm
S100P: yolo11n_nashm_640x640_nv12.hbm
S600:  yolo11n_nashp_640x640_nv12.hbm
```

`--output-dir` 只改变最终制品和编译日志位置。X5 成功时可能留下
`hb_mapper_makertbin.log`，S 成功时可能留下 `hb_compile.log`。使用
`--save-cache` 时，在输出的唯一工作区子目录检查 `config.yaml`、标定目录
和 `bpu_model_output/`。

## 代码流程和兼容符号

维护入口保持为：

```text
export_monkey_patch.py
  ├─ 通用检测器 -> 共用补丁 + Ultralytics 导出
  └─ YOLO26 detect -> yolo26/export_yolo26_detect_bpu.py

mapper.py -> mapper_x5.py / mapper_s.py
          -> --family yolo26 时的 yolo26/mapper_x5.py / mapper_s.py
          -> workflow.py
             inspect_onnx -> calibration_images/select_calibration_images
             -> prepare_calibration -> render_config -> 编译 -> 移动制品/日志
```

`platforms/x5/...` 和 `platforms/s/...` 下的历史路径仍是转发入口，保留旧
类名和命令默认值，不再含有第二套标定或编译实现。Mapper dispatcher 继续
接受既有 `--family`，所以旧分割、姿态、分类命令不会被改送到检测协议。

YOLOv8/YOLO11 检测使用三层 DFL 输出契约；YOLO26 检测使用直接 LTRB，两者
不能互换。导出或编译成功不能证明自定义图满足运行时协议；运行时会在板端
推理前检查输入输出形状和类型。有限的运行时检测协议见
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md)，转换旧新符号对应及
目标适配边界见 [`CONVERSION_CONTRACT.md`](CONVERSION_CONTRACT.md)。

## 故障排查

* **找不到 `hb_mapper` 或 `hb_compile`：** 加载对应 OpenExplore 环境，先
  执行表中的工具检查命令。板端镜像不是编译器环境。
* **ONNX 是动态尺寸、非 float32 或多输入：** 重新导出静态 batch-one、
  NCHW 图。输入契约不明确时 Mapper 会在写标定前停止。
* **没有标定图片：** 直接在 `--cal-images` 中放入 JPG/JPEG/PNG；其他后缀
  会忽略。`--cal-sample false` 保留全部图片，或调整 `--cal-sample-num`。
* **`--platform` 与 `--march` 冲突：** X5 不要传 Nash march；S100/S100P/S600
  分别配 `nash-e`/`nash-m`/`nash-p`。
* **已有制品：** 选择新的 `--output-dir`，或确认替换目的后显式传
  `--overwrite`。
* **编译失败：** 使用 `--save-cache` 重跑，检查唯一工作区的 `config.yaml`、
  标定文件和编译器日志。默认失败工作区也会保留。

本地合并包含规划、配置、入口和运行时契约的主机测试；本地没有执行真实
ONNX 导出或 OpenExplore 编译。板卡验证记录引用发布证据中的既有运行时制品，
不能被解释为本地这次转换已经执行。

<a id="known-gaps"></a>
## 未验证与缺失前提

- 本轮没有实际训练环境、权重导出或 OpenExplore 编译验证；旧发布制品的板测不能替代本地转换验证。
- 未随仓库提供已固定的数据集版本、校准图片集或全部源权重摘要；20–50 张是脚本建议，不是某个已复现实验。默认 `--cal-sample true --cal-sample-num 20`，`--cal-sample false` 使用全部符合扩展名的图片；复现时保存所选文件列表与摘要。
- `--quantized` 默认 int8，可选 int16；`--jobs` 默认 16，`--save-cache` 默认 false。不同工具链的优化选项见 `mapper.py --platform x5 --toolchain-help` 或对应 S 目标；选择器不代替工具链兼容性验证。
- 容器版本链接是原分支保留的环境示例，不代表已核实的最新版本或覆盖所有目标。按实际发布环境保存镜像标识及版本输出。
- 转换得到的模型需在匹配板卡上完成绑定检查、单图 smoke 和必要的数据集/参考数值比较，才能声明该制品已验证；当前新转换板测为 not-run。
