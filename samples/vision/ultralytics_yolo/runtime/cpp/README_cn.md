[English](README.md) | 简体中文

# Ultralytics YOLO C++ 运行时

[Python](../python/README_cn.md)

<a id="overview"></a>
## C++ 推理

在匹配的开发板上运行受支持的 Ultralytics 检测、分割、姿态、分类或旋转框
模型。单个可执行程序覆盖五个任务（`--task detect|segment|pose|classify|obb`）；
原生后端栈和 packed/split NV12 输入协议均由模型元数据自动选择。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── CMakeLists.txt  # 板端构建（C++17，YOLO_TARGET=auto|x5|s100|s100p|s600）
├── run.sh          # 树外构建并启动，转发全部参数
├── inc/
│   ├── yolo.hpp            # 不依赖 SDK 的词汇层：解码数学、计划、任务类型
│   ├── backend.hpp         # 共享 DNN 后端：栈探测、推理、RAII 所有者
│   ├── cli.hpp             # 内联选项解析、benchmark 数学与入口
│   └── imagenet_labels.hpp # 1000 项 ImageNet 标签顺序（分类）
├── src/
│   ├── main.cpp     # 构造任务模型、调用 predict、交给报告层
│   ├── cli.cpp      # 图片加载、渲染、benchmark 驱动
│   ├── backend.cpp  # SDK 生命周期：infer_sync、输入探测、NV12 上传
│   ├── detect.cpp   # YoloDetect 模型（LTRB + DFL 头）
│   ├── segment.cpp  # YoloSegment 模型（mask、原型）
│   ├── pose.cpp     # YoloPose 模型（17 个 COCO 关键点）
│   ├── classify.cpp # YoloClassify 模型（1000 logits Top-K）
│   └── obb.cpp      # YoloObb 模型（直接 LTRB + 角度、旋转 NMS）
└── test/  # 主机单元测试（ctest；test/fake_dnn_io 提供 X5/UCP SDK 替身）
```

<a id="supported-boards"></a>
## 支持的协议

运行时从模型元数据选择协议：X5 使用 packed NV12 `.bin` 输入，
S100/S100P/S600 使用 split Y/UV `.hbm` 输入。每个板端目标构建一次，
同一份源码可在两种栈上编译。

| 任务 | 实现的协议 | 限制 |
|---|---|---|
| detect | YOLO26 四通道 LTRB；YOLO11 系 64 通道 DFL | 支持显式指定 head |
| pose | DFL/LTRB 框及对应关键点编码 | 功能参考 |
| segment | DFL/LTRB 框及 mask 系数/原型 | 功能参考 |
| classify | 单个未量化 FLOAT32 1000 logits 向量 | 按步长取 Top-5，无结果图 |
| obb | YOLO26 直接 LTRB 框加逐单元角度，九个输出 | 方形 stride-32 输入，DOTA 类别（默认 15） |

自定义类别/模型布局必须与所选任务的标签和维度匹配。YOLOv10 无 NMS 任务
接口请使用链接的 Python 入口。

<a id="dependencies"></a>
## 依赖

板端构建需要 CMake ≥3.16、C++17 编译器、OpenCV 开发文件和匹配的板端
DNN SDK。`YOLO_TARGET=auto`（默认）先检查 `/usr/include/dnn/hb_dnn.h` 与
`/usr/lib`；否则使用 `/usr/include/hobot`、`/usr/include/hobot/dnn`、
`/usr/hobot/include`、`/usr/hobot/lib` 并链接 `hbucp`。可用
`-DYOLO_TARGET=x5|s100|s100p|s600` 显式指定；构建过程绝不读取主机 SoC。
这些文件来自匹配的板端镜像/SDK，请使用目标镜像对应的 SDK 版本。主机测试
另需 CTest ≥3.20（`ctest --test-dir`）。

<a id="build"></a>
## 构建

在目标板上的 sample 根目录（`samples/vision/ultralytics_yolo`）运行。
`run.sh` 在 `runtime/cpp/build/` 下配置树外构建、完成构建并执行二进制；
所有参数原样转发：

```bash
bash runtime/cpp/run.sh --task detect model/yolo26n_detect_bayese_640x640_nv12.bin \
  test_data/bus.jpg /tmp/cpp-result.jpg
```

等价的手动构建：

```bash
cmake -S runtime/cpp -B /tmp/ultralytics-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ultralytics-cpp -j2
/tmp/ultralytics-cpp/ultralytics_yolo_cpp --task detect ...
```

<a id="run"></a>
## 运行

每个任务都有各自的模型、图片和结果路径默认值，在 sample 根目录下仅带
`--task` 即可运行参考流程。以下 X5 路径显式准备模型并输出检测图：

```bash
bash model/download_model.sh --platform x5 --family yolo26 --task detect --model-size n
bash runtime/cpp/run.sh --task detect \
  model/yolo26n_detect_bayese_640x640_nv12.bin \
  test_data/bus.jpg /tmp/cpp-result.jpg
```

其他任务请先按[模型说明](../../model/README_cn.md)准备对应制品，再传入
实际路径。S 需要匹配 march 的 `.hbm` 文件；重命名 X5 文件不能完成转换。

```bash
runtime/cpp/run.sh --task classify /models/classification.bin test_data/zebra_cls.jpg
runtime/cpp/run.sh --task pose /models/pose.bin test_data/bus.jpg /tmp/cpp-pose.jpg
runtime/cpp/run.sh --task segment /models/segmentation.bin test_data/bus.jpg /tmp/cpp-segment.jpg
runtime/cpp/run.sh --task obb /models/obb.bin test_data/dota.jpg /tmp/cpp-obb.jpg
```

<a id="parameters"></a>
## 参数

所有任务都接受位置参数 `model`、`image` 和（classify 除外）`result` 路径，
以及下表中的 kebab-case 选项，拼写与 Python CLI 一致。detect/pose/segment
绘制结果图；classify 仅打印 Top-K。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--task` | `detect` | 任务运行时：detect、segment、pose、classify 或 obb |
| `--head` | `auto` | auto/dfl/ltrb，仅 detect |
| `--score-thres` | `0.25` | 分数阈值（内部按原始 logit 比较） |
| `--nms-thres` | detect `0.7`，segment/pose `0.45`，obb `0.2` | NMS IoU 阈值 |
| `--kpt-conf-thres` | `0.5` | 姿态关键点置信度 |
| `--topk` | `5` | 分类 Top-K |
| `--classes` | `15` | OBB 类别通道数 |
| `--angle-sign` | `1` | OBB 角度符号约定 |
| `--angle-offset` | `0` | OBB 角度偏移（度） |
| `--no-regularize` | `false` | 保留未正则化的 OBB 框 |
| `--resize-type` | `1` | 0 拉伸，1 letterbox（所有任务） |
| `--opencv-threads` | `all` | 0/all 使用在线 CPU 数；可指定正整数 |
| `--benchmark` | `false` | 有限轮次端到端测量（所有任务） |
| `--warmup` | `20` | 每轮预热帧数 |
| `--runs` | `200` | 每轮计时帧数 |
| `--rounds` | `3` | 轮次数 |
| `--pipeline-streams` | `1` | 1 或 2 条独立的完整并发流水线 |
| `--json` | `empty` | 汇总 benchmark JSON 路径 |
| `--runtime-source-sha256` | `empty` | 记入 benchmark JSON 的来源哈希 |
| `--executable-sha256` | `empty` | 记入 benchmark JSON 的可执行文件哈希 |
| `--no-save` | `false` | 跳过验证结果绘制与保存 |
| `--help / -h` | `false` | 打印 CLI 帮助 |

省略位置参数时的各任务默认值：detect 使用
`model/yolo26n_detect_bayese_640x640_nv12.bin`、`test_data/bus.jpg` 和
`cpp_result.jpg`；segment 与 pose 使用对应的
`source/reference_bin_models/{seg,pose}/yolo11n_*_bayese_640x640_nv12.bin`，
输出分别为 `segment_result.jpg`/`pose_result.jpg`；classify 使用
`source/reference_bin_models/cls/yolo11n_cls_bayese_224x224_nv12.bin` 和
`test_data/zebra_cls.jpg`；obb 使用 `yolo26n_obb_640x640_nv12.bin`、
`test_data/dota.jpg` 和 `obb_result.jpg`。路径相对调用目录解析。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

这是参考可执行程序加一套可读的模型 API，不是与 Python 相同的稳定库契约。
`main.cpp` 用 `Config` 构造具名模型（`yolo::YoloDetect`、`YoloSegment`、
`YoloPose`、`YoloClassify`、`YoloObb`），填充调用方拥有的 `Input`（原始 BGR 像素及
源图几何），调用 `predict`，再把返回的 `Prediction` 交给 CLI 报告层。每个
模型自持生命周期：构造函数加载运行时（模型句柄、输入协议探测、输出绑定），
`preprocess → infer → postprocess` 是模型自身的阶段，每次调用拥有自己的
输入/输出缓冲区——没有隐藏的"上次调用"状态。CLI 负责选项、图片加载、渲染
和 benchmark。

每个并发 benchmark stream 使用独立运行上下文和张量，不能跨未完成请求复用
缓冲区。共享 DNN 后端（`inc/backend.hpp` + `src/backend.cpp`）拥有 SDK 接口
面：X5/UCP 栈探测、`infer_sync`、NV12 输入所有者、`PackedModelOwner`/
`OutputTensorOwner` 以及按步长校验的输出读取器。它同时向样本外消费者
（ASR/Paraformer）暴露 `infer_tensors_sync`，用于超过两个已校验输入张量的
场景；面向图像的 `infer_sync` 保持一/二输入限制。集成时核对所有 SDK 返回码、
stride 和有效形状，不要只抽取一次推理调用。

<a id="classification-contract"></a>
## 分类输出契约

分类运行时只接受一个模型和一个输出：1000 个有限 FLOAT32 logits，量化类型
为 `NONE`。支持一维 `(1000,)`，或 rank 2–4、batch 为 1、仅一个轴为 1000、
其余轴均为 1 的形状，例如 `(1,1000)`、`(1,1000,1,1)`、`(1,1,1,1000)`。
不再固定将 axis 1 当成类别轴；其他类别数、整数输出、展开的空间图和
batch>1 均明确拒绝。

物理分配大小必须为正。X5 从 aligned shape 推导类别步长，S 读取字节
stride；跳过填充区域，并在读取前确认最后一类仍位于分配范围内。稳定
softmax 使用双精度累加，Top-K 按概率降序，概率完全相等时按类别 ID 升序。
标签遵循 `inc/imagenet_labels.hpp` 中的 1000 项
ImageNet 顺序，分类数学位于
`inc/yolo.hpp`。不执行手动反量化。

模型和输出资源由所有者在提前返回或 C++ 异常时释放；分配失败、输出缓存
失效操作失败时停止解码。描述符错误或非有限 logits 会带诊断信息非零退出。

分类前处理默认仍是 **letterbox、灰色 127 填充**；Python YOLO26 分类默认
stretch，Python S 分类也默认 stretch。C++ 侧传 `--resize-type 0` 即可切换
为 stretch，无需重新构建；请勿将 C++ 默认行为直接当作 Python 的等价基线。

<a id="pose-segment-output-contract"></a>
## 姿态与分割输出契约

两者要求输入为边长可被 32 整除的方形、batch=1、未量化 FLOAT32 NHWC 输出。
按空间形状和通道数匹配角色，不依赖 SDK 输出顺序：stride 8/16/32 各有分类
（姿态 1 类、分割 80 类）、框（4 通道直接 LTRB 或 64 通道 DFL）、附加信息
（姿态 51 个关键点值、分割 32 个 mask 系数）。三个尺度必须使用同一框编码。
分割还需要唯一的 stride-4、32 通道 NHWC 原型。缺失/重复角色、混合编码、
整数或 SCALE 张量、不支持的布局和非有限值均明确拒绝。C++ 原型路径**不支持**
Python 可接受的 NCHW 原型；请选择对应 NHWC 导出，或使用 Python 入口。

按 X5 aligned shape 或 S 字节 stride 读取。缓存失效操作成功后，将有效值
复制到独立拥有内存的紧凑 NHWC 向量，跳过填充；这会增加一份有效输出的临时
主机副本。分配或缓存操作失败会中止，所有退出路径释放已取得的输出和模型
资源。DFL 和直接距离数学位于 `inc/yolo.hpp`（`decode_box_dfl`、
`decode_box_ltrb`），不再维护私有副本；关键点公式、NMS 和绘图策略保持
原有行为。

这两个任务仍是功能参考，行为比 Python 窄：两者都会丢弃越过模型输入边界
的框。分割采用不区分类别的 NMS，输出的是**模型输入尺寸**的三联图（框、
彩色 mask、叠加图），总宽为 `3 * input_width`。姿态使用原有缩放/填充运算
在原图绘制。结果图保存失败会非零退出。

<a id="obb-output-contract"></a>
## 旋转框输出契约

旋转框运行时（`--task obb`）要求输入为边长可被 32 整除的方形、batch=1，
并恰好有九个未量化 FLOAT32 NHWC 输出：stride 8/16/32 各有分类图
（`--classes`，默认 15 个 DOTA 类别）、4 通道直接 LTRB 框图和 1 通道角度图。
按空间形状和通道数匹配角色；类别数为 1 或 4 会与角度/框角色冲突而被拒绝，
此类布局请显式传 `--classes`。解码遵循 `runtime/python/obb.py`：
以单元中心为基准的 LTRB 距离乘 stride，角度按 `--angle-sign` 缩放并按
`--angle-offset`（度）平移。默认对框做正则化（宽 ≥ 高、角度回绕）；
`--no-regularize` 保留原始几何。

平台策略同样与 Python 入口一致：X5 将角度回绕到 [-π/2, π/2)，执行逐类别
贪心旋转 NMS（`--nms-thres`，默认 0.2），并把还原后的框裁剪到源图内；
S 系列执行不区分类别的旋转 NMS，保留未裁剪几何。letterbox 还原使用实际
整数取整后的缩放比，而不是理想比例。解码出现非有限值会显式失败；
`--angle-sign`/`--angle-offset` 必须为有限值。

<a id="results-interpretation"></a>
## 结果与验证

框、mask 和关键点绘制进结果图；分类打印类别和概率。检查进程返回码并查看
输出图。分类 ID 索引模型内置类别顺序。数据集评分见 evaluator 指南。

有限轮次 benchmark，所有任务可用：

```bash
runtime/cpp/run.sh --task detect \
  model/yolo26n_detect_bayese_640x640_nv12.bin test_data/bus.jpg /tmp/cpp-result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score-thres 0.25 --nms-thres 0.7 --no-save --json /tmp/yolo-e2e-cpp.json
```
计时从内存中的 BGR 图像开始，到还原坐标的结果结束：包含
resize/letterbox、NV12 转换、拷贝/缓存操作、BPU 和解码/NMS；不包含模型
加载、文件 I/O、绘图和保存。`--pipeline-streams 2` 表示两条完整流水线。
吞吐率为共享墙上时间内的总完成帧数；延迟按单请求统计。OpenCV 线程数与
stream 数无关，不做 CPU 亲和绑定。C++/Python、runtime-only/端到端、单/多
stream 的测量请分开记录。

进程只执行一次验证 predict，且每条 stream 恰好持有一个运行上下文
（stream 0 复用为验证构造的模型）。JSON 记录按任务报告 `output_kind`
（`detections`、`instance_masks`、`pose_instances`、`topk_predictions`、
`rotated_boxes`）及 `outputs_per_frame`，在适用时记录 `resize_type`，
旋转框记录角度选项，检测类任务记录 score/NMS。
`--runtime-source-sha256`/`--executable-sha256`（64 位十六进制）将可选的
来源信息写入同一记录。

共享输入所有者在分配前先校验元数据。X5 packed NV12 仅接受物理形状紧凑的
RGB 形 NCHW/NHWC 描述符，显式拒绝带填充的 packed 存储；split Y/UV 接受
精确 batch-one 几何和未量化字节平面。动态 `-1` stride/容量按实际行距
解析；零/重叠 stride、分配不足和溢出均被拒绝。`upload_planes` 接收精确
长度的紧凑 Y 和交错 UV 缓冲区，写入分配的行距并 clean 两个缓存。原有
I420 `upload` 委托到该路径。分配成功前上传或计划不匹配都会失败；部分
分配会释放全部已取得的缓冲区。

同步推理要求非空任务句柄，在创建/提交/等待失败时释放已返回的任务并保留
首个错误码。UCP 提交选择 `HB_UCP_BPU_CORE_ANY`。请在目标板上用匹配的
DNN/UCP SDK 头文件和库构建。如果模型绑定拒绝了某个制品，请核对它的目标
板、任务、布局、dtype 和输出头与所选程序是否匹配。

共享的 `OutputTensorOwner` 在 SDK 分配返回错误但地址非空时保留该缓冲区
以便清理，并在返回成功但地址为空时判定失败。
