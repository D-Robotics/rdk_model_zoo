# Ultralytics YOLO C++ 运行

[English](README.md) · [Python](../python/README_cn.md)

<a id="overview"></a>
## C++ 推理

本目录提供C++ 推理所需的程序与操作说明。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── classify/  # classify 相关文件
├── common/  # common 相关文件
├── detect/  # detect 相关文件
├── pose/  # pose 相关文件
├── segment/  # segment 相关文件
├── test/  # test 相关文件
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="supported-boards"></a>
## 实现范围与板卡状态

当前 C++ 入口依据模型元数据选择协议：X5 使用 packed NV12 `.bin` 输入，S100/S100P/S600 使用 split Y/UV `.hbm` 输入。各任务的目标板构建与运行方法见下文。

| 程序 | 已实现协议 | 限制 |
|---|---|---|
| detect | YOLO26 四通道 LTRB；YOLO11 类 64 通道 DFL | 支持显式 head 选择和 benchmark |
| pose | DFL/LTRB 检测框及对应关键点编码 | 功能参考入口，无 benchmark CLI |
| segment | DFL/LTRB 检测框及对应 mask 系数/prototype | 功能参考入口，无 benchmark CLI |
| classify | 单个未量化 FLOAT32、1000 类 logits 向量 | 按物理 stride 读取 Top-5，无结果图 |

自定义类别/模型需符合所选 C++ 任务使用的标签和维度。YOLOv10 NMS-free 任务入口见上方 Python 链接。

<a id="dependencies"></a>
## 依赖

下列命令需要 CMake/CTest ≥3.20（包括 `ctest --test-dir`）。板端构建需要C++11 编译器、OpenCV 开发文件和对应板端 DNN SDK。CMake 优先查 `/usr/include/dnn/hb_dnn.h` 和 `/usr/lib`；否则使用 `/usr/include/hobot`、`/usr/include/hobot/dnn`、`/usr/hobot/include`、`/usr/hobot/lib` 并链接 `hbucp`。这些文件由匹配的板端镜像/SDK 提供；主机 OE 编译器不能替代它们。

<a id="build"></a>
## 构建

从仓库根目录在目标板上执行。每个任务各有 CMakeLists.txt，本层没有统一 CMake 工程或 run.sh。

```bash
for task in detect classify pose segment; do
  cmake -S "samples/vision/ultralytics_yolo/runtime/cpp/$task" \
    -B "/tmp/ultralytics-cpp-$task" -DCMAKE_BUILD_TYPE=Release
  cmake --build "/tmp/ultralytics-cpp-$task" -j2
done
```

<a id="run"></a>
## 运行

以下为 X5 的显式模型准备到检测结果流程，工作目录仍为仓库根目录：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolo26 --task detect --model-size n
/tmp/ultralytics-cpp-detect/ultralytics_yolo_detect \
  samples/vision/ultralytics_yolo/model/yolo26n_detect_bayese_640x640_nv12.bin \
  samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-result.jpg
```
其余功能命令如下。先按 [模型说明](../../model/README_cn.md) 准备对应任务制品，把 `/models/*.bin` 替换为实际路径；S 使用匹配 march 的 `.hbm`，不要把 X5 文件改名使用。

```bash
/tmp/ultralytics-cpp-classify/ultralytics_yolo_classify \
  /models/classification.bin samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg
/tmp/ultralytics-cpp-pose/ultralytics_yolo_pose \
  /models/pose.bin samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-pose.jpg
/tmp/ultralytics-cpp-segment/ultralytics_yolo_segment \
  /models/segmentation.bin samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-segment.jpg
```

<a id="parameters"></a>
## 参数

detect/pose/segment 的前三个位置参数依次为模型、图片、结果路径；classify 只使用模型和图片。pose/segment/classify 没有 Python 风格的 `--platform`、`--model-path` 或通用 `--help`，不要将选项当作位置参数传入。请显式传入模型、图片和结果文件路径。

只有 **detect** 接受以下选项：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--head` | `auto` | auto/dfl/ltrb，仅 detect |
| `--score` | `0.25` | 检测分数阈值 |
| `--nms` | `0.7` | 检测 NMS IoU，不采用 Python S 默认值 |
| `--resize-type` | `1` | 0 拉伸，1 letterbox |
| `--benchmark` | `false` | 有限轮次端到端测量 |
| `--warmup` | `20` | 每轮预热帧数 |
| `--runs` | `200` | 每轮计时帧数 |
| `--rounds` | `3` | 轮次数 |
| `--pipeline-streams` | `1` | 独立的完整并发流水线数量 |
| `--opencv-threads` | `0` | 0/all 使用在线 CPU 数；可指定正整数 |
| `--json` | `empty` | 汇总 benchmark JSON 路径 |
| `--no-save` | `false` | 跳过验证结果绘制与保存 |
| `--help / -h` | `false` | 打印 detect 帮助 |

pose/segment 的源码阈值为 score=0.25、NMS=0.45；pose 点阈值 0.5，classify Top-K=5；这些不是可用的 CLI 参数。detect 默认模型/图片/输出为 `yolo26n_detect_bayese_640x640_nv12.bin`、`bus.jpg`、`cpp_result.jpg`，相对调用目录。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

这些是独立可执行参考程序，不是与 Python 相同的稳定库 API。`detect/main.cc` 的 `DetectRuntime` 管理模型/张量；`common/dnn_io` 绑定输入，负责 NV12 拷贝及输入 cache clean；执行结束后使输出 cache 可读，再解码/NMS；detect/pose 还原原图坐标，segment 在模型输入空间绘图（见下文）。释放张量和模型必须发生在请求完成后。

每个并发 stream 使用独立运行上下文和张量，不能跨未完成请求复用缓冲区。几何、head 探测/解码与 benchmark 统计集中在 `common/`；pose/segment/classify 仍保留各自的主程序和资源流程。集成时核对所有 SDK 返回码、stride 和有效形状，不要只抽取一次推理调用。

<a id="classification-contract"></a>
## 分类输出契约

分类入口只接受一个模型和一个输出：1000 个有限 FLOAT32 logits，量化类型为
`NONE`。支持一维 `(1000,)`，或 rank 2–4、batch 为 1、仅一个轴为 1000、其余轴
均为 1 的形状，例如 `(1,1000)`、`(1,1000,1,1)`、`(1,1,1,1000)`。
不再固定将 axis 1 当成类别轴；其他类别数、整数输出、展开的空间图和 batch>1
均明确拒绝。

物理分配大小必须为正。X5 从 aligned shape 推导类别步长，S 读取字节 stride；
跳过填充区域，并在读取前确认最后一类仍位于分配范围内。稳定 softmax 使用
双精度累加，Top-5 按概率降序，概率完全相等时按类别 ID 升序。这明确了平局
规则，可能与历史代码未指定的顺序不同。原源码的 1000 项 ImageNet 标签顺序
保存在 `common/imagenet_labels.h`，分类数学位于 `common/classification.h`。
不执行手动反量化。

模型和输出资源由所有者在提前返回或 C++ 异常时释放；分配失败、输出缓存失效
操作失败时停止解码。描述符错误或非有限 logits 会带诊断信息非零退出。

前处理仍保留原 C++ 的 **letterbox、灰色 127 填充**。程序不从制品名称推断
YOLO 家族，也没有 resize 参数；Python YOLO26 分类默认 stretch，Python S 分类
也默认 stretch。因此不能将 C++ 默认行为当成 Python 的等价精度基线。如果模型
评估配方要求 stretch，请将 `PREPROCESS_TYPE` 改为 `RESIZE_TYPE` 后重新构建，
并在结果中记录该选择。此处没有新增数据集精度结论。

<a id="pose-segment-output-contract"></a>
## 姿态与分割输出契约

两者要求输入为边长可被 32 整除的方形、batch=1、未量化 FLOAT32 NHWC 输出。
按空间形状和通道数匹配角色，不依赖 SDK 输出顺序：stride 8/16/32 各有分类
（姿态 1 类、分割 80 类）、框（4 通道直接 LTRB 或 64 通道 DFL）、附加信息
（姿态 51 个关键点值、分割 32 个 mask 系数）。三个尺度必须使用同一框编码。
分割还需要唯一的 stride-4、32 通道 NHWC 原型。缺失/重复角色、混合编码、整数或
SCALE 张量、不支持的布局和非有限值均明确拒绝。C++ 原型路径**不支持** Python
可接受的 NCHW 原型；请选择对应 NHWC 导出，或使用 Python 入口。

按 X5 aligned shape 或 S 字节 stride 读取。缓存失效操作成功后，将有效值复制到
独立拥有内存的紧凑 NHWC 向量，跳过填充；这会增加一份有效输出的临时主机副本。
分配或缓存操作失败会中止，所有退出路径释放已取得的输出和模型资源。
DFL 和直接距离数学复用 `common/decode.h`，不再维护私有副本；关键点公式、NMS
和绘图策略保持原有行为。

这两个程序仍是功能参考，行为比 Python 窄：两者都会丢弃越过模型输入边界的框。
分割采用不区分类别的 NMS，输出的是**模型输入尺寸**的三联图（框、彩色 mask、
叠加图），总宽为 `3 * input_width`，不返回 Python 的原图 ROI mask。
姿态使用原有缩放/填充运算在原图绘制。共用张量传输不能证明 Python 等价性或
数据集精度；保存图片失败现在会非零退出。

<a id="results-interpretation"></a>
## 结果解读与验证

检测框、mask 和关键点绘制到结果图，分类打印类别与概率。确认进程正常退出且输出文件实际存在、内容合理；按源码内置类别顺序解释 ID，数据集精度按评估文档测量。

有限轮次检测 benchmark：

```bash
/tmp/ultralytics-cpp-detect/ultralytics_yolo_detect \
  samples/vision/ultralytics_yolo/model/yolo26n_detect_bayese_640x640_nv12.bin \
  samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save --json /tmp/yolo-e2e-cpp.json
```
计时从内存中的 BGR 图片开始，到还原后的检测结果结束：包括 resize/letterbox、NV12、拷贝/cache、BPU 与解码/NMS，不包括模型加载、图片文件读取、绘图和保存。`--pipeline-streams 2` 可测两条完整流水线；吞吐量是总完成帧数/共同墙钟时间，延迟是每请求值。OpenCV 线程数和流水线数相互独立，不限定 CPU affinity。不要将 C++ 与 Python、runtime-only 与端到端、单流与多流数据混成同一结论。

主机套件包含六个纯辅助测试（解码、head 探测、NV12 几何、benchmark 统计、分类、
任务输出绑定），四个分类/任务描述符与资源测试，以及两个输入/任务生命周期测试，
共 12 项。后两项使用精简 X5/UCP 替身，并以 AddressSanitizer 与
UndefinedBehaviorSanitizer 检查生产代码；均不使用真实板卡 SDK：

```bash
cmake -S samples/vision/ultralytics_yolo/runtime/cpp/test -B /tmp/ultralytics-cpp-host
cmake --build /tmp/ultralytics-cpp-host
ctest --test-dir /tmp/ultralytics-cpp-host --output-on-failure
```

共用输入对象在分配前校验元数据。X5 packed NV12 接受 RGB 形状的 NCHW/NHWC
描述，但要求物理形状紧凑；带填充的 packed 存储显式拒绝。Split Y/UV 要求精确的
单 batch 几何及无量化字节平面。动态 `-1` 跨度/容量按实际行跨度推导；零跨度、
重叠、容量不足或溢出均拒绝。`upload_planes` 接收长度精确的紧凑 Y 与交错 UV，
按已分配行跨度拷贝并清理两平面 cache；原有 I420 `upload` 委托同一路径。
未完成分配或使用不同 plan 时拒绝上传；中途分配失败会释放所有已获取缓冲区。

同步推理要求非空任务句柄；创建、提交或等待报错时仍释放返回的任务，并保留首个
错误码。UCP 提交选择 `HB_UCP_BPU_CORE_ANY`。在目标板上使用匹配的 DNN/UCP SDK
头文件和库进行构建。模型绑定失败时，核对 artifact 的 target、任务、layout、dtype
和输出 head 是否符合所选程序。

共用 `OutputTensorOwner` 在分配返回错误但已给出非空地址时保留缓冲区以供清理，
并拒绝“返回成功但地址为空”的结果。

共享 `common/dnn_io.h` 另提供 `infer_tensors_sync`，供 Paraformer 等超过两个且
已完整核验输入张量的调用方复用。它仅执行同步 SDK 提交／等待／释放，调用方负责按
模型元数据核验并持有完整数组。图像入口 `infer_sync` 仍限定一／两个输入，因此不会
扩展 YOLO 的图像输入协议。
