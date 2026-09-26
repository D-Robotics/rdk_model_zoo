# Ultralytics YOLO C++ 运行

[English](README.md) · [Python](../python/README_cn.md)

<a id="supported-boards"></a>
## 实现范围与板卡状态

当前源码实现 X5 packed NV12 `.bin` 与 S100/S100P/S600 split Y/UV `.hbm` 输入，使用模型元数据选择输入/输出协议。C++ 已不是仅 X5 的代码，但接口实现与实板验证是不同维度。本轮没有板端编译、运行或性能测量：以下实现按 supported-not-run 记录，不能用 Python 板测替代 C++ 证据。

| 程序 | 已实现协议 | 限制 |
|---|---|---|
| detect | YOLO26 四通道 LTRB；YOLO11 类 64 通道 DFL | 支持显式 head 选择和 benchmark |
| pose | DFL/LTRB 检测框及对应关键点编码 | 功能参考入口，无 benchmark CLI |
| segment | DFL/LTRB 检测框及对应 mask 系数/prototype | 功能参考入口，无 benchmark CLI |
| classify | 分类 logits | 打印 Top-5，无结果图 |

没有 C++ OBB 入口。这里不声明 S YOLOv10 的 Python NMS-free 语义已被 C++ 覆盖。自定义类别/模型需核对源码中的标签与固定维度，不能仅因扩展名相同就套用。完整 Python 任务入口见上方链接。

<a id="dependencies"></a>
## 依赖

板端构建需要 CMake ≥3.10、C++11 编译器、OpenCV 开发文件和对应板端 DNN SDK。CMake 优先查 `/usr/include/dnn/hb_dnn.h` 和 `/usr/lib`；否则使用 `/usr/include/hobot`、`/usr/include/hobot/dnn`、`/usr/hobot/include`、`/usr/hobot/lib` 并链接 `hbucp`。这些文件由匹配的板端镜像/SDK 提供；主机 OE 编译器不能替代它们。源码没有给全部 SDK 版本作兼容承诺。

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

detect/pose/segment 的前三个位置参数依次为模型、图片、结果路径；classify 只使用模型和图片。pose/segment/classify 没有 Python 风格的 `--platform`、`--model-path` 或通用 `--help`，不要将选项当作位置参数传入。旧内置路径依赖历史目录，建议总是显式传文件路径。

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

这些是独立可执行参考程序，不是与 Python 相同的稳定库 API。`detect/main.cc` 的 `DetectRuntime` 管理模型/张量；`common/dnn_io` 绑定输入，负责 NV12 拷贝及输入 cache clean；执行结束后使输出 cache 可读，再解码/NMS并还原原图坐标。释放张量和模型必须发生在请求完成后。

每个并发 stream 使用独立运行上下文和张量，不能跨未完成请求复用缓冲区。几何、head 探测/解码与 benchmark 统计集中在 `common/`；pose/segment/classify 仍保留各自的主程序和资源流程。集成时核对所有 SDK 返回码、stride 和有效形状，不要只抽取一次推理调用。

<a id="results-interpretation"></a>
## 结果解读与验证

检测框、mask 和关键点绘制到结果图，分类打印类别与概率。确认进程正常退出且输出文件实际存在、内容合理；旧的同名图片不能证明本次成功。按源码内置类别顺序解释 ID，不把单图成功当作全数据集精度。

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

以下四个纯辅助测试不依赖板卡 SDK，只验证解码/head 探测、NV12 几何和 benchmark 统计：

```bash
cmake -S samples/vision/ultralytics_yolo/runtime/cpp/test -B /tmp/ultralytics-cpp-host
cmake --build /tmp/ultralytics-cpp-host
ctest --test-dir /tmp/ultralytics-cpp-host --output-on-failure
```

缺少 OpenCV 开发包或 DNN/UCP 头/库会导致构建失败；请核对目标 SDK，而不是复制其他平台库。输入/输出协议拒绝时检查模型 target、任务、layout、dtype 和 head。主机辅助测试通过不代表所有四个板端程序可编译运行，也不代表新的性能结果。
