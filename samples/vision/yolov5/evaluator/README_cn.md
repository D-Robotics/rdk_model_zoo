[English](README.md) | 简体中文

# YOLOv5 评估器

<a id="dataset"></a>
## 数据集

源提供 X5 的 `test_data/bus.jpg`、S 的 `test_data/kite.jpg` 以及 `coco_classes.names`，没有带标签的 mAP benchmark harness。评估器在相同图片、目标板卡、制品和阈值下比较两套实现的完整运行，是一致性证据工具，不是 mAP 评估器。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── native/  # native 相关文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── compare.py  # Python 脚本
└── source_reference.py  # Python 脚本
```

<a id="environment"></a>
## 环境

在目标板上安装 Python、NumPy、OpenCV 和配套 `hbm_runtime`，准备模型与图片后运行。评估器从固定 Git 版本加载参考实现，所需版本必须存在于本地仓库。

<a id="command"></a>
## 评估命令

在仓库根目录、准备好精确模型且板卡身份已识别后，选择一个全新的空证据目录：

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

工具运行锁定原始实现与本示例实现，把完整原生输入/输出/结果数组保存为 `.npy`，在 `comparison.json` 记录元数据、代码/模型/图片摘要、阈值和板卡身份；所有比较通过才返回 `0`。输出目录必须不存在。在准备好的板卡上按需运行——X5 命令覆盖全部九个变体，S100/S600 命令覆盖 `x-672` 模型（`kite.jpg`）；一次运行验证的是它所执行的工作树。

### 原生 C++ 实现对照（板端）

C++ 对照使用 `evaluator/native/`，在同一板卡、模型、图片和阈值下运行两套实现。参考版本为 X5 `main.cc` 的 `ac115717197920355fc390bb04299b20e6436864`，或 S `src/yolov5.cpp`/`src/main.cpp` 的 `380e1a2bf42041af54be6f34935e50197cfadff9`。工具保存各自的预处理、原始输出和解码检测结果。

在目标板上（仓库已检出、模型已下载；X5 默认
`yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` + `test_data/bus.jpg`，S 默认
x-672 资产 + `test_data/kite.jpg`；两侧阈值一致）：

```bash
# 0) 固定快照必须先取到本地。板上浅克隆不含这些提交；先 fetch（否则
#    instrument.py 会带同样提示 fail closed）：
git fetch --depth=1 origin ac115717197920355fc390bb04299b20e6436864   # x5
git fetch --depth=1 origin 380e1a2bf42041af54be6f34935e50197cfadff9   # s100/s600

# 1) 生成插桩固定源构建（SHA/anchor 失配即拒绝；审计见
#    instrumentation-audit.json——S 的构建闭包（yolov5.hpp + utils/c_utils）
#    一并 SHA 固定并复制进 work dir，构建不引用任何可变工作树文件）。
#    X5 重绑定示例：
python3 samples/vision/yolov5/evaluator/native/instrument.py \
  --target x5 --repo-root . --work-dir /tmp/yolov5-fixed-src \
  --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image-path samples/vision/yolov5/test_data/bus.jpg
#    （S100/S600 在此都传 --target s100，选择共同的 S 源代码；
#    统一入口构建和比较时必须选择实际板卡目标。）

# 2) 编译插桩固定源，经外部 runner 运行。runner 记录真实进程证据（含被
#    gflags 删除参数在内的完整 argv、分离的 stdout/stderr、真实退出码、
#    起止 UTC、cwd、板卡身份、binary/model/image 运行前后 hash、插桩 audit
#    校验）。C++ 观测头只记录进程外不可得的张量/阶段数据，不再 tee
#    stdout/stderr、不再冒充进程退出码。捕获目录必须为空；任何崩溃都会
#    留下 in-progress 标记，比较阶段据此拒绝：
cmake -S /tmp/yolov5-fixed-src -B /tmp/yolov5-fixed-src/build && \
  cmake --build /tmp/yolov5-fixed-src/build -j2
python3 samples/vision/yolov5/evaluator/native/run_capture.py \
  --binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --capture-dir /tmp/yolov5-source-capture \
  --model <模型文件> --image <图片文件> \
  --audit /tmp/yolov5-fixed-src/instrumentation-audit.json \
  -- [--model_path <m> --test_img <i> --label_file <l>]   # S 的参数原样透传

# 3) 统一二进制也经同一 runner 运行（--role unified：统一侧同样获得完整
#    进程证据——真实 argv/退出码与自己的 binary/model/image hash；该角色无
#    插桩 audit）。构建时 -DYOLOV5_TARGET=s100/s100p/s600 须与板卡一致。
python3 samples/vision/yolov5/evaluator/native/run_capture.py --role unified \
  --binary <build>/yolov5_cpp --capture-dir /tmp/yolov5-unified-run \
  --model samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image samples/vision/yolov5/test_data/bus.jpg \
  -- --target x5 --test-img samples/vision/yolov5/test_data/bus.jpg \
     --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
     --dump-dir /tmp/yolov5-unified-dump

# 4) 比较两次同板运行（--target 必须与统一 build_target 完全一致——s100
#    比较拒绝 s100p/s600 构建；s600 构建用 --target s600 比较；两侧都需要
#    各自的 runner 记录）：
python3 samples/vision/yolov5/evaluator/native/compare_native.py \
  --target x5 --repo-root . \
  --source-capture /tmp/yolov5-source-capture --unified-dump /tmp/yolov5-unified-dump \
  --source-binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --unified-binary <build>/yolov5_cpp \
  --unified-run-record /tmp/yolov5-unified-run/run-record.json \
  --output /tmp/yolov5-native-comparison
```

比较要求两侧目标身份、张量数量和形状、数据类型、量化参数、阈值和原图坐标检测结果一致。输入输出、二进制程序、模型和图片按 SHA-256 记录核对。支持均匀行距和精确紧凑布局；阈值和 scale 按 float32 值比较。布局错误或捕获数据缺失会返回错误。复制结果时，完整保留双方进程记录目录、张量文件和 `instrumentation-audit.json`。

比较按两侧记录的物理布局（stride/dtype）还原逻辑数组，因此 padded 布局正确
比较，而未初始化 padding 字节保留在证据（`originals/`）中、不参与比较。固定判据、运行中不得调整：inputs 精确相等；raw 输出
`allclose(atol=1e-5, rtol=0)`；scale/zero-point 描述符精确相等；boxes
`atol=1e-4`、scores `atol=1e-5`、class id 精确相等，比较前做已声明的排序
归一（按 class_id、score、x1..y2）。检测在**模型输入空间**比较，并**强制**比较最终原图坐标（双侧
`detections_original`——捕获端与统一 dump 都会产出）；最终坐标缺失
或单侧缺失直接判整个比较失败，而不是以免责声明放行。任何材料缺失、运行
非零返回码、模型/图片摘要或阈值不一致都以非零返回码失败并保留已收集
证据；原生失败绝不能以空数组通过。这些原生对照步骤在真实板卡上执行
（构建说明见 `runtime/cpp/README_cn.md`）。

<a id="metrics"></a>
## 指标

输入要求精确一致；raw 输出检查 shape/dtype，使用 `rtol=0, atol=1e-5`；结果 boxes 使用 `atol=1e-4`，scores `1e-5`，class IDs 精确相等。X5/S 各自按源协议比较；板端结论以真实 SDK 运行为准。源性能数据列于下方。

<a id="outputs"></a>
## 输出

每个 run 目录含 `legacy_*`、`unified_*` `.npy` 数组和 `comparison.json`。模型预加载失败或比较不一致会保留 error/failed 记录并返回非零，比较不一致时返回非零状态。数组保留全部捕获的输入/输出张量和解码结果字段。

<a id="reference-results"></a>
## 参考结果

源 X5 参考数据表：

| 模型 | 尺寸 | 参数量 | BPU 吞吐 | Python 后处理 |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |


### 板卡性能

以下 COCO 检测器数据分别列出 BPU 执行与 Python 后处理耗时，每项注明线程数。X3 数据仅为历史参考；当前运行板卡以 Sample 支持矩阵为准。

### RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | 参数量 | BPU吞吐量 | 后处理时间 |
|---------|---------|-------|---------|---------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 106.8 FPS | 2.3 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 M | 45.2 FPS | 2.3 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 M | 21.8 FPS | 2.3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 12.3 FPS | 2.3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 277.2 FPS | 2.3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 124.2 FPS | 2.3 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 M | 48.4 FPS | 2.3 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 M | 23.3 FPS | 2.3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 13.1 FPS | 2.3 ms |

### RDK X3 & RDK X3 Module（历史性能记录）
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | 参数量 | BPU吞吐量 | 后处理时间 |
|---------|---------|-------|---------|---------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 38.2 FPS | 3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 3.9 FPS | 3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 37.2 FPS | 3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 20.9 FPS | 3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 3.6 FPS | 3 ms |

注: 详细性能数据见文末.


## 补充参考数据

各表按发布时的模型、板卡和测量条件列出，不同配置的数据分别保留。

### RDK X5 & RDK X5 Module

| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | BPU延迟/BPU吞吐量(线程) |  后处理时间 |
|-----|----------|-------|----------|------------------------|----------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5  | 13.0 ms / 76.6 FPS (1 thread  ) <br/> 16.0 ms / 124.8 FPS (2 threads) | 2.3 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 | 23.9 ms / 41.7 FPS (1 thread  ) <br/> 37.7 ms / 52.9 FPS (2 threads) | 2.3 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 | 44.0 ms / 22.7 FPS (1 thread  ) <br/> 78.2 ms / 25.5 FPS (2 threads) | 2.3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 | 74.1 ms / 13.5 FPS (1 thread  ) <br/> 137.6 ms / 14.5 FPS (2 threads) | 2.3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 | 8.1 ms / 122.7 FPS (1 thread  ) <br/> 8.6 ms / 232.3 FPS (2 threads) <br/> 9.7 ms / 307.9 FPS (3 threads) | 2.3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 | 11.9 ms / 83.8 FPS (1 thread  ) <br/> 13.7 ms / 145.3 FPS (2 threads) | 2.3 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 | 22.7 ms / 44.0 FPS (1 thread  ) <br/> 35.3 ms / 56.6 FPS (2 threads) | 2.3 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 | 41.6 ms / 24.0 FPS (1 thread  ) <br/> 73.1 ms / 27.3 FPS (2 threads) | 2.3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 | 69.7 ms / 14.4 FPS (1 thread  ) <br/> 129.0 ms / 15.5 FPS (2 threads) | 2.3 ms |

<a id="boundaries"></a>
## 适用范围

运行评估器前准备模型。X5 将 XYXY 框传给按 XYWH 解释输入的 OpenCV `NMSBoxes`；S 使用按类 XYXY NMS。比较时使用同一目标板卡与模型。

请完整保留双方捕获目录、`instrumentation-audit.json` 和 stdout/stderr 日志。比较工具会核对这些文件，并将它们复制到输出目录的 `originals/` 中。


### X3 与 X3 Module COCO 测量记录（中文来源）

[Source measurement table](https://github.com/D-Robotics/rdk_model_zoo/blob/cb86079ae5befcef9ca50fb46c8a6d8980106dec/samples/vision/yolov5/README_cn.md).

该来源的五行后处理均为 3 ms，上方英文来源表为 13 ms，两份完整记录分别标注来源。条件为 X3 / X3 Module、COCO 检测、4 × A53@1.8 GHz 且全核心 performance 调度、2 × Bernoulli2@1.0 GHz。单线程延迟是单 BPU 核心的一个任务耗时，多线程 FPS 是排队任务的吞吐量。此处 X3 测量记录不表示当前运行时支持 X3。

| 模型 | 尺寸(像素) | 类别数 | 参数量(M) | 浮点精度<br/>(mAP:50-95) | 量化精度<br/>(mAP:50-95) | BPU延迟/BPU吞吐量(线程) |  后处理时间 |
|---------|---------|-------|---------|---------|----------|--------------------|--------------------|
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | - | - | 55.7 ms / 17.9 FPS(1 thread) <br/> 61.1 ms / 32.7 FPS(2 threads) <br/> 78.1 ms / 38.2 FPS(3 threads)| 3 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | - | - | 512.4 ms / 2.0 FPS(1 thread) <br/> 519.7 ms / 3.8 FPS(2 threads) <br/> 762.1 ms / 3.9 FPS(3 threads) | 3 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 28.0 | - | 85.4 ms / 11.7 FPS(1 thread) <br/> 88.9 ms / 22.4 FPS(2 threads) <br/> 121.9 ms / 32.7 FPS(4 threads) <br/> 213.0 ms / 37.2 FPS(8 threads) | 3 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 37.4 | - | 175.4 ms / 5.7 FPS(1 thread) <br/> 182.3 ms / 11.0 FPS(2 threads) <br/> 217.9 ms / 18.2 FPS(4 threads) <br/> 378.0 ms / 20.9 FPS(8 threads) | 3 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 50.7 | - | 1021.5 ms / 1.0 FPS(1 thread) <br/> 1024.3 ms / 2.0 FPS(2 threads) <br/> 1238.0 ms / 3.1 FPS(4 threads)<br/> 2070.0 ms / 3.6 FPS(8 threads) | 3 ms |
