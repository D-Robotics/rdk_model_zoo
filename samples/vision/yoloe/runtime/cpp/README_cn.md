[English](README.md) | 简体中文

# YOLOE C++ Runtime


使用 `run.sh` 执行 E11/E26 无提示实例分割，或将 C++ 三阶段库嵌入应用。启动器精确选择模型，核验本机和文件身份，按显式要求构建原生程序，并保留日志、图片和掩码结果。真实 SDK 编译与板端推理按下文构建/运行说明执行。Python 入口见 [Python runtime](../python/README_cn.md)，注意其 X5 掩码协议不同。

<a id="overview"></a>
## C++ 推理

使用匹配的 X5 或 S 模型运行免提示 YOLOE 实例分割。程序解码候选目标、还原掩码，并写入预测结果与可视化文件。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── inc/  # 模型与 CLI 头文件：detect.hpp、cli.hpp
├── src/  # 五文件运行时：detect.cpp、cli.cpp、main.cpp
├── tests/  # 自动化测试
├── CMakeLists.txt  # 构建配置
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── launcher.py  # Python 脚本
└── run.sh  # 运行示例
```

<a id="supported-boards"></a>
## 支持的板卡

| 目标 | 变体 / 默认值 | 模型前提 | 支持 |
| --- | --- | --- | --- |
| X5 | 11s/m/l；默认 11s | 已发布的浮点输出 BIN，匹配 X5 SDK | 支持 |
| S100 | 11s、26n/s/m/l/x；默认 11s | 自行转换的浮点输出 HBM，显式提供 SHA-256 | 支持 |
| S100P | 26n/s/m/l/x；默认 26n | 自行转换的浮点输出 HBM，显式提供 SHA-256 | 支持 |
| S600 | 无 | 无发布路线，显式拒绝 | 不支持 |

14 个发布身份用于模型选择。S 已发布文件的输出为量化数据，本入口加载浮点输出；通过[转换流程](../../conversion/README_cn.md)准备 S 浮点制品，文件命名需与实际契约一致。

## 选择、构建与运行

下面命令从仓库根目录执行。启动器需要 Python 3.10+ 和 NumPy，本原生入口不需要 Python OpenCV；用 `PYTHON` 指定 `run.sh` 使用的解释器。脚本也支持从其他目录调用。用户传入的相对路径及默认输出位置相对于调用目录解析；子进程在仓库根目录运行。

以下命令可在主机执行，仅查看当前清单和准备命令，不下载、不构建、不检查板卡、不执行推理：

```bash
bash samples/vision/yoloe/runtime/cpp/run.sh --list-models
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s --dry-run
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n --dry-run
```

Dry-run 必须显式指定 target，`executed`、`downloaded`、`runtime_metadata_verified` 始终为 false。S 发布制品会提示需要先完成本地浮点转换才能执行。

在具备匹配 SDK 和 C++ 依赖的 X5 上，先显式下载模型，再构建运行。板端命令：

```sh
python3 samples/vision/yoloe/model/download.py --target x5 --variant 11s
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s \
  --build --output outputs/yoloe_cpp_x5_11s_run1
```

S 侧先执行转换流程并保留记录，使用转换记录中的预期摘要。将下面两个占位符换为真实制品路径及其 64 位十六进制 SHA-256：

```sh
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n \
  --model-path /path/to/converted-float.hbm \
  --local-float-sha256 REPLACE_WITH_EXPECTED_64_HEX_SHA256 \
  --build --output outputs/yoloe_cpp_s100p_26n_run1
```

摘要绑定本地字节，不证明转换来源或 SDK 兼容性；十个输出仍需通过运行时 metadata 检查。本目录不提供兼容的 S 浮点 HBM。启动器不隐式下载、不提供板卡身份覆盖，也不静默回退 target。

`--build` 配置 Release 构建，生成 `runtime/cpp/build/<target>/yoloe_demo`；不传则使用该位置的现有程序。自行构建时传 `--binary /absolute/path/yoloe_demo`，不能同时传 `--build`。在匹配的 SDK 环境中可显式配置依赖位置：

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-cli \
  -DYOLOE_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release \
  -DYOLOE_DNN_INCLUDE_DIR=/path/to/sdk/include \
  -DYOLOE_DNN_LIBRARY=/path/to/sdk/lib/libdnn.so
cmake --build /tmp/yoloe-board-cli --parallel 2
```

UCP 环境若无法发现匹配的 `libhbucp`，还需提供 `YOLOE_UCP_LIBRARY`；随后在选定的运行命令中使用 `--binary /tmp/yoloe-board-cli/yoloe_demo`。正常启动器在构建前核验身份和输入。缺少厂商依赖会显式失败；不能把测试替身作为部署依赖。

## 启动器参数

| 参数 | 默认值 / 作用 |
| --- | --- |
| `--target` | `auto`；执行时识别本机，dry-run 必须显式指定 |
| `--variant`、`--asset-id` | 默认变体见上表；asset ID 是当前清单的精确引用，冲突选择拒绝 |
| `--model-path` | 选定模型的标准路径，可覆盖为现有文件 |
| `--local-float-sha256` | 自行转换浮点模型的摘要，必须同时提供 `--model-path` |
| `--test-img` | sample 的 `test_data/office_desk.jpg`，按 BGR 解码 |
| `--label-file` | sample 的 `test_data/classes.names`；固定顺序的 4585 类，要求摘要一致 |
| `--output` | `outputs/yoloe_cpp`；必须为新目录，不覆盖已有运行 |
| `--build`、`--binary` | 显式构建或指定已构建程序，互斥 |
| `--score-thres` | 0.25，严格位于 0 与 1 之间 |
| `--nms-thres` | 仅 E11，默认 0.7，范围 [0,1]；E26 即使显式传默认值也拒绝 |
| `--resize-type` | 1 = letterbox；E11 还支持 0 = stretch；E26 只能为 1 |
| `--max-det` | 仅 E26 调整，默认 300，范围 1..8400；E11 只接受不变的默认值 |
| `--multi-label` | 仅 E26；默认每个选中 anchor 只保留一个类别 |
| `--no-morph` | 关闭 S E11 CLI 默认的 5×5 开运算；X5 E11/E26 原本就不开启 |
| `--no-contour` | 不画轮廓，不改变检测和掩码 |
| `--list-models`、`--dry-run` | 只读检查模式，互斥 |

原生程序还要求显式提供 `--target`、`--variant`、`--model-path`、`--model-sha256`、`--test-img`、`--label-file`、`--output`，通常由启动器传入。原生 `--model-sha256` 是核验后的字节摘要，不是发布制品选择器。单独执行程序的 `--help` 可查看参数。未提供 CPU/BPU 核数或调度优先级参数，SDK 适配器使用默认调度契约。

## 输出及失败记录

```text
outputs/yoloe_cpp_x5_11s_run1/
  launch-report.json
  configure.stdout.log / configure.stderr.log   # only with --build
  build.stdout.log / build.stderr.log           # when configuration succeeds
  native.stdout.log / native.stderr.log
  result/
    report.json
    annotated.png
    masks/000000.png ...
```

`launch-report.json` 记录发布引用、本地/发布模型类别、预期及观察摘要、生效选项、可执行程序摘要、精确子进程 argv/cwd、UTC 时间、退出码及结果文件摘要。stdout/stderr 保留原始字节，包括非 UTF-8 输出。清单没有校验值时，观察摘要不构成发布来源认证，`publisher_checksum_verified` 保持 false。自行转换模型所对应的发布 ID 只是架构参考，不是新文件的来源证明。

原生 `report.json` 使用 `rdk-model-zoo/yoloe-native-run/v1` schema，包含 target/variant、模型/图片/词表摘要、图像形状、生效配置及对齐的实例。每个实例记录从 0 起的类别 ID、固定标签、分数、原图 `[x1,y1,x2,y2]` 框及 ROI 掩码路径/形状。PNG 存储值为 0/255，内存掩码为 0/1；零面积实例保留精确的空轴和 `mask: null`，不写空 PNG。`annotated.png` 是展示叠图，不是评估掩码。**所有原生掩码均为 ROI，包括 X5；Python X5 使用整图概率掩码。** 评估输入不能直接互换，需适配相应协议。

原生报告最后写入。退出码为 0 但缺少身份一致的有效报告或结果文件，启动器仍判失败；只有原生结果成功后才记录 metadata 已验证。构建/推理/结果核验失败会保留日志和失败记录；图片保存中断可能留下部分文件。输出目录建立前的预检失败仅打印错误，不创建运行目录。重试时选择新输出路径。原生校验错误退出码为 2；启动器保留原生正退出码，信号终止映射为 2。

## 模块与模型契约

| 模块 | 职责 |
| --- | --- |
| `launcher.py`、`run.sh` | 发布制品选择、显式构建、进程日志和运行记录 |
| `src/main.cpp` | CLI 编排：预检门 → 输入 → 以 model/config/gate 构造 `YOLOE` → `predict` → 保存结果 |
| `src/cli.cpp`、`inc/cli.hpp` | 严格参数解析、图片/标签逐字节读取、全新输出目录策略及报告/掩码写入 |
| `src/detect.cpp`、`inc/detect.hpp` | 单一单元承载完整 YOLOE 模型：几何、NV12 准备、`bind_heads` 角色绑定、E11/E26 解码、ROI 掩码还原、板卡/模型/词表预检、`preprocess`/`infer`/`postprocess`/`predict` 阶段及板端 DNN 适配器（以 `YOLOE_HAS_SDK` 编译） |
| `tests/test_float_heads.cc` | 验证形状、精度、步长、分配长度及有限值边界 |
| `tests/test_e11_decode.cc` | DFL、同/异类抑制、分数/IoU 相等边界及非法输出 |
| `tests/test_e26_decode.cc` | 验证候选排序、单/多标签、阈值和非法输出 |
| `tests/decode_probe.cc` | 对照已保存原始浮点张量的主机工具，不是推理应用 |

浮点读取复用 Ultralytics 的共享词汇头 `inc/yolo.hpp`：`nhwc_float_plan` 要求未量化 FLOAT32、batch=1 的 NHWC、有效物理行/单元步长及足够的内存分配，`copy_float_output` 去除物理填充并拒绝非有限值。`bind_heads` 按唯一形状绑定十个逻辑输出角色；数据类型、步长和分配要求由这两个工具强制执行，模型家族与词表身份由下文的显式选择规则核验。

| 角色 | stride 8 / 16 / 32 对应形状 |
| --- | --- |
| 类别 logits | `[1,80,80,4585]` / `[1,40,40,4585]` / `[1,20,20,4585]` |
| E11 DFL 框 | 相同空间形状，64 通道 |
| E26 直接 LTRB 框 | 相同空间形状，4 通道 |
| 掩码系数 | 相同空间形状，32 通道 |
| Prototype | 单个 `[1,160,160,32]` 张量 |

调用者明确选择 E11（框通道 64）或 E26（框通道 4）；家族不符、缺失/重复角色、错误词表宽度及额外输出均拒绝。SDK 管理复用 Ultralytics 共享 DNN 后端（`inc/backend.hpp` + `src/backend.cpp`）提供的 `PackedModelOwner`、`Nv12Input`、`TaskOutputs` 及跨栈可移植的同步推理；YOLOE 语义角色在分配前校验，量化输出直接拒绝，不在后处理手动反量化。[转换说明](../../conversion/README_cn.md)给出浮点输出模型的准备方法；S 浮点 HBM 按转换说明生成。

<a id="dependencies"></a>
## 依赖

前置条件：C++17 编译器和仓库检出。主机测试需要 OpenCV C++ core/imgproc/imgcodecs 开发库，但不需要板端 SDK：几何/解码/预检函数与基于 OpenCV 的准备代码位于同一模型翻译单元，因此即使核心数值测试也链接模型核心（进而依赖 OpenCV）。文档中的构建/测试命令需要 CMake/CTest 3.20+（使用 `ctest --test-dir`）；若不能自动发现 OpenCV，将 `OpenCV_DIR` 指向已安装的 OpenCV CMake 包目录。仅安装 Python opencv-python 不会提供这里需要的 C++ 开发环境。

在 macOS 上，开启 OpenCV 的测试项目以 sanitizer 构建（`YOLOE_TEST_OPENCV=ON` 且 `YOLOE_SANITIZERS=ON`）时，还会通过 CMake 标准 TBB CONFIG 包解析 OpenCV 构建所依赖的真实 TBB，并将其直接链接到使用 OpenCV 的测试可执行程序。在 ASan 下，仅经由 `libopencv_core` 间接加载 libtbb 的进程会在退出时于 `tbb::detail::r1::__TBB_InitOnce::~__TBB_InitOnce` 中中止；仅链接 OpenCV core 的空 `main` 即可复现该崩溃。直接的 TBB 引用使 ASan+UBSan 得以保留；若 CMake 无法发现该包，将 `CMAKE_PREFIX_PATH` 指向安装 TBB 的前缀。Linux 构建与关闭 OpenCV 的构建不经过该分支，也不需要 TBB。该行为只影响主机 sanitizer 构建，板端构建不受影响。

<a id="build"></a>
## 构建主机测试

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-core \
  -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-core --parallel 4
```

使用真实 OpenCV 安装构建全部十一个测试（不需要板端 SDK）：

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-opencv \
  -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-opencv --parallel 4
```

从独立 CMake 项目构建可复用阶段库及全部十一个测试：

```bash
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-stage-core \
  -DYOLOE_BUILD_TESTS=ON -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-stage-core --parallel 4
```

产物为 `libyoloe_core.a`，并非板端可执行程序。正常集成构建可省略 `YOLOE_BUILD_TESTS` 和 `YOLOE_SANITIZERS`（库项目中均默认 OFF）。使用方可通过 CMake `add_subdirectory` 加入目录并链接 `yoloe_core`，头文件路径与 OpenCV 依赖会传递给调用方。`YOLOE_TEST_OPENCV` 控制独立测试项目，不会移除阶段库本身的 OpenCV 依赖。

<a id="run"></a>
## 运行主机测试

```bash
ctest --test-dir /tmp/yoloe-native-core --output-on-failure
```

成功时退出码为 0、无输出。解码测试分配完整的 4585 类张量，开启 sanitizer 时应预留数百 MB 内存。断言/契约异常及 sanitizer 报错均为失败。这些命令验证 C++ 数学及浮点内存工具；SDK ABI 兼容性由下文板端构建验证。

对 CMake 构建执行全部十一个检查并显示失败输出：

```bash
ctest --test-dir /tmp/yoloe-native-opencv --output-on-failure
```

执行库项目的十一个测试：

```bash
ctest --test-dir /tmp/yoloe-stage-core --output-on-failure
```

<a id="parameters"></a>
## 候选解码

`decode_e11` 接受相同语义排列，但框为 64 通道，复用 Ultralytics 数值稳定的 16-bin DFL 期望及 sigmoid。默认分数 0.25、分类 NMS IoU 0.7；分数范围为 `(0,1)`，NMS 为 `[0,1]`。每 anchor 保留一个类别，不套用 E26 候选上限。NMS 过程中系数随候选一起保留。

E11 边界：分数**大于等于**阈值时接受；同类框仅在 IoU **大于** NMS 阈值时抑制（Python NMS 还会抑制 IoU 恰好相等的情况，两种语言在该边界上可能不同）。原生结果按类别升序、分数降序排列；分数精确相等时优先较低的 scale/anchor 索引。解码前校验全部张量的精确长度和有限值，包括候选数学暂不使用的 prototype。

`decode_e26` 输入十个紧凑、有限值浮点向量，语义顺序为各 stride 的类别/框/系数，最后为 prototype。读取前检查每个向量的精确长度。默认分数阈值 0.25、最多 300 个候选、每 anchor 仅保留一个类别；合法阈值为 `(0,1)`，候选上限范围为 `1..8400`。

筛选采用静态 Top-K：候选按分数排序，精确平局依次优先较小的 scale、anchor、类别，多标签扩展按已选 anchor 排名和类别打破平局。原始阈值比较使用严格大于。框处于 640×640 模型画布，分数通过 sigmoid，掩码系数按实例对齐。本模块不执行 IoU NMS、几何恢复、掩码生成或反量化；框解码产生非有限值时拒绝，包括有限距离乘 stride 后溢出的情况。当前多标签扩展内存与已选 anchor 数乘 4585 成正比，大候选上限的开销明显高于默认值。

`prepare_bgr` 返回拥有独立存储的 640×640 BGR 像素和显式几何。E11 letterbox 截断缩放尺寸、填充 127，可选 stretch 使用最近邻；E26 letterbox 使用 ties-to-even 舍入、填充 114，拒绝 stretch。两种 letterbox 均使用线性插值，极窄图片的缩放尺寸至少为 1。框还原采用实际水平/垂直缩放比例，使用实际缩放尺寸还原坐标。

`restore_e26_masks` 组合 prototype logits 与系数并检查结果有限值，线性插值到 640×640，按零阈值二值化并在模型坐标裁剪，去除记录的填充，再用最近邻恢复原图尺寸，最后复制裁剪且向零取整后的 ROI。返回独立的浮点框及取值 0/1 的 `CV_8UC1` 掩码，保留空/退化实例位置。反向或非有限框、非法几何、错误 prototype 长度、非有限系数/prototype、数值溢出均失败。不执行 sigmoid、NMS、形态学或反量化。

`restore_e11_masks` 实现 S11 ROI 协议：将模型框裁至实际图片内容，按 prototype 尺度向零取整边界，组合原始 prototype 值与系数，以严格大于 0.5 二值化，对二值裁剪区执行 Lanczos4 缩放，可选 5×5 矩形开运算（库默认 `do_morph=false`）。Lanczos 可能把 uint8 二值数据插成 2，最终统一将正值转为 1，因此 C++ ROI 掩码是严格的 0/1 `CV_8UC1` 图像。空框保留精确的零尺寸轴。Python S11 运行时按 uint8 原样返回 Lanczos 输出（个别像素可能为 2，非零即前景）。该协议不同于 X5 Python 的全图概率掩码流程。`do_morph` 在相同候选输入上开关 5×5 开运算。

<a id="interface-lifecycle"></a>
## 接口与生命周期

`YOLOE` 独占一个 `std::unique_ptr<Runner>`。构造时验证配置和后端协议，构造失败也会释放传入后端。原生构造函数 `YOLOE(SdkModel, Config, SdkPreflight)` 自行创建并持有板端适配器；注入 runner 的构造函数供自定义后端与测试使用。后端必须返回十个独立拥有存储的紧凑语义 FLOAT32 向量，并在推理前完成硬件/制品身份与 SDK metadata 校验；基类接口本身不承担这些校验。`SdkRunner` 实现下述低层 SDK 边界。

`preprocess` 返回独立紧凑 Y 平面（409600 字节）、交错 UV 平面（204800 字节）和实际几何；`infer` 恰好调用 runner 一次，返回独立原始输出并携带对应几何；`postprocess` 返回框/分数/类别/ROI 掩码对齐的 `Instance`。不同 task 的 prepared/raw 批次不能串用，即便协议相同也会拒绝；不用自行缓存上一张图的几何。原始输出跨后续调用仍有效，结果掩码不借用 SDK 缓冲。每个推理线程使用一个 task，不承诺后端并发安全。

下面的函数展示嵌入 API 的用法；应用提供匹配的后端：

```cpp
#include "detect.hpp"
yoloe::Result process_image(yoloe::Config config,
                            std::unique_ptr<yoloe::Runner> backend,
                            const cv::Mat& image) {
    yoloe::YOLOE task(config, std::move(backend));
    auto prepared = task.preprocess(image);
    auto raw = task.infer(prepared);
    return task.postprocess(raw);
    // task.predict(image) composes the same three operations.
}
```

配置默认 E11、score 0.25、NMS 未设置（E11 解析为 0.7）、形态学关闭、letterbox、max_det 300、single_label true。E11 拒绝仅属于 E26 的参数覆盖；E26 拒绝显式 NMS、形态学或 stretch。仅在后端匹配时设置 `config.protocol = yoloe::Protocol::E26`。这些是库默认值；demo CLI 在参数缺省时应用自身的默认值。

解码/几何函数提供纯接口，不管理 SDK 资源。`bind_heads` 返回输出索引，不保留引用。`decode_e11` 和 `decode_e26` 仅在调用期间借用十个输入向量，返回拥有独立框、分数和系数的检测结果；返回后调用方可释放输入张量。内部指针视图不逃逸。OpenCV 结果矩阵拥有引用计数管理的独立存储，不与调用方图片/prototype 别名；需要像素时应保留返回对象。参数非法时抛出 `std::invalid_argument`，内存分配失败可能向外传播。不加载模型、不隐式选择硬件，也不保存跨调用的图片几何状态。

## SDK 后端库

`SdkRunner(SdkModel, SdkPreflight)` 持有一个模型及其输入输出分配。
`SdkModel` 包含 `path`、`target`、`variant`；允许 X5 E11s/m/l、S100
E11s/E26n/s/m/l/x、S100P E26n/s/m/l/x，且必须与编译使用的 SDK 栈一致。
它检查非空模型文件、唯一具名模型、目标对应的 640×640 NV12 输入，以及十个
有限、无量化 FLOAT32 NHWC 输出角色。物理输出顺序可以不同，返回向量始终遵循
上面的语义顺序，不暴露 SDK 缓冲区。析构顺序为输出、输入、模型；构造失败也会
释放已获取资源。推理直接上传 Y/UV、清理输入 cache、提交/等待/释放任务，随后
使输出 cache 可读并复制输出；不做解码或绘图。

**预检回调必填，没有默认值。** 应用提供 `void(const SdkModel&)`，核验实际
板卡、选定的发布制品或自编译浮点 SHA-256，以及词表/转换来源。回调先于任何
SDK 调用执行，抛异常即停止构造。适配器的形状、target/variant 和 SDK 栈检查
不能证明这些身份。内置 `make_preflight(expected_model_sha256, label_path)` 已提供本机身份和字节核验；
启动器先选择发布制品或本地浮点文件，再将核验后的摘要传给可执行程序。

`make_preflight` 按共用规则读取真实本机 sysfs/device-tree，拒绝未知/不符板卡、
缺失/空模型、格式错误或不匹配的模型摘要，以及不符合固定有序 PF 文件的词表。
64 位模型预期摘要由调用者明确提供，不会对同一文件现算现认。`SdkRunner` 另行
检查 target/variant/SDK 栈及张量契约，没有身份覆盖入口。自转换模型摘要匹配只
证明字节一致，不认证编译器来源；调用者仍需保留转换证据，本 API 不提供兼容的
S 浮点 HBM。

第二个完整 API 示例展示带预检的原生路径；task 自行构造并持有板端适配器：

```cpp
#include "detect.hpp"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                const std::string& expected_model_sha256,
                                const std::string& label_path) {
    yoloe::Config config;
    config.protocol = model.variant.rfind("26", 0) == 0 ? yoloe::Protocol::E26
                                                        : yoloe::Protocol::E11;
    yoloe::YOLOE task(model, config,
                      yoloe::make_preflight(expected_model_sha256, label_path));
    return task.predict(image);
}
```

在安装了匹配板端 SDK 的环境中构建后端库：

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-lib \
  -DYOLOE_BUILD_SDK=ON
cmake --build /tmp/yoloe-board-lib --parallel 4
```

产物为 `libyoloe_core.a` 和 `libyoloe_sdk.a`，不含可执行程序。通过 CMake
`add_subdirectory` 集成时将应用链接到 `yoloe_sdk`。自动查找失败时，将
`YOLOE_DNN_INCLUDE_DIR` 指向包含 `dnn/hb_dnn.h` 的目录，
`YOLOE_DNN_LIBRARY` 指向匹配的 DNN 库；UCP 头还要求 `YOLOE_UCP_LIBRARY`。
同时暴露 hbSys/UCP 头会拒绝构建。不要给部署库使用测试替身头；仍需 OpenCV
开发文件，缺省主机构建保持 `YOLOE_BUILD_SDK=OFF`。

OpenCV 主机配置运行十一项测试：六项数值/阶段检查、X5/UCP 适配器、预检、CLI I/O 及夹具 help。显式夹具还使用合成输出执行真实入口；它不是 SDK 后端。
预检位于模型核心内，需要 OpenCV 但不需要 SDK；适配器测试使用 ASan/UBSan 检查生产代码。覆盖先于 SDK 的预检拒绝、先于分配的元数据/精度拒绝、
部分分配和初始化失败清理、任务/cache 错误、语义输出顺序及跨调用独立持有。
使用的是精简 API 替身，不是厂商 SDK 头或库。

<a id="results-interpretation"></a>
## 部署与结果使用

使用所选板卡 SDK 和 OpenCV 构建原生入口，按[转换指南](../../conversion/README_cn.md)准备浮点输出模型，再以匹配的 profile 和固定 4585 类词表运行启动器。

每次运行保留完整证据：`result/report.json`（schema `rdk-model-zoo/yoloe-native-run/v1`）列出对齐实例——从 0 起的类别 ID、固定标签、分数、原图 `[x1,y1,x2,y2]` 框及 ROI 掩码路径；`launch-report.json` 记录所选发布引用、预期/观察摘要、精确 argv/cwd 和退出码。`annotated.png` 是展示叠图；后续处理请使用存储的掩码和报告字段。

所有原生掩码在全部目标上均为 ROI 布局（包括 X5）；Python X5 使用整图概率掩码，混合评估输入前需适配协议。PNG 存储值为 0/255，内存掩码为 0/1。数据集级精度来自[评估器](../../evaluator/README_cn.md)流程。
