# YOLOE C++ Runtime

[English](README.md) | 简体中文

使用 `run.sh` 执行 E11/E26 无提示实例分割，或将 C++ 三阶段库嵌入应用。启动器精确选择模型，核验本机和文件身份，按显式要求构建原生程序，并保留日志、图片和掩码结果。**当前仅有主机验证：真实 SDK 编译和板端推理仍为 not-run。** 另一个统一入口见 [Python runtime](../python/README_cn.md)，注意其 X5 掩码协议不同。

<a id="supported-boards"></a>
## 目标范围

| 目标 | 变体 / 默认值 | 模型前提 | 当前验证 |
| --- | --- | --- | --- |
| X5 | 11s/m/l；默认 11s | 已发布的浮点输出 BIN，匹配 X5 SDK | 仅主机；SDK/板端 not-run |
| S100 | 11s、26n/s/m/l/x；默认 11s | 自行转换的浮点输出 HBM，显式提供 SHA-256 | 仅主机；兼容 HBM/SDK/板端未验证 |
| S100P | 26n/s/m/l/x；默认 26n | 自行转换的浮点输出 HBM，显式提供 SHA-256 | 仅主机；兼容 HBM/SDK/板端未验证 |
| S600 | 无 | 无发布路线，显式拒绝 | 不支持 |

14 个发布身份用于模型选择，**不表示已有 14 个可执行原生制品**。S 已发布文件的输出为量化数据，本入口拒绝加载；通过[转换流程](../../conversion/README_cn.md)准备浮点输出，改名不能改变契约。原始 [S E11](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) 和 [S E26](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README_cn.md) 保留为历史参考，其能力和测量记录不代表本实现的验收结果。

## 选择、构建与运行

下面命令从仓库根目录执行。启动器需要 Python 3.10+ 和 NumPy，本原生入口不需要 Python OpenCV；用 `PYTHON` 指定 `run.sh` 使用的解释器。脚本也支持从其他目录调用。用户传入的相对路径及默认输出位置相对于调用目录解析；子进程在仓库根目录运行。

以下命令可在主机执行，仅查看当前清单和准备命令，不下载、不构建、不检查板卡、不执行推理：

```bash
bash samples/vision/yoloe/runtime/cpp/run.sh --list-models
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s --dry-run
bash samples/vision/yoloe/runtime/cpp/run.sh --target s100p --variant 26n --dry-run
```

Dry-run 必须显式指定 target，`executed`、`downloaded`、`runtime_metadata_verified` 始终为 false。S 发布制品会提示需要本地浮点转换；dry-run 返回成功不代表该量化文件可执行。

在具备匹配 SDK 和 C++ 依赖的 X5 上，先显式下载模型，再构建运行。以下板端命令在本轮主机迁移中**未执行**：

```sh
python3 samples/vision/yoloe/model/download.py --target x5 --variant 11s
bash samples/vision/yoloe/runtime/cpp/run.sh --target x5 --variant 11s \
  --build --output outputs/yoloe_cpp_x5_11s_run1
```

S 侧先完成转换并保留证据，提供转换记录中的预期摘要，不能随意填写摘要来绕过校验。将下面两个占位符换为真实制品路径及其 64 位十六进制 SHA-256：

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

原生报告最后写入。退出码为 0 但缺少身份一致的有效报告或结果文件，启动器仍判失败。主机夹具报告标为 `host-fixture`，正常启动器拒绝将其当作 SDK 证据；只有原生结果成功后才记录 metadata 已验证。构建/推理/结果核验失败会保留日志和失败记录；图片保存中断可能留下部分文件。输出目录建立前的预检失败仅打印错误，不创建运行目录。重试时选择新输出路径。原生校验错误退出码为 2；启动器保留原生正退出码，信号终止映射为 2。

## 模块与模型契约

| 模块 | 职责 |
| --- | --- |
| `launcher.py`、`run.sh` | 发布制品选择、显式构建、进程日志和运行记录 |
| `src/main.cpp`、`src/cli_options.cpp`、`src/cli_io.cpp` | CLI 编排、参数解析、图片/标签读取及结果保存 |
| `common/float_heads.h` | 按唯一形状绑定十个逻辑角色，不依赖物理输出顺序 |
| `inc/yoloe.h`、`src/yoloe.cpp` | 构造及 pre_process/infer/post_process/predict 编排 |
| `inc/runner.h`、`inc/pipeline_io.h` | 后端契约、独立输入输出和实例身份 |
| `inc/sdk_runner.h`、`src/sdk_runner.cpp` | 必需预检之后的模型加载、SDK 资源及共用输入输出传输 |
| `inc/model_identity.h`、`inc/preflight.h`、`src/preflight.cpp` | 显式模型选择及基于共用原生工具的本机/模型/词表核验 |
| `inc/config.h` | 分协议配置校验 |
| `common/nv12.h` | BGR 转 I420 及共用 split-NV12 打包 |
| `common/postprocess.h` | 家族分派及对齐实例结果组装 |
| `common/geometry.h` | 显式 E11/E26 缩放几何及实际比例框还原 |
| `common/image_ops.h` | OpenCV BGR 前处理及 E11/E26 ROI 掩码恢复 |
| `common/candidate.h` | 两个家族共用、拥有独立数据的候选结果 |
| `common/e11_decode.h` | DFL16、分类 NMS 及对齐的 E11 系数 |
| `common/e26_decode.h` | 筛选 E26 PF 候选，解码 LTRB 框并保持掩码系数对齐 |
| `tests/test_float_heads.cc` | 验证形状、精度、步长、分配长度及有限值边界 |
| `tests/test_e11_decode.cc` | DFL、同/异类抑制、分数/IoU 相等边界及非法输出 |
| `tests/test_e26_decode.cc` | 验证候选排序、单/多标签、阈值和非法输出 |
| `tests/decode_probe.cc` | 对照已保存原始浮点张量的主机工具，不是推理应用 |

浮点读取复用 Ultralytics 的 `common/task_outputs.h`：`nhwc_float_plan` 要求未量化 FLOAT32、batch=1 的 NHWC、有效物理行/单元步长及足够的内存分配；`copy_float_output` 去除物理填充并拒绝非有限值。`bind_heads` 仅识别逻辑角色，形状吻合本身不证明数据类型、内存安全、模型家族或词表身份。

| 角色 | stride 8 / 16 / 32 对应形状 |
| --- | --- |
| 类别 logits | `[1,80,80,4585]` / `[1,40,40,4585]` / `[1,20,20,4585]` |
| E11 DFL 框 | 相同空间形状，64 通道 |
| E26 直接 LTRB 框 | 相同空间形状，4 通道 |
| 掩码系数 | 相同空间形状，32 通道 |
| Prototype | 单个 `[1,160,160,32]` 张量 |

调用者明确选择 E11（框通道 64）或 E26（框通道 4）；家族不符、缺失/重复角色、错误词表宽度及额外输出均拒绝。SDK 管理已复用 Ultralytics 的 `PackedModelOwner`、`Nv12Input` 和 `TaskOutputs`；YOLOE 语义角色在分配前校验，量化输出直接拒绝，不在后处理手动反量化。[转换说明](../../conversion/README_cn.md)给出浮点输出模型的准备方法；本轮迁移尚未编译或验证兼容的 S 浮点 HBM。

<a id="dependencies"></a>
## 依赖

需要 C++17 编译器和仓库检出；四个几何/候选测试不需要 OpenCV 或板端 SDK；图像/掩码及阶段测试需要 OpenCV C++ core/imgproc/imgcodecs 开发库。文档构建/测试命令需要 CMake/CTest 3.20+（使用 `ctest --test-dir`）；若不能自动发现 OpenCV，将 `OpenCV_DIR` 指向已安装的 OpenCV CMake 包目录。仅安装 Python opencv-python 不会提供这里需要的 C++ 开发环境。在仓库根目录执行：

<a id="build"></a>
## 构建主机测试

```bash
mkdir -p /tmp/yoloe-native-tests
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  -I samples/vision/ultralytics_yolo/runtime/cpp \
  samples/vision/yoloe/runtime/cpp/tests/test_float_heads.cc \
  -o /tmp/yoloe-native-tests/float-heads
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  samples/vision/yoloe/runtime/cpp/tests/test_e26_decode.cc \
  -o /tmp/yoloe-native-tests/e26-decode
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  -I samples/vision/ultralytics_yolo/runtime/cpp \
  samples/vision/yoloe/runtime/cpp/tests/test_e11_decode.cc \
  -o /tmp/yoloe-native-tests/e11-decode
c++ -std=c++17 -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I samples/vision/yoloe/runtime/cpp/common \
  samples/vision/yoloe/runtime/cpp/tests/test_geometry.cc \
  -o /tmp/yoloe-native-tests/geometry
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
/tmp/yoloe-native-tests/float-heads
/tmp/yoloe-native-tests/e26-decode
/tmp/yoloe-native-tests/e11-decode
/tmp/yoloe-native-tests/geometry
```

成功时退出码为 0、无输出。解码测试分配完整的 4585 类张量，开启 sanitizer 时应预留数百 MB 内存。断言/契约异常及 sanitizer 报错均为失败。这些命令实际编译 C++ 数学及浮点内存工具，不证明真实 SDK ABI 兼容。

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

保留原始 E11 C++ 边界：分数**大于等于**阈值时接受；同类框仅在 IoU **大于** NMS 阈值时抑制。现有 Python NMS 也抑制恰好相等的情况，因此不声明精确边界等价。原生结果按类别升序、分数降序排列；分数精确相等时优先原始 scale/anchor 索引。这使源 unordered-map/OpenMP 合并结果具有确定顺序，但不复现源实现未指定的平局顺序或 NumPy 平局排序。解码前校验全部张量的精确长度和有限值，包括候选数学暂不使用的 prototype。

`decode_e26` 输入十个紧凑、有限值浮点向量，语义顺序为各 stride 的类别/框/系数，最后为 prototype。读取前检查每个向量的精确长度。默认分数阈值 0.25、最多 300 个候选、每 anchor 仅保留一个类别；合法阈值为 `(0,1)`，候选上限范围为 `1..8400`。

筛选保持源静态 Top-K 契约。精确平局依次优先较小的 scale、anchor、类别；多标签扩展按已选 anchor 排名和类别打破平局。原始阈值比较使用严格大于。框处于 640×640 模型画布，分数通过 sigmoid，掩码系数按实例对齐。本模块不执行 IoU NMS、几何恢复、掩码生成或反量化；框解码产生非有限值时拒绝，包括有限距离乘 stride 后溢出的情况。当前多标签扩展内存与已选 anchor 数乘 4585 成正比，大候选上限的开销明显高于默认值。

`prepare_bgr` 返回拥有独立存储的 640×640 BGR 像素和显式几何。E11 letterbox 截断缩放尺寸、填充 127，可选 stretch 使用最近邻；E26 letterbox 使用 ties-to-even 舍入、填充 114，拒绝 stretch。两种 letterbox 均使用线性插值，极窄图片的缩放尺寸至少为 1。框还原采用实际水平/垂直缩放比例，修正归档原生实现按理想 gain 反推导致的取整误差。

`restore_e26_masks` 组合 prototype logits 与系数并检查结果有限值，线性插值到 640×640，按零阈值二值化并在模型坐标裁剪，去除记录的填充，再用最近邻恢复原图尺寸，最后复制裁剪且向零取整后的 ROI。返回独立的浮点框及取值 0/1 的 `CV_8UC1` 掩码，保留空/退化实例位置。反向或非有限框、非法几何、错误 prototype 长度、非有限系数/prototype、数值溢出均失败。不执行 sigmoid、NMS、形态学或反量化；

`restore_e11_masks` 实现 S11 ROI 协议：将模型框裁至实际图片内容，按 prototype 尺度向零取整边界，组合原始 prototype 值与系数，以严格大于 0.5 二值化，对二值裁剪区执行 Lanczos4 缩放，可选 5×5 矩形开运算（库默认 `do_morph=false`）。Lanczos 可能把 uint8 二值数据插成 2，最终统一将正值转为 1；空框保留精确的零尺寸轴。这两项修正同时用于共用 Python DFL ROI 工具，正常前景范围不变。该协议不同于 X5 Python 的全图概率掩码流程。[E11 掩码证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-masks-review.md)在相同真实原生候选输入上对照开/关形态学两种设置。

<a id="interface-lifecycle"></a>
## 接口与生命周期

`YOLOE` 独占一个 `std::unique_ptr<Runner>`。构造时验证配置和后端协议，构造失败也会释放传入后端。后端必须返回十个独立拥有存储的紧凑语义 FLOAT32 向量，并在推理前完成硬件/制品身份与 SDK metadata 校验；基类接口本身不证明真实 SDK 实现。`SdkRunner` 已实现下述低层 SDK 边界；测试 runner 仍为显式主机夹具。

`pre_process` 返回独立紧凑 Y 平面（409600 字节）、交错 UV 平面（204800 字节）和实际几何；`infer` 恰好调用 runner 一次，返回独立原始输出并携带对应几何；`post_process` 返回框/分数/类别/ROI 掩码对齐的 `Instance`。不同 task 的 prepared/raw 批次不能串用，即便协议相同也会拒绝；不用自行缓存上一张图的几何。原始输出跨后续调用仍有效，结果掩码不借用 SDK 缓冲。每个推理线程使用一个 task，不承诺后端并发安全。

下面的函数已在主机验证中编译。应用需要提供真正匹配的后端；示例不会下载或伪造模型：

```cpp
#include "yoloe.h"
yoloe::Result process_image(yoloe::Config config,
                            std::unique_ptr<yoloe::Runner> backend,
                            const cv::Mat& image) {
    yoloe::YOLOE task(config, std::move(backend));
    auto prepared = task.pre_process(image);
    auto raw = task.infer(prepared);
    return task.post_process(raw);
    // task.predict(image) composes the same three operations.
}
```

配置默认 E11、score 0.25、NMS 未设置（E11 解析为 0.7）、形态学关闭、letterbox、max_det 300、single_label true。E11 拒绝仅属于 E26 的参数覆盖；E26 拒绝显式 NMS、形态学或 stretch。仅在后端匹配时设置 `config.protocol = yoloe::Protocol::E26`。这些是库默认值，源 demo CLI 默认值可能不同。

数值模块头文件提供纯函数，不管理 SDK 资源。`bind_heads` 返回输出索引，不保留引用。`decode_e11` 和 `decode_e26` 仅在调用期间借用十个输入向量，返回拥有独立框、分数和系数的检测结果；返回后调用方可释放输入张量。内部指针视图不逃逸。OpenCV 结果矩阵拥有引用计数管理的独立存储，不与调用方图片/prototype 别名；需要像素时应保留返回对象。参数非法时抛出 `std::invalid_argument`，内存分配失败可能向外传播。不加载模型、不隐式选择硬件，也不保存跨调用的图片几何状态。

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
不能证明这些身份。主机夹具中的空操作回调不能作为生产策略。内置 `make_preflight(expected_model_sha256, label_path)` 已提供本机身份和字节核验；
统一启动器先选择发布制品或本地浮点文件，再将核验后的摘要传给可执行程序。

`make_preflight` 按共用规则读取真实本机 sysfs/device-tree，拒绝未知/不符板卡、
缺失/空模型、格式错误或不匹配的模型摘要，以及不符合固定有序 PF 文件的词表。
64 位模型预期摘要由调用者明确提供，不会对同一文件现算现认。`SdkRunner` 另行
检查 target/variant/SDK 栈及张量契约，没有身份覆盖入口。自转换模型摘要匹配只
证明字节一致，不认证编译器来源；调用者仍需保留转换证据，本 API 不提供兼容的
S 浮点 HBM。

第二个完整 API 示例同样在主机验证中编译；调用者传入选定模型的预期摘要与词表路径：

```cpp
#include "preflight.h"
#include "sdk_runner.h"
#include "yoloe.h"
yoloe::Result process_sdk_image(const cv::Mat& image, yoloe::SdkModel model,
                                const std::string& expected_model_sha256,
                                const std::string& label_path) {
    auto gate = yoloe::make_preflight(expected_model_sha256, label_path);
    auto backend = std::make_unique<yoloe::SdkRunner>(model, std::move(gate));
    yoloe::Config config;
    config.protocol = backend->protocol();
    yoloe::YOLOE task(config, std::move(backend));
    return task.predict(image);
}
```

在安装了匹配板端 SDK 的环境中构建后端库（本轮主机工作未执行真实 SDK 构建）：

```sh
cmake -S samples/vision/yoloe/runtime/cpp -B /tmp/yoloe-board-lib \
  -DYOLOE_BUILD_SDK=ON
cmake --build /tmp/yoloe-board-lib --parallel 4
```

产物为 `libyoloe_core.a`、`libyoloe_preflight.a` 和 `libyoloe_sdk.a`，不含可执行程序。通过 CMake
`add_subdirectory` 集成时将应用链接到 `yoloe_sdk`。自动查找失败时，将
`YOLOE_DNN_INCLUDE_DIR` 指向包含 `dnn/hb_dnn.h` 的目录，
`YOLOE_DNN_LIBRARY` 指向匹配的 DNN 库；UCP 头还要求 `YOLOE_UCP_LIBRARY`。
同时暴露 hbSys/UCP 头会拒绝构建。不要给部署库使用测试替身头；仍需 OpenCV
开发文件，缺省主机构建保持 `YOLOE_BUILD_SDK=OFF`。

OpenCV 主机配置运行十一项测试：六项数值/阶段检查、X5/UCP 适配器、预检、CLI I/O 及夹具 help。显式夹具还使用合成输出执行真实入口；它不是 SDK 后端。
预检库本身不需要 OpenCV 或 SDK；适配器测试使用 ASan/UBSan 检查生产代码。覆盖先于 SDK 的预检拒绝、先于分配的元数据/精度拒绝、
部分分配和初始化失败清理、任务/cache 错误、语义输出顺序及跨调用独立持有。
使用的是精简 API 替身，不是厂商 SDK 头或库。

<a id="results-interpretation"></a>
## 验证边界

[原生预检证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-native-preflight-review.md)覆盖注册表一致性、模型/词表拒绝及共用流式摘要。

[SDK 适配器证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-sdk-runner-review.md)记录资源/元数据测试及其主机验证边界。

[阶段/NV12 证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-stages-review.md)覆盖实际字节对照、生命周期/异常路径及 API 示例编译，不据此认证板端后端。

[实现证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md)记录先前 E26 主机验证；[E11 扩展证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md)记录三个原生测试、E26 回归及真实 E11 s/m/l 对照。E26n 对照覆盖单/多标签模式下与 Python 的候选解码比较。[几何/掩码证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md)另行记录真实 OpenCV 编译及完整 ROI 像素对照。类别和顺序要求完全一致，框、分数、系数按明确数值容差比较。候选解码测试不比较掩码。

统一选择、CLI 和结果输出已有实现。[原生入口证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-native-cli-review.md)分别记录真实 OpenCV 主机执行、Python 进程策略测试与 SDK API 替身。板端推理、真实 SDK 编译、OE 产出的 S 浮点制品及原生数据集精度仍未验证；主机检查不等于独立迁移验收关闭。
