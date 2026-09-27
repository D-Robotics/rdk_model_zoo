# YOLOE C++ Runtime 迁移

[English](README.md) | 简体中文

当前目录提供 C++ 三阶段库，包含独立 NV12 输入、E11/E26 解码及 ROI 掩码，通过显式传入的 runner 执行推理，**尚未提供完整板端可执行程序**。已经实现的统一入口见 [Python runtime](../python/README_cn.md)，运行前仍需满足其制品和 SDK 条件。原始 C++ 程序保留在 [S E11 快照](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md)及 [S E26 快照](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README_cn.md)；原始量化制品和手动反量化路径不符合这里的新浮点契约。

<a id="supported-boards"></a>
## 目标范围

原始原生能力为 S100 E11s 及 S100/S100P E26 n/s/m/l/x。本增量仅验证主机模块，没有启用任何板端可执行程序。S600 仍不支持。新增 SDK 适配器也接受 X5 SDK 下的 E11s/m/l；这部分已有实现，但未使用真实 SDK 构建或在板端运行。

## 模块与模型契约

| 模块 | 职责 |
| --- | --- |
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

需要 C++17 编译器和仓库检出；四个几何/候选测试不需要 OpenCV 或板端 SDK；图像/掩码及阶段测试需要 OpenCV C++ core/imgproc 开发库。文档构建/测试命令需要 CMake/CTest 3.20+（使用 `ctest --test-dir`）；若不能自动发现 OpenCV，将 `OpenCV_DIR` 指向已安装的 OpenCV CMake 包目录。仅安装 Python opencv-python 不会提供这里需要的 C++ 开发环境。在仓库根目录执行：

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

使用真实 OpenCV 安装构建全部九个测试（不需要板端 SDK）：

```bash
cmake -S samples/vision/yoloe/runtime/cpp/tests -B /tmp/yoloe-native-opencv \
  -DYOLOE_TEST_OPENCV=ON -DYOLOE_SANITIZERS=ON
cmake --build /tmp/yoloe-native-opencv --parallel 4
```

从独立 CMake 项目构建可复用阶段库及全部九个测试：

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

对 CMake 构建执行全部九个检查并显示失败输出：

```bash
ctest --test-dir /tmp/yoloe-native-opencv --output-on-failure
```

执行库项目的九个测试：

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
统一发布制品选择的 launcher 仍待完成，这个 API 本身还不是客户可直接运行的入口。

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

OpenCV 主机配置现在运行九项测试：原有六项，加上 X5/UCP 适配器测试及预检测试。
预检库本身不需要 OpenCV 或 SDK；适配器测试使用 ASan/UBSan 检查生产代码。覆盖先于 SDK 的预检拒绝、先于分配的元数据/精度拒绝、
部分分配和初始化失败清理、任务/cache 错误、语义输出顺序及跨调用独立持有。
使用的是精简 API 替身，不是厂商 SDK 头或库。

<a id="results-interpretation"></a>
## 验证与后续集成

[原生预检证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-native-preflight-review.md)覆盖注册表一致性、模型/词表拒绝及共用流式摘要。

[SDK 适配器证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-sdk-runner-review.md)记录资源/元数据测试及其主机验证边界。

[阶段/NV12 证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-stages-review.md)覆盖实际字节对照、生命周期/异常路径及 API 示例编译，不据此认证板端后端。

[实现证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md)记录先前 E26 主机验证；[E11 扩展证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md)记录三个原生测试、E26 回归及真实 E11 s/m/l 对照。E26n 对照覆盖单/多标签模式下与 Python 的候选解码比较。[几何/掩码证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-masks-review.md)另行记录真实 OpenCV 编译及完整 ROI 像素对照。类别和顺序要求完全一致，框、分数、系数按明确数值容差比较。候选解码测试不比较掩码。

后续仍需完成统一发布制品选择、CLI 入口及完整可执行程序构建运行说明；适配器 API 不代表这些要求已经关闭。板端、真实 SDK、OE 编译和原生数据集精度均未验证；本目录尚不代表 C++ 迁移完成，也尚未替代归档原始程序。
