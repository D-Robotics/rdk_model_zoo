# YOLOE C++ Runtime 迁移

[English](README.md) | 简体中文

当前目录提供统一原生运行时所需的浮点输出绑定和 E11/E26 候选解码模块，**尚未提供完整板端可执行程序**。已经实现的统一入口见 [Python runtime](../python/README_cn.md)，运行前仍需满足其制品和 SDK 条件。原始 C++ 程序保留在 [S E11 快照](../../../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md)及 [S E26 快照](../../../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README_cn.md)；原始量化制品和手动反量化路径不符合这里的新浮点契约。

<a id="supported-boards"></a>
## 目标范围

原始原生能力为 S100 E11s 及 S100/S100P E26 n/s/m/l/x。本增量仅验证主机模块，没有启用任何板端可执行程序。S600 仍不支持。X5 发布了 E11 Python 制品；共用形状绑定模块本身不构成 X5 原生实现。

## 模块与模型契约

| 模块 | 职责 |
| --- | --- |
| `common/float_heads.h` | 按唯一形状绑定十个逻辑角色，不依赖物理输出顺序 |
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

调用者明确选择 E11（框通道 64）或 E26（框通道 4）；家族不符、缺失/重复角色、错误词表宽度及额外输出均拒绝。本目录尚未实现掩码解码及 SDK 输入输出资源管理。[转换说明](../../conversion/README_cn.md)给出浮点输出模型的准备方法；本轮迁移尚未编译或验证兼容的 S 浮点 HBM。

<a id="dependencies"></a>
## 依赖

需要 C++17 编译器和仓库检出；三个单元测试不需要 OpenCV 或板端 SDK。在仓库根目录执行：

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
```

<a id="run"></a>
## 运行主机测试

```bash
/tmp/yoloe-native-tests/float-heads
/tmp/yoloe-native-tests/e26-decode
/tmp/yoloe-native-tests/e11-decode
```

成功时退出码为 0、无输出。解码测试分配完整的 4585 类张量，开启 sanitizer 时应预留数百 MB 内存。断言/契约异常及 sanitizer 报错均为失败。这些命令实际编译 C++ 数学及浮点内存工具，不证明真实 SDK ABI 兼容。

<a id="parameters"></a>
## 候选解码

`decode_e11` 接受相同语义排列，但框为 64 通道，复用 Ultralytics 数值稳定的 16-bin DFL 期望及 sigmoid。默认分数 0.25、分类 NMS IoU 0.7；分数范围为 `(0,1)`，NMS 为 `[0,1]`。每 anchor 保留一个类别，不套用 E26 候选上限。NMS 过程中系数随候选一起保留。

保留原始 E11 C++ 边界：分数**大于等于**阈值时接受；同类框仅在 IoU **大于** NMS 阈值时抑制。现有 Python NMS 也抑制恰好相等的情况，因此不声明精确边界等价。原生结果按类别升序、分数降序排列；分数精确相等时优先原始 scale/anchor 索引。这使源 unordered-map/OpenMP 合并结果具有确定顺序，但不复现源实现未指定的平局顺序或 NumPy 平局排序。解码前校验全部张量的精确长度和有限值，包括候选数学暂不使用的 prototype。

`decode_e26` 输入十个紧凑、有限值浮点向量，语义顺序为各 stride 的类别/框/系数，最后为 prototype。读取前检查每个向量的精确长度。默认分数阈值 0.25、最多 300 个候选、每 anchor 仅保留一个类别；合法阈值为 `(0,1)`，候选上限范围为 `1..8400`。

筛选保持源静态 Top-K 契约。精确平局依次优先较小的 scale、anchor、类别；多标签扩展按已选 anchor 排名和类别打破平局。原始阈值比较使用严格大于。框处于 640×640 模型画布，分数通过 sigmoid，掩码系数按实例对齐。本模块不执行 IoU NMS、几何恢复、掩码生成或反量化；框解码产生非有限值时拒绝，包括有限距离乘 stride 后溢出的情况。当前多标签扩展内存与已选 anchor 数乘 4585 成正比，大候选上限的开销明显高于默认值。

<a id="interface-lifecycle"></a>
## 接口与生命周期

数值模块头文件提供纯函数，不管理 SDK 资源。`bind_heads` 返回输出索引，不保留引用。`decode_e11` 和 `decode_e26` 仅在调用期间借用十个输入向量，返回拥有独立框、分数和系数的检测结果；返回后调用方可释放输入张量。内部指针视图不逃逸。参数非法时抛出 `std::invalid_argument`，内存分配失败可能向外传播。不加载模型、不隐式选择硬件，也不保存跨调用的图片几何状态。

<a id="results-interpretation"></a>
## 验证与后续集成

[实现证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-kernels-review.md)记录先前 E26 主机验证；[E11 扩展证据](../../../../../docs/releases/unified-migration/2026-09-28-yoloe-cpp-e11-review.md)记录三个原生测试、E26 回归及真实 E11 s/m/l 对照。E26n 对照覆盖单/多标签模式下与 Python 的候选解码比较。类别和顺序要求完全一致，框、分数、系数按明确数值容差比较。候选解码测试不比较掩码。

后续仍需完成掩码恢复、图像/NV12 几何、SDK 资源管理、目标与制品身份门禁、公开 pre/infer/post/predict 阶段、库/CLI 入口及完整板端构建运行说明。板端、真实 SDK、OE 编译和原生数据集精度均未验证；本目录尚不代表 C++ 迁移完成，也尚未替代归档原始程序。
