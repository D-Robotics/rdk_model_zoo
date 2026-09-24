# B7 native SDK 独立核对（2026-09-24）

基点：HP 作者整改提交 `5fc14a4`；本地 GLM 开发正在继续，本报告不代表其后续修改状态。

## 阻断：S adapter 使用不存在的 X5 API

通过已有 SSH 配置在真实 S100 读取 `/usr/include/hobot/dnn/hb_dnn.h`：`hbDNNTensorProperties` 没有 `alignedShape`；`hbDNNQuantiType` 只有 NONE/SCALE，没有 SHIFT；`dnn/hb_dnn_ext.h` 不存在。当前 `samples/vision/yolov5/runtime/cpp/src/s_adapter.cpp` 却包含该头、引用 SHIFT，并访问 properties.alignedShape。必须按 S UCP 实际 SDK 单独适配，不能用 host portable tests 证明 native 可编译。S 头位于 `/usr/include/hobot/dnn`，CMake 的 include root 也须配合 `#include <dnn/hb_dnn.h>` 核查。

X5 的对应字段/枚举确实存在，不能将 S 修复机械应用到 X5。完整命令、原始输出和时间见 [SDK evidence](evidence/2026-09-24-b7-native-sdk-preflight.json)。

## 环境与边界

X5 8GB/4GB、S100、S100P、S600 64GB 五个既有 SSH 目标均可达。真实板端构建正在准备 GitHub 上的上述提交，推理尚未执行。不得将连接成功或头文件核对记录为板端推理 passed。开发仍交给本地 Claude Code + GLM，由 Codex 统筹 GitHub 和独立评审。

## 实板执行追加

同一 `5fc14a4995110990f2abaa6ec8189d052a0195e6` 在 X5 8GB 编译 rc=0，S100 编译 rc=2，首个编译错误为 `dnn/hb_dnn.h` 找不到。原始构建输出保存在 evidence/2026-09-24-b7-board-initial/。

X5 C++ 默认 s-v2.0 已实际推理 rc=0，bus.jpg 上输出五个 detection，模型 observed SHA-256 为 `1b627740f4a9d322efda8ece8ae01a088900ab28bd62c11a47b31981651e2aee`。发布者未提供 hash；运行库报告 model/runtime HBRT 小版本不一致，原始警告完整保留。此结果仅为 smoke，不是源数值一致性验收。

进一步阻断：

- C++ dump 实际只保存三个 output bin。manifest 的 inputs 只有描述、dtype 为 unknown，没有 native NV12 输入 bytes，也没有部署代码/二进制 digest。仍不满足完整 source/unified 比较证据，不能将 dump 的存在当作该 finding 已关闭。
- X5 Python evaluator 在真实 SDK 抛 `TypeError: cannot pickle hbm_runtime.HB_HBMRuntime.QuantParams object`：`compare.py` 对 `RuntimeMetadata` 调用 dataclasses.asdict，触发对 SDK 对象的 deepcopy。失败发生在 legacy 模型初始化的 metadata capture 中，尚未比较输出。需要显式序列化 SDK metadata/quant 字段，并用禁止 deepcopy 的测试替身覆盖；排查其他 evaluator 的同类模式。失败 comparison.json 已原样保留，禁止修改为通过。

证据目录：[首次板端构建/推理/失败记录](evidence/2026-09-24-b7-board-initial/)。C++ raw 输出仍在板端 `/tmp/rdk-b7-x5-native-dump`，尚未作为仓库可携带原始数组归档；故当前不支持完整对照验收。
