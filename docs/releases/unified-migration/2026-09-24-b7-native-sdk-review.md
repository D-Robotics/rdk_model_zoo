# B7 native SDK 独立核对（2026-09-24）

基点：HP 作者整改提交 `5fc14a4`；本地 GLM 开发正在继续，本报告不代表其后续修改状态。

## 阻断：S adapter 使用不存在的 X5 API

通过已有 SSH 配置在真实 S100 读取 `/usr/include/hobot/dnn/hb_dnn.h`：`hbDNNTensorProperties` 没有 `alignedShape`；`hbDNNQuantiType` 只有 NONE/SCALE，没有 SHIFT；`dnn/hb_dnn_ext.h` 不存在。当前 `samples/vision/yolov5/runtime/cpp/src/s_adapter.cpp` 却包含该头、引用 SHIFT，并访问 properties.alignedShape。必须按 S UCP 实际 SDK 单独适配，不能用 host portable tests 证明 native 可编译。S 头位于 `/usr/include/hobot/dnn`，CMake 的 include root 也须配合 `#include <dnn/hb_dnn.h>` 核查。

X5 的对应字段/枚举确实存在，不能将 S 修复机械应用到 X5。完整命令、原始输出和时间见 [SDK evidence](evidence/2026-09-24-b7-native-sdk-preflight.json)。

## 环境与边界

X5 8GB/4GB、S100、S100P、S600 64GB 五个既有 SSH 目标均可达。真实板端构建正在准备 GitHub 上的上述提交，推理尚未执行。不得将连接成功或头文件核对记录为板端推理 passed。开发仍交给本地 Claude Code + GLM，由 Codex 统筹 GitHub 和独立评审。
