# ResNet / YOLO 可读范例：Codex 独立验收

日期：2026-10-01。评审基线 `bfbe6aaf`；评审实现提交
`9041122fe344104321f2facc6b6b97ded22bb866`。分支
`codex/readable-model-examples-20261001`，本地实施，未合并 develop、未推送。
代码与测试由 Claude Code + GLM 实施；本文由 Codex 核对源码、diff、命令结果后撰写。

**结论：本轮正式方案的代码、文档、历史目录迁移及主机验收已完成。**
通过范围不包含真实模型导出、量化编译、板端推理或生产 SDK ABI 验证。
这几项按已确认范围延期，不用假 SDK、模拟导出或主机测试替代。
本结论对应 [2026-09-30 方案](../../superpowers/specs/2026-09-30-readable-model-examples-design.md)
及其 [实施计划](../../superpowers/plans/2026-09-30-readable-model-examples.md)，
不代表 2026-09-16 全仓集成计划的所有后续事项已完成。

## 逐项验收

| 方案要求 | 实现与核对依据 | 结论 |
| --- | --- | --- |
| 简单 main，显式构造模型并调用 predict | ResNet main 直接构造 ResNetClassifier；YOLO main 用已有任务分派构造模型，直接调用 predict。参数与展示留在样例本地辅助模块，入口无模型算法 | 通过 |
| 单文件可读模型主线 | classify.py / detect.py 显式包含初始化、preprocess、infer、postprocess、predict。复用已有解码/绑定，未按三个函数机械拆文件 | 通过 |
| 输入与输出契约 | 两个新范例接收 BGR 数组或图片路径；异尺寸连续调用、输入不修改、路径错误、单次推理及旧别名有行为测试。其他 YOLO 任务保留原数组接口 | 通过 |
| 轻量公共 Runtime | RuntimeSession 收敛板卡身份检查、SDK 懒导入与模型构造；run 为可选透传。runner 使用同一 SDK 实例负责元信息、绑定、推理与调度。此边界经 Codex 评估接受，未新增框架/rdk_zoo 包 | 通过 |
| 目标、制品、输出协议独立 | 板卡检查先于 SDK；官方选择读取活动清单；YOLO 按 DFL/LTRB/NMS-free 协议分派，配置/元信息不符明确失败，不靠板型猜测输出 | 通过 |
| 自训练接入 | ResNet custom_selection 保留张量/类别校验，库调用无需 Manifest 注册；ResNet18 增加严格 state_dict checkpoint 导出接口。YOLO 本地模型不套默认官方标签，显式标签核对绑定类别数；空/稀疏标签有明确错误，旧 JSON/字典/列表格式保留 | 通过 |
| 完整操作路径与迁移 | 双语文档连接官方模型、自训练权重/导出/目标编译配置/Runtime、业务改造。自然 checkpoint CLI 已修复默认参数冲突。真实转换未重跑，官方已发布制品的来源缺口继续保留 | 通过 |
| 历史平台副本移除 | 活动树 platforms/ 跟踪文件为 0；活动 c_utils/字体已迁移。X3 Catalog 与历史对照读取固定 Git 对象，不改名保存 archive。两个历史 pin 均可从本地 origin/develop 历史到达 | 通过 |
| 兼容与证据 | 旧类导入/阶段别名、其它 YOLO 任务/协议保留；明确记录删除的旧 platforms 入口不兼容。受影响回归与干净检出有逐命令退出码；VLA gitlink 未改变 | 通过 |

公共 Runtime 的实际边界见 [架构说明](../../architecture/model-examples.md)；
文件/API/行为迁移见 [迁移说明](../../migration/2026-09-30-model-examples.md)。
ResNet 自训练 CLI 导出测试只验证模拟 Torch 的调用约定，不证明实际图可导出/编译。
脚本打印的 torch/torchvision 版本是当前导出环境版本；checkpoint 的训练版本仍须由训练记录提供。

## 主机证据与版本

本地执行证据目录：`local-execution/20261001-readable-model-examples`（仓库同级）。
Codex 独立检查记录为 `codex-independent-checks.json`。

- `consolidated-affected-results.json`：41 个检查任务全部退出码 0。
  其中 39 项在 `6fbd7ac2`：32 个受影响 Python 测试目录（含共享代码且排除 VLA）、
  额外 B3 测试目录，以及 6 项契约/Skill 辅助检查；2 项 Catalog 构建/检查在
  `427703b1`。每项保留实际 `verified_head`，不称所有项目在最新提交重跑。
- 关键完整套件：ResNet 72、YOLO 168、共享代码 172、YOLOE 44，均退出码 0；
  Catalog 130 测试通过，数据仍为 57 families / 812 benchmarks；
  migration 契约检查 51 samples、0 violations（帮助检查 skips 不当作板端通过）。
- Catalog 汇总首次发现 1 个失败：真实仓库误用了临时夹具的 HEAD pin。
  `427703b1` 修正真实仓库与临时夹具的来源配置选择，保留原断言，未改生产解析逻辑。
  失败 JSON/日志以 pre-catalog-fix / pre-fix 文件保留，最终记录采用修复后结果。
- `execution-clean-checkout.json`：在 `9041122f` 的本地 detached checkout 执行
  Catalog build/check、两个 main 的 help/list/dry-run、checkpoint CLI 和标签行为测试，
  10 个命令全部退出码 0。临时 worktree 已移除，ACT/Pi0 均未初始化。
  复用了现有 venv 与 node_modules，证明干净源码检出可用，不能称零依赖环境复现。
- C++ 已有主机证据为 YOLO 12、Gemma 19、迁移 file_io/内联数学检查；
  本轮后续纯 Python/文档/Catalog 夹具修改未影响其源码，因此未重复。
  限制与 SDK 耦合单元的 not-run 见 [原生验证报告](2026-10-01-readable-examples-native-validation.md)。
- `git diff bfbe6aaf --check` 无错误；platforms/ 无跟踪文件；
  ACT/Pi0 gitlink 差异为空。早期误包含的 2 个 VLA 集成测试已在实施报告披露，
  最终汇总明确排除，不把早期记录作为本轮 VLA 验收。

## 后续板端验证清单（本轮未执行）

1. 按实际支持清单选择 X5 / S100 / S600 等目标与对应制品，记录板卡身份、
   SDK 版本、模型 SHA-256、代码提交。无制品的组合不记通过；板卡身份不是模型选择依据。
2. 在板端运行官方 ResNet 和 YOLO 检测入口，核对 SDK 元信息、NV12 布局、输出绑定，
   保存结果和退出码。连续异尺寸图片核对几何与类别，不只看程序是否退出。
3. 使用已准备的自训练制品验证自定义类别与标签；不传标签时核对类别 ID，
   标签数量/协议/目标不符时核对明确失败。库调用 predict 与 CLI 输出相互核对。
4. 在具备 SDK 的环境构建受迁移影响的生产 C++ 应用与 SDK 耦合 c_utils 单元。
   本地主机头文件替身测试不替代真实 ABI/链接/板端运行。
5. 如执行真实模型转换或精度/性能复验，另列 ONNX 导出、工具链编译与板端结果，
   记录权重、训练/导出版本、校准集、编译配置与产物。它们不属于本轮为验证现有 README 的必需步骤。

本轮交付状态为本地开发分支可评审；合并、推送及板端验收尚未执行。
