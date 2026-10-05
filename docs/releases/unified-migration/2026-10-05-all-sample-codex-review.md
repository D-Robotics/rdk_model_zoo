# 全 51 Sample 可读 Runtime 独立验收（2026-10-05）

结论：本轮 51 个本仓 Sample 的可读 Runtime 重构完成，源码评审和主机验证通过，
状态为 **accepted_host**。49 个 Python 入口可见模型的构造或获取，以及 `predict`
调用；模型文件保留真实的阶段主线。两个仅有原生 C++ Runtime 的 LLM 保留原生
生成/流式/会话接口。该结论限定本轮代码架构，未关闭整个 X5/S 迁移的板测和发布。

分支：`codex/readable-model-examples-20261001`。基线：
`eba4adbea32e88a9398444f2a57db846113b5905`。最终核对的源码版本：
`82ab9eb4c8ea2e1558c2d2b7b69d5825e150c57e`；本报告及最终状态更新为其后的文档提交，
未再修改 Sample 或检查器源码。`develop` 与远端未变动。

实现与批次自测：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`。
独立评审与下表验证：Codex。没有服务器任务或 SSH 操作。逐样例模型、入口、后端、
特殊语义及验证计数见[覆盖表](2026-10-05-all-sample-coverage.json)，旧新调用映射见
[迁移说明](../../migration/2026-10-05-all-sample-readable-runtime.md)。

## 验收范围

- 45 个 vision、3 个 speech、1 个 robotics、2 个 llm，共 51 个本仓样例；ResNet
  范例复核并回归，MiniCPM5 原生接口已符合标准，复核保留。
- 49 个 Python `main` 均有直接 `predict` 调用。AST 对账用于防漏；实际流程另外由
  源码评审和各 Sample 行为测试验证，方法名存在本身不构成行为验收。
- `preprocess`/`infer`/`postprocess` 为首选名；旧名保留同一实现的兼容面。
  21 分类引入本地具名模型类，原 `classification.ClassificationTask` 导入保留。
  SAM 本地 canonical 视图复用共享旧名实现，不复制数值代码。
- 默认 `predict` 返回契约不变。需要合法中间产物的任务通过可选 details 返回
  当次 prepared/raw/context；文件保存与显示在应用层。测试覆盖额外 details 请求
  不重复执行生产推理、交错尺寸的逐调用上下文、量化/输出绑定和旧别名委托。
- OCR 保留有序 crop 和零检测短路；SAM、CLIP/SigLIP 保留多模型组合；Paraformer
  保留逐 utterance 的 encoder→predictor→CPU CIF→decoder 与零 token 短路。ASR
  为独立 chunk 识别，不宣称跨 chunk 模型状态。ByteTrack 保留逐视频跟踪状态；
  HIMLoco 保留离线观测→动作，不执行机器人控制。
- Gemma 将交互入口与应用会话分开；`InteractiveChatApp` 是应用类，引擎仍为
  `TextEngine`/`VisionEngine`。MiniCPM5 的原生 `Generate` 和会话重置接口保持。
  其余 Sample 的 C++/legacy 能力维持既有方式，本轮没有逐方法重写。
- 共享 Runtime 仍是 SDK 加载/目标身份边界，不推断模型算法。本轮未改变后端
  执行语义，未新增框架、顶层模型包或 `platforms/` 历史副本。

## 独立验证结果

| 检查 | 规模 | 结果与实际边界 |
| --- | --- | --- |
| 全部 Sample 主机套件 | 51 套，1871 项 | 1859 项执行通过，12 项跳过；各套退出码 0 |
| 共享模块 | 172 项 | 通过；显式排除 `test_vla_integration.py` |
| sample-contract 自身 fixture | 32 项 | 通过；canonical 拼写和 CLI 应用边界覆盖 |
| board-validation 工具 | 30 项 | 通过；工具测试，不是板端验证 |
| Skills 工具 | 63 项 | 通过；没有安装或发布 Skills |
| Catalog Vitest | 17 文件，130 项 | 通过；`npm run check` 的校验、类型、构建和可重复性检查均通过 |
| 全仓静态契约 | 51 样例 | 0 violations，87 个显式 policy skips，0 exemptions |
| 干净 Git checkout 入口 | 149 项 | 49 Python × help/list/dry-run + 2 原生 launcher help，全部通过 |
| 测试源码与最终源码对账 | 51 个目录摘要 | 测试前后与最终源码 SHA-256 均一致，无遗漏/重复行 |

合计 2298 项测试：2286 项执行通过、12 项跳过。静态契约与 149 项入口检查另外
计数，不混入单元测试总数。Paraformer 的 12 项既有可选依赖测试依赖 Torch/FunASR
或 ONNX/ONNX Runtime，本轮环境不具备这些依赖；这些测试没有被改成通过。
原有转换配方保留，不将 host fixture、export CLI 参数检查视为真实模型导出。

契约的 87 个 skip 是明确的应用入口、CLI 模块级助手、兼容文件或原生无 Python
表面等策略边界；类方法不会因位于 `cli.py` 而被豁免。0 violations 表示本次扫描
未发现违约，不表示这些跳过区域或 SDK 行为已由扫描证明。

两个原生 LLM 的主机测试包含真实生产 C++ 源码的编译和运行。Gemma 新应用源与
`main.cpp` 使用明确标注的引擎/SDK/第三方编译桩，入口检查链接主机真实 gflags；
MiniCPM5 沿用已有 OELLM 替身场景。这些检查验证可观察的应用、阶段和生命周期，
不加载真实 HBM、不执行 BPU，也未验证完整 SDK/tokenizer/OpenCV 生产链接。

## 干净 checkout 与证据版本

独立检查使用临时本地 Git clone，版本为上述 `82ab9eb4`，未初始化 ACT/Pi0
子模块。从 clone 外的工作目录运行全部入口，成功前后的 Git 状态均干净，
`hbm_runtime` 模块发现结果为 `None`。临时 clone 已清理。

这是**干净源码 checkout 复现**，复用了既有 macOS Python 3.14.7 虚拟环境及
主机编译依赖，未做依赖从零安装或 Linux CI。list/dry-run 使用各 Sample 支持的
显式目标；不支持的目标发现尝试保留在原始记录，不当作该目标通过。原生入口只
检查 help，不伪称它们提供 Python 的 list/dry-run 接口。

本地执行证据根目录为工作区外层的
`local-execution/20261005-all-sample-readable-runtime/final-verification/`：

- `stable47.json`：45 个视觉 Sample 和两个原生 LLM 的 1691 项测试；
  `remaining4.json`：语音/策略 180 项，其中 12 项跳过。
- `auxiliary.json` 与各任务 `.log`：公共模块和工具退出码、命令与输出；
  `contracts.json`：51 个 Sample 的 findings/skips。
- `clean-checkout/results.json`：149 个成功入口命令、显式目标尝试、SDK/子模块/
  Git 状态；`source-state.json`：最终 51 目录摘要与 49 个入口 AST 对账。

各 Sample 回归在不同提交时点开始；不能将开始时的 HEAD 冒充全仓冻结测试版本。
每套记录了测试前后目录摘要，Codex 再对照最终版本，51 个摘要全部一致。
摘要按排序后的 Sample 相对路径和文件字节计算，覆盖 `.py/.hpp/.h/.cpp/.cc/.md/
.json/.yaml/.sh` 与 `CMakeLists.txt`，排除缓存和依赖目录；摘要验证源码一致性，
不代替行为测试。结构化覆盖表保存每个 Sample 的摘要、计数和对应日志位置。

复跑主机检查：在满足各 Sample 已声明依赖的环境中，为每个
`samples/{vision,speech,robotics,llm}/<sample>/tests` 单独运行
`python -m unittest discover -s <tests-dir> -q`；然后运行 checker、公共/工具测试
以及 `npm --prefix tools/catalog-publisher run check`。VLA 在本轮范围之外，
不要初始化其 gitlink 来扩大验收范围。

## 本轮未执行的操作

真实权重下载、ONNX 导出、校准/量化、OE/Mapper 编译、完整板端 SDK 链接、板端
推理、数据集精度和性能复测均为 **not-run**。已有历史板测与已发布参数保留其
原版本和条件，不以本轮主机结果覆盖。ACT/Pi0 仍按用户要求排除，未改 gitlink。
本轮为本地开发提交，未合入 develop、推送或发布。
