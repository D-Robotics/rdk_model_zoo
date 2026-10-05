# 全部本仓 Sample 的可读 Runtime 设计

日期：2026-10-05。用户已认可 ResNet / YOLO 架构，明确授权按同一标准完成全部 Sample。实施基线 `eba4adbea32e88a9398444f2a57db846113b5905`；沿用本地开发 worktree，不改 develop，不推送。

## 目标和覆盖

覆盖 `samples/vision`、`samples/speech`、`samples/robotics`、`samples/llm` 下 51 个本仓维护样例。49 个具有 Python Runtime，两个 LLM 只有原生 C++ Runtime。ACT / Pi0 是独立 gitlink，继续按用户要求排除。ResNet 已完成，本轮复核并回归。其他语言保持已有能力，不增加假 Python Runtime，不重写 SDK 引擎。

## 已认可的标准

1. Python `main.py` 是薄入口：解析参数、处理 model-free 模式、构造模型对象、调用 `predict()`、展示结果。参数声明、展示、视频/音频读取等应用辅助工作可移到本地 `cli.py`；不得把业务代码机械拆成一函数一文件，也不得只把整个旧 main 原封不动搬到 cli 并让入口隐藏模型构造。
2. 模型文件内可读初始化和 `preprocess`、`infer`、`postprocess`、`predict` 的真实主线。既有 `pre_process`、`forward`、`post_process` 保留兼容委托。可复用真正的张量、数学、绑定和协议实现，不建立新顶层包、插件、通用注册框架。
3. Runtime 保持已有薄会话 / runner 边界，不让模型算法进入 SDK 会话。目标身份、编译制品、输出协议分别解析并严格校验；保持 SDK 懒导入和主机 model-free 入口。
4. 单图分类按照 ResNet 范例提供本地具名模型类，完整三步可见，不是空子类或共享类的转出。官方 Manifest、变体、预处理/量化约定和默认参数保持原有事实；不扩大自训练导出支持。
5. 多阶段任务展示真实编排。SAM 保留 encoder → prompt/decoder；OCR 保留 detect → crop → recognize；CLIP/SigLIP 保留图像/文本匹配；Paraformer 保留 encoder → predictor → CIF → decoder。不能为外观三段式把 stage IO 弄错或重复推理。
6. 跟踪、策略、语音和生成保留状态、输入长度、会话及流式语义。`predict` 本身不主动打印、绘图或写文件；显式流回调和结果结构保持兼容。原生生成 API 如 `Generate` 可保留，增加可读 facade/映射时不得制造不支持的板卡或统一假协议。

## 兼容和文档

保留 CLI 参数/默认值/返回码、公开构造参数、结果结构、现有导入路径及别名。样例 runtime README（中英已有版本）、模型 integration/stage 导航与迁移说明同步，不改已有 ONNX/量化配方事实。主架构文档和全仓迁移表给出入口、模型类、实际后端、特殊调用以及旧新接口映射。已符合标准的代码以审核证据覆盖，不要求无意义改动。

## 验证边界

每个批次先运行所属基线测试，再以无板端 SDK 的可注入 runner 验证真实 preprocessing/postprocessing、预测与手动阶段的结果一致、连续调用几何/状态正确、执行次数和错误传播正确。对新增接口先有能体现缺失接口的失败测试；不为机械搬文件堆砌 AST-only 测试。保留并运行现有 host/native fixture 测试。

最终逐样例执行现有测试、SDK-free help/list/dry-run 可用项，公共模块与契约检查器回归、Catalog build/check 以及干净 checkout 入口复现。每条结果记录命令、退出码、对应源码版本与日志，不凭执行器的自述验收。不下载真实权重、不做真实 ONNX 导出/OE/Mapper/HMCT 编译；板端 SDK 链接/推理明确 not-run，主机 fixture 通过不能替代这些证据。

## 执行责任

Codex 制定方案、分派并评估；本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]` 实现和测试。每批新上下文，不访问服务器。互不重叠的样例可并行修改；公共文件和最终集成串行。执行器不得全仓 git add/commit，Codex 在评估后按路径提交，防止共享 index 混入别批工作。

实施状态：2026-10-05 完成全部 51 个本仓 Sample 的本轮源码架构与主机验收。
结果与边界见[独立验收](../../releases/unified-migration/2026-10-05-all-sample-codex-review.md)。
