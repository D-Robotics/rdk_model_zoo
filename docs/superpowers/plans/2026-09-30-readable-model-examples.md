# ResNet / YOLO 可读范例实施计划

> 执行者：用户指定的 Claude Code + GLM（2026-10-01 起在本地工作区实施，原为 sz-dev / machao）。按任务顺序实施；适用时使用 superpowers:executing-plans，不另行派生 Agent。Codex 负责制定任务和核对结果。

**Goal:** 简单 main.py 调用模型类 predict，完成 ResNet 和 YOLO 的可读、可扩展范例。

**Architecture:** 模型类保留前处理、推理和后处理的完整主线。公共层只复用 SDK/平台基础能力，继续使用现有绑定和解码器。

**Tech Stack:** Python、NumPy、现有图像处理依赖、板端 hbm_runtime；既有 C++ 和 Catalog 仅处理受影响范围。

**Spec:** [正式方案](../specs/2026-09-30-readable-model-examples-design.md)。执行前必须同时阅读方案和仓库 AGENTS.md。

状态：已于 2026-10-01 在本地实施并完成控制器两轮纠偏评审（实施分支
`codex/readable-model-examples-20261001`，工作基线 `bfbe6aaf`）。任务 1–5
对应提交：`5afdb6a5`（薄会话）、`966eeb90`（ResNet）、`eb9a33a4`（YOLO）、
`8f83c402`（文档）、`e324e356`（platforms/ 移除）；评审修复 `15a8553a`
及收尾提交见分支 git log。实施者自查及 Codex 独立核对已完成；通过范围为本轮代码、文档、历史迁移
和受影响主机检查，真实模型转换与板端验证保持 not-run。结论及后续板端清单见
[Codex 验收报告](../../releases/unified-migration/2026-10-01-readable-examples-codex-review.md)。

> 执行位置修订（2026-10-01 用户指示）：继续开发，不去服务器，都在本地进行。
> 本计划所有“sz-dev / 服务器执行 / 服务器干净 checkout”步骤改为在本地工作区
> （`rdk-b7-board-integration` worktree）用既有 `rdk_model_zoo/.venv` 解释器执行；
> “干净 checkout 验证”改为本地临时目录的 `git worktree add` 检出。其余全局约束
> （不下载权重、不跑实际导出/量化编译/板端推理、不动 VLA 子仓库、不发布、不合并
> develop）不变。
>
> 入口简约性裁定（2026-10-01 用户纠偏）：main.py 必须真正简单（约 50–80 行量级，
> 非硬性行数指标）——解析参数、必要时委托 list/dry-run 等准备模式、解析选择、
> 构造模型类、在主流程中可见地调用 `predict`、经命名辅助函数展示结果。冗长选项
> 声明与既有 list/dry-run 辅助移入示例本地 `cli.py`（YOLO 侧沿用平铺命名
> `yolo_cli.py`）；不得把旧 main 整体塞进不透明的 `run(args)`，不引入通用 CLI
> 框架；测试/调用方依赖的旧导入（如契约检查器要求 main 暴露 `build_parser`）以
> 显式再导出保留。

## 全局约束

- 在本地工作区实施及测试（2026-10-01 修订，原为仅在 sz-dev 以 machao 身份）。复用用户指定 Claude Code + GLM，使用既有 `rdk_model_zoo/.venv`，不新装环境。
- main.py 只做必要参数处理、模型构造、predict 与结果使用；禁止把算法放进入口。
- 模型类显式串起 preprocess、infer、postprocess，不按函数拆文件。
- 不新增 rdk_zoo 顶层包、通用插件或统一模型描述文件框架。
- 不初始化或修改 ACT/Pi0 子仓库，不运行 VLA 集成测试。
- 不下载权重，不跑实际导出、量化编译或板端推理；这些证据单列 not-run。
- 不触发发布、默认分支变更或新 release/tag。每个实施任务独立提交。
- 保留已有支持目标、算法、调度、C++ 能力与来源归属；范围外重构另列，不顺手扩大。

## 重点失败路径

1. 实际板卡和显式目标不一致：首次 SDK 加载前失败（任务 1）。
2. 两张尺寸不同的图片连续推理：分别还原，输入数组不变（任务 2、3）。
3. 自定义类别数和标签/输出不一致：具体报错，不能套用 COCO/ImageNet（任务 2、3）。
4. SDK 缺失时 help/导入仍可用，加载失败后不得留下假成功状态（任务 1、2、3）。
5. 删除历史目录后新克隆缺少历史对象：说明精确 fetch 需求；Catalog 不静默丢历史数据（任务 5）。

## 任务 1：薄 SDK 会话

**文件：** 新增 `samples/_shared/runtime.py`、`samples/_shared/tests/test_runtime_session.py`。读取现有 `platforms.py`、`model_runner.py` 和两个范例的 model_runner.py；本任务不改其他模型。

**接口：** `RuntimeSession(model_path: str, *, target: str)`；`load() -> None`；`run(inputs: Mapping[str, Any]) -> Mapping[str, Any]`；只读 `runtime` 返回已加载 SDK 对象，未加载时报错。run 接受/返回 SDK 原生映射。保持内部 SDK 工厂可 mock，不公开关闭硬件检查的 CLI 开关。

- [x] 在新测试中覆盖：构造不导入 SDK；目标不匹配时 SDK 工厂调用次数为 0；连续 run 只构造一次模型；run 原样传递映射；加载失败后可重试且未标记成功。
- [x] 在本地执行 `python -m unittest discover -s samples/_shared/tests -p test_runtime_session.py`，记录新增接口尚不存在时的失败。
- [x] 实现会话，复用 require_execution_target 和 HB_HBMRuntime，不引入分类/YOLO 类型，不假设 SDK 有 close 接口。
- [x] 重跑同一组测试，检查通过；核对异常保留原始原因。记录为包装单元测试，非 SDK 实机测试。
- [x] 独立提交代码和相关测试，提供 diff 与命令结果。

## 任务 2：ResNet 完整可读入口

**文件：** 新增 `samples/vision/resnet/runtime/python/classify.py`、`samples/vision/resnet/tests/test_predict_entry.py`；修改同目录 main.py、model_runner.py、model_binding.py、README.md、README_cn.md。保留 classification.py 的已有导出，其他模型使用的公共分类代码不删除。

**接口：** `ResNetClassifier(selection: ModelSelection, *, top_k: int = 5, labels: Mapping[int, str] | Sequence[str] | None = None, resize_type: int | None = None)`；`predict(source: str | Path | np.ndarray) -> ClassificationResult`。preprocess 返回现有 PreparedInput，infer 接收 PreparedInput 并调用 runner，postprocess 返回 ClassificationResult；分类后处理不需要坐标上下文。配置沿用 ClassificationTask 的参数，不增加第二套配置对象。未提供标签时结果保留类别 ID，不强制加载 ImageNet 标签。

- [x] 阅读现有分类任务、选择与标签逻辑，列出准备迁移的函数及公开参数；已有未说明行为写入任务报告。
- [x] 增加行为测试：BGR 数组不被原地改写；路径不存在明确报错；predict 调用一次 runner；自定义类别不默认使用 ImageNet 标签；两次不同尺寸输入不共享错误上下文。
- [x] 执行 `python -m unittest discover -s samples/vision/resnet/tests -p test_predict_entry.py`，确认新增行为测试失败。
- [x] 实现 ResNetClassifier；runner 采用任务 1 会话，保留原模型绑定、输入校验、输出检查与调度行为。
- [x] 核对自定义 ModelSelection.contract 的构造路径，使显式自定义分类合同不受官方资产枚举限制；保持张量、类别数、目标检查。增加非 1000 类输出用例和标签长度不匹配用例，前者返回对应类别 ID，后者明确失败；官方选择路径保持原约束。
- [x] 简化 main.py 为参数处理、模型构造、predict、结果展示。保留可用的旧参数别名，移走入口中的实际分类算法。
- [x] 更新双语 Runtime 文档，给出准确的最小导入示例及旧入口对应关系。
- [x] 执行 `python -m unittest discover -s samples/vision/resnet/tests` 和 `python samples/vision/resnet/runtime/python/main.py --help`；通过后提交。不把假 runner 结果称为 ResNet 实机通过。

## 任务 3：YOLO 检测主线与多任务兼容

**文件：** 新增 `samples/vision/ultralytics_yolo/runtime/python/detect.py`、`samples/vision/ultralytics_yolo/tests/test_predict_entry.py`；修改 main.py、model_runner.py、yolo_dispatch.py、yolo_detect.py 及对应 Runtime 双语 README。仅在必要时修改其他任务文件。

**接口：** detect.py 提供 `YoloDetect` 与 `YoloDetectConfig`，沿用现有构造参数与 DetectionResult；`predict(source, image_format="BGR", score_thres=None, nms_thres=None)` 扩展本地图片路径支持。数组路径保持现有参数语义。旧 yolo_detect.py 作为再导出/转接入口；pre_process/forward/post_process 保留别名。

- [x] 读取现有 DFL、LTRB、NMS-free 检测及其他任务分派，记录公开入口与配置。不能把这些协议归并成仅按板卡区分。
- [x] 写行为测试：连续异尺寸图像坐标恢复正确、输入不变；错误类别/输出协议报具体错误；predict 调一次推理；旧导入得到同一实现；不同协议继续选到对应任务类。
- [x] 执行 `python -m unittest discover -s samples/vision/ultralytics_yolo/tests -p test_predict_entry.py`，确认新增行为尚未支持。
- [x] 将 DFL 检测类的完整主线放入 detect.py；复用现有 decode/geometry/binding。其他协议沿用自身类，保持流程可读，不强塞进 DFL 类。
- [x] model_runner 复用任务 1 会话，保持输入输出适配、元信息校验与调度；main.py 用已有显式任务分派构造对象并调用 predict，删除入口里的算法重复。
- [x] 同步 README 的最小 main 与库调用示例，明确不同任务/协议入口和自训练支持边界。
- [x] 执行 YOLO tests 以及各现有任务的 --help。Catalog 对比用例需要生成 Catalog 时，在本地使用原有 build；缺少历史 Git 对象应记录并补取精确对象，不能改测试掩盖失败。
- [x] 全部受影响检查通过后独立提交；分类、分割、姿态、OBB 原有能力不得默默减少。

## 任务 4：完整操作路径与迁移说明

**文件：** 修改两模型根 README.md/README_cn.md、conversion 和 model 下 README 对；按需补充现有导出脚本帮助与配置说明；新增 `docs/architecture/model-examples.md`、`docs/migration/2026-09-30-model-examples.md`，更新根 README 和 AGENTS.md 导航。

- [x] 从现有源码/README 收集支持的网络、上游版本、权重格式、编译配置与产物，制作 ResNet/YOLO 两张对应表；没有证据的支持项标明限制。
- [x] 编写官方模型运行、自训练权重接入、修改业务三条路径。自训练配置使用现有配置类型与标签文件，不新造配置框架。
- [x] 按实际导出脚本记录 ONNX 与输入输出协议、目标编译产物的对应；不通过实际模型导出来验证 README。
- [x] 编写旧路径、类、函数、参数到新接口的映射，以及保留转接和明确不兼容项。
- [x] 检查新文档的本地链接、命令参数与代码签名一致；用既有检查工具限制到改动范围，不新增仅为文档搬移服务的测试。
- [x] 提交文档及必要的参数说明修改，列出没有执行的导出/编译/板端步骤。

## 任务 5：移除活动树中的历史平台副本

**文件：** `tools/catalog-publisher/sources.json`、`tools/catalog-publisher/src/sources.ts`（仅现有固定引用机制不能满足时修改）、相关 sources 测试；`platforms/`；通过引用清点确定的活动源码、文档与测试；迁移说明。

- [x] 使用 `rg -n 'platforms/' samples tools docs .github AGENTS.md README*` 清点引用，区分活动文件访问、来源链接、历史报告文字。
- [x] 核实保留平台快照的已有远程可访问提交及其路径，记录完整 SHA；不新增发布标签，不把历史副本换个目录继续保存。
- [x] 将 X3 Catalog 来源改为已核实固定 Git 引用读取，先使用现有 source reader 能力；若缺少对应配置模式，只为固定引用补最小支持。
- [x] 在 Catalog sources 测试覆盖：固定对象读取正确；对象缺失明确报错并指示获取所需对象，不能返回空成功。执行 `npm --prefix tools/catalog-publisher run check`，预期既有和新增检查通过。
- [x] 迁移其余活动文件依赖，修正指向删除目录的当前使用指南；历史报告保持历史事实并增加集中访问说明。
- [x] 删除已解除依赖的 platforms/ 跟踪文件；在本地临时 `git worktree` 干净 checkout 验证两个范例入口、相关回归与 Catalog。不得初始化 VLA 子仓库。
- [x] 更新迁移说明，提交移除改动与证据。该验证仅证明来源解析和代码依赖完整，不代表模型链路复现。

## 交付与核对

- [x] 每任务报告提交 SHA、改动摘要、运行命令、失败及修复、未验证项。
- [x] 最终确认 main.py 简单、模型类完整可读、公共 Runtime 无模型算法，旧能力和自训练路径有据可查。
- [x] 将真实导出、工具链编译、板端推理分别列为 not-run；不要累计测试数包装为模型认证。
- [x] Codex 已独立核对方案、源码、diff、41 项受影响检查的版本/退出码及最终干净 checkout 的 10 项检查，记录剩余板端验证事项。详见 [独立验收报告](../../releases/unified-migration/2026-10-01-readable-examples-codex-review.md)；未将主机证据当作实际模型转换或板端通过。

## 计划自查

需求覆盖：main/模型类由任务 2、3 实现；SDK 共享由任务 1 实现；完整范例与迁移由任务 4 实现；历史移除由任务 5 实现。五类失败路径均有归属。其他模型、C++ 全面重构、VLA 和实际模型转换不在本轮任务内。
