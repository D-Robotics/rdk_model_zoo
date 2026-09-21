# B1 独立评审 — 2026-09-21

> **最新独立复审（Codex，HEAD dd60911）：changes-required / not-ready，Closed=no。** R2–R6 已关闭；R1 文件保留已通过，双语转换 README 尚有执行目录错误与 scale 一致性错误。见 [整改独立复审](2026-09-21-b1-independent-rereview.md)。历史内容和原始结论保留。


- Reviewer：Codex（独立于 B1 实现作者 Claude）
- review_decision：**changes-required**
- delivery_readiness：**not-ready**
- Base：c9e6bb8；Head：c218e8622a2cb2025b6807ce2d36ef5c3df3d500。
- 本报告完成独立评审动作，不表示评审通过。X5 8GB 由用户安排原执行者补测；即使补测通过，下列必需整改仍须关闭后才能进入 B2。

## Findings

### B1-R1 [P1 / major] ResNet152 转换能力未迁入，ResNet50/152 转换说明未接入

- Axis：Delivery Specification；confidence：confirmed；relation：introduced（迁移遗漏）。
- Location：`samples/vision/resnet/conversion/README_cn.md:1` 及同目录文件清单；英文同样仅介绍 ResNet18。
- Evidence：`rdk_s@380e1a2` 的 resnet152/conversion 有 `get_calibration_data.py`、`resnet152_config.yaml`、`x86_inference.py`；统一目录仅有 `export_resnet18_onnx.py` 与两份 README。ResNet50 的原转换说明也未在统一文档提供变体入口。源平台目录尚在，所以目前不是历史文件已被删除，而是 B1 宣称的全能力迁移未完成。
- Rule：计划 B1、Q1 conversion 内容要求与旧能力保留要求。
- Impact：新入口能推理 50/152，但客户/Agent 无法在统一 sample 找到相应转换材料；按当前 done 台账继续收尾会遗漏已有能力。
- Fix：在 resnet/conversion 内按变体安置/适配原材料和文档，记录来源，保留 ResNet18 现有流程；补逐能力旧→新映射和文件保留检查。缺少上游配方的部分仍如实声明，不要求凭空补造编译流程。
- Acceptance：逐个源文件有经审核去向；中英 README 明确 18/50/152 可复现范围、配置/输入/输出；相关主机/路径测试通过。

### B1-R2 [P2 / major] MobileNetV2 C++ 启动器遗漏 S100P 的已登记身份形式

- Axis：Technical Correctness & Regressions；confidence：confirmed；relation：introduced（新增门禁不完整）。
- Location：`samples/vision/mobilenetv2/runtime/cpp/run.sh:14-24`。
- Evidence：启动器只读 soc_name，s100 直接放行并选 s100 模型；`docs/release/platforms.json` 明确登记 `soc_name=s100 + board_type=rdk s100p/s100p` 为 S100P。主机最小复现：共享 Python `match_target('S100','RDK S100P')` 返回 s100p，而提取的 shell case 在 SOC_RAW=s100 时 exit 0 放行。没有连接板卡，也未声称本次实测板恰好使用这个别名。
- Rule：无静默回退、目标身份按 boardinfo/board_type 识别；C++ README 声称拒绝 S100P。
- Impact：采用该已知身份形式的 S100P 会被启动器当作 S100，选用未声明支持的制品；现有 Python 负例不能验证 C++ 门禁。
- Fix：C++ launcher 使用与共享身份规则一致的 board_type 细分，检查在构建/模型加载前完成；无须引入推理 SDK 来识别身份。
- Acceptance：主机 shell/identity fixture 覆盖 S100、S600、显式 S100P、S100+S100P board_type、缺失/未知身份；S100P 两种形式均拒绝，S100 正例仍可执行。

### B1-R3 [P2 / major] V3/V4 校准文档没有给出与各 YAML 匹配的实际步骤

- Axis：Repository Standards / Delivery Specification；confidence：confirmed；relation：introduced documentation / exposed source limitation。
- Location：`samples/vision/mobilenetv3/conversion/README_cn.md:40`、`samples/vision/mobilenetv4/conversion/README_cn.md:41`，及英文对应章节。
- Evidence：V3 脚本固定输出 calibration_data_bgr，X5 YAML 要 calibration_data_rgb_f32；V4 脚本默认仅生成 BGR 224，medium S YAML 要 BGR 256，X5 要 RGB；V4 X5 medium YAML 还读取 mobilenetv4_conv_medium_deploy.onnx，而 README 导出步骤只产生 mobilenetv4_conv_medium.onnx。当前段落称脚本生成 YAML 消费的数据，却省略脚本硬编码输入目录、256 切换和 RGB 配方缺口。
- Rule：README 命令五要素、已知缺口必须具体、不能把同名材料拼接为可操作配方。
- Impact：照文档导出/编译会遇到不存在的输入或校准目录；仅给目录改名也不能保证 RGB/BGR 和数值变换正确。
- Fix：按 target/variant 列清 ONNX、校准布局/数值、输入目录、生成步骤与 YAML 关系。确有源步骤则具体化；无法证明的 X5 RGB 校准、medium deploy 图明确作为缺失前提，不能提供无条件可执行的完整链路，也不能把重命名当数值修复。
- Acceptance：静态路径/参数对照无矛盾；每条配方标明可执行范围与缺失步骤；不把未跑 OE 说成转换通过。

### B1-R4 [P2 / major] V4 转换验收文档把 medium 的 S 输入写成 224

- Axis：Technical Correctness & Regressions / Documentation；confidence：confirmed；relation：introduced。
- Location：`samples/vision/mobilenetv4/conversion/README_cn.md:72-73`，`README.md:81-83`。
- Evidence：两语言验证章节统一写 Y=[1,224,224,1]、UV=[1,112,112,2]；同一文档前文、binding 及板测记录均明确 S medium 为 256，正确为 Y=[1,256,256,1]、UV=[1,128,128,2]。
- Impact：客户或 Agent 按错误 shape 验收时会错误拒绝正确 medium 制品，或误改转换配置。
- Fix：按 target×variant 表达验收形状，并加入 README 关键 shape 与 binding 契约一致性检查；不能仅检查章节存在。

### B1-R5 [P2 / major] 板后证据未同步回客户 README 与分维度台账

- Axis：Repository Standards / Delivery Specification；confidence：confirmed；relation：introduced（板后更新遗漏）。
- Location：MobileNetV1–V4 根 README 的 support-matrix、evaluator 的 reference-results；如 `samples/vision/mobilenetv1/README_cn.md:14`、`mobilenetv2/evaluator/README_cn.md:72`；ResNet 根新变体行；MobileNetV2 C++ README 末段。中英文均需核对。
- Evidence：根文档仍称全部板测未执行；evaluator 仍列 14 测试和板测 not-run，而最新证据为 17 测试及三板结果；C++ 文档仍说比较未执行。台账把 x5+s / python+s cpp 聚合成 passed 再括号写 x5-8g not-run，且 Review passed 来自作者自评。报告把独立审阅推迟收尾，违反逐批门槛。
- Impact：客户和 Agent 得到互相冲突的支持/实测事实，自动门禁也无法按 target×variant×language 判断完成。
- Fix：按证据更新已验证范围与引用，X5 4GB/8GB 分开；S100P 只记负例，S600 C++ 保持 not-run，不扩大声明；拆分不同状态的台账行。独立审阅以本报告为准，修复前不能 Review=passed。
- Acceptance：中英文档、台账和最新 evidence 一致；作者历史报告保留并显式标注被新记录取代的结论。

### B1-R6 [P2 / major] B9 文档延期没有在 CI 门禁中落实，当前工作流持续失败

- Axis：Technical Correctness & Regressions / Delivery workflow；confidence：confirmed；relation：pre-existing / exposed（不归咎于 B1 模型实现）。
- Location：`.github/workflows/sample-contract.yml` 的 Check migration scope 步骤；B1 报告 §6.3。
- Evidence：原样执行 `python tools/sample_contract/check.py --scope migration --parser-mode import` 返回 1，7 samples / 84 violations / 10 skips；84 条来自已约定 B9 处理的 ultralytics_yolo README。B1 单样例检查均通过，不能因此宣称 CI 门槛已通过。
- Rule：计划已明确 YOLO 文档延至 B9，同时要求新增/变更规则回归已迁移样例；延期必须有可审计边界。
- Impact：每次修改 samples 都触发持续红灯，无法把新增违规与已接受历史欠账区分。
- Fix：在 checker/CI 明确实现已批准的有限欠账基线或等价的分阶段规则范围：只容纳这些既有 README 规则问题，新增违规、B1 违规及 YOLO 其他规则仍失败，B9 必须移除延期。禁止 continue-on-error、整体排除 YOLO 或降低全仓门槛。
- Acceptance：CI 同一命令通过；新增违规负例仍失败；旧问题消除时检测过期基线，不能永久吞掉。

## Scope 与限制

本轮为 B1 change-review 并扩展核对 sample 交付：共享分类 task/runner/binding/tensor IO、四个 MobileNet 的绑定/CLI/下载器及测试、ResNet 变体/下载/转换目录、MobileNetV2 C++ launcher/CMake/main 与资源实现、各级关键 README 内容及源分支 conversion 清单、Manifest 覆盖、checker/CI、台账与作者证据。不是全仓完整审计。没有运行 OE、下载模型、连接板卡、执行完整数据集精度评估或 C++ 编译。图片内容和上游外链未逐项视觉/在线验证；源脚本内所有算子/数值未作工具链认证。

规则来源：用户批准的 golden-scribbling-micali 计划 Q1–Q5、逐批独立评审要求；仓库 `docs/sample-standards/{readme-contract,inference-contract}.md`。保留原有能力不等于要求编造源仓库缺失的配方。

## Passed checks（本 reviewer 实际执行）

- 四个 MobileNet 各 17，ResNet 46，OCR 44，YOLO 59，shared 71，checker 24：共 **312 tests，全部通过**。
- 五个 B1 sample 分别使用 `--parser-mode import`：0 violations；MobileNet 各 1 个 CLI policy skip，ResNet 2 个 CLI/legacy skip；这些 skip 不等于整个模块已审计通过。
- 代码结构抽查：classification.pre_process/forward/post_process/predict 职责分离，模型下载留在 model 层，可视化留在 CLI；公共辅助函数与任务有关，未发现把下载/NMS/文件写入塞入本批 Python forward 的问题。
- B1-D1 修复依实际 float32 dtype 放行，描述符不再次应用；B1-D2 明确转发 variant。相关主机回归通过；板上复验来自作者证据。
- 主机最小探针确认 B1-R2 的 Python/shell 身份判定不一致；源文件清单确认 B1-R1。
- 全量 checker 的失败结果也被保留，未挑选性忽略。

## 板测矩阵（来自作者报告，本 reviewer 未亲测）

| Target | Python | C++ | 证据边界 |
| --- | --- | --- | --- |
| X5 4GB | 5/5 | B1 MobileNet 无 X5 C++ 交付 | 固定输入 Top-K 对照与状态隔离 |
| X5 8GB | not-run，交原执行者补测 | 不适用 | 本报告冻结时尚无结果 |
| S100 | 7/7 | MobileNetV2 构建+运行通过 | C++ 不能据此声称完整数值对照通过 |
| S600 | 7/7 | not-run | 原批次只要求 S100 代表 C++；不扩展为 S600 C++ 已验 |
| S100P | 两项负例通过 | not-run | 没有 B1 发布制品的正向推理验证；不等于支持声明 |

数值证据主要是 Top-K class IDs、score 容差、labels（部分为 null）和隔离稳定；不能扩写为完整 raw 张量/全部预处理逐位等价或全数据集精度通过。板测 JSON 记录了 bundle、输入和制品 hash，但 runner/原始完整日志部分在 /tmp；交接宜归档可重放 runner 与完整命令，并明确其对应代码，而不是只保留摘要。

## 关闭条件与下一步

1. 原执行者修复 B1-R1–R6，逐项提供修改位置与回归证据。
2. 完成 X5 8GB 既定板测，保留其他已通过板位事实；行为改动只重跑受影响范围。
3. 更新双语 README 与拆分台账，作者自检/独立审阅分开。
4. 独立 reviewer 复核修正后将本报告 findings 逐条关闭，记录新 HEAD；只有 review=passed 且所有 required 项通过才 Closed=yes 并进入 B2。

新增问题不推翻 sample 中心和深度重构方案；说明目录/标题/单元测试通过还不足以证明完整交付。
