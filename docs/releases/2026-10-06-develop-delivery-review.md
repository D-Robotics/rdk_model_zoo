# 全 Sample 重构与 Develop 交付验收（2026-10-06）

状态：**实施源码与 Develop 主机/CI 验收通过**。板端与真实模型导出/编译范围见本文边界。

实施提交为 `545a3b5874ae723663d2c817ae9ef964bc495746`。该提交已快进整合并推送到 `develop`；整合时本地、跟踪引用与实际远端均为此实施提交，工作树干净。本记录绑定该实施提交；后续文档收尾提交也须在其实际 Develop HEAD 上通过等价 CI。

本仓 51 个 Sample 已按认可架构完成重构：45 个视觉、3 个语音、1 个 robotics/HIMLoco 策略样例（离线观测到动作，不控制机器人）、2 个原生 C++ LLM。49 个 Python 入口由简洁 `main.py` 构造本地具名模型并调用 `predict`；模型文件保留可读的实际处理链，旧方法名是兼容委托。运行经样例 runner 校验绑定和张量，再复用薄 SDK session 的加载/身份边界。平台/资产解析和公共数学/IO 继续共享；模型处理主线、多阶段编排、流式与跟踪状态在 Sample 内显式表达。原生 LLM 保留 generate/stream/reset 接口。ACT/Pi0 为独立固定上游 gitlink，未纳入 51 个 Sample。

架构与逐 Sample 映射见 [架构说明](../architecture/model-examples.md)、[迁移指南](../migration/2026-10-05-all-sample-readable-runtime.md)、[历史架构验收](unified-migration/2026-10-05-all-sample-codex-review.md)。历史报告中的日期、提交、测试计数和板测条件保留原样；本记录增加本轮完整主机与 CI 证据。

## 干净克隆与入口验收

独立 clone 使用完整 Git 历史、不借用对象、detached 到实施提交；按声明依赖建立的隔离 Python 3.12 环境和 Node 22 执行默认维护者命令。只运行 `npm ci`，由维护者入口在 Python 套件前构建 Catalog；从 clone 外部目录调用 runner，未注入 PYTHONPATH、个人编译路径或板卡 SDK。

| 检查 | 实际结果 |
| --- | --- |
| Python | 58/58 套件，2315 项，0 failures / 0 errors；2 项可选导出测试跳过 |
| 可选模块 | YOLOE conversion 的 Torch export 模块缺失，明确记录，未计为通过 |
| 原生 CTest | 6 个项目，56/56 项；保持安全 SDK/生产程序开关关闭，保留适用 sanitizer |
| Catalog | 136/136 项，schema 和可重复构建通过 |
| Sample 契约 | 51 samples / 0 violations / 87 policy skips / 0 exemptions |
| 不依赖 SDK 的入口 | 149 次最终成功检查：49 Python × help/list enumeration/dry-run + 2 原生 launcher help |
| 入口范围 | 17 次预期的不支持目标探测保留；dry-run 使用接受的目标，list-models 列出可用模型/target 信息且可能忽略传入 target，不证明该目标执行；未覆盖全 target/variant |
| 源码一致性 | 6223 文件，摘要前后 `2a2b285113deda4b659cdbba1e400d9accd1c034b7b516ee37669bc82eca5e6f`，Git 状态前后干净 |

Python 与原生 CTest 是分开的计数，不相加声称唯一覆盖量。两个 Paraformer 跳过要求 Torch/FunASR；该范围未缩减、未改成通过。父仓 VLA gitlink 完整性测试包含在共享套件中，ACT/Pi0 上游未初始化或运行。历史 pin 均存在。

完整本地 report SHA-256 为 `3d223fd62a74f9795f1337fdb4d351991a9b47ac8f1c6faa9d8f1ba1ce02e1dd`；runner SHA-256 为 `b531eb8cb1fe915cd9caa00228bc799b28753ad08f7cc225a39001185a3c75ad`。入口 report SHA-256 为 `206c18ee9dfadc93e2501970f4a2daaf4b4739fe5051f6fe9ce24aa96334a799`。原始本地证据在外层工作目录的 `local-execution/20261005-develop-delivery-readiness/independent-clean-final-545a3b58/`。

## 实际 CI

以下均为实施提交 `545a3b5874ae723663d2c817ae9ef964bc495746` 的真实 Develop push 运行。

| 工作流 / 环境 | 实际运行 | 结果 |
| --- | --- | --- |
| sample-contract / check | [37360844575](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37360844575) | success |
| host-validation / macos-15 py3.12 | [37360844437](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37360844437) | success |
| host-validation / ubuntu-24.04 py3.10 | [37360844437](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37360844437) | success |
| host-validation / ubuntu-24.04 py3.12 | [37360844437](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37360844437) | success |
| Model catalog data / Validate platform sources and catalog data | [37360844432](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37360844432) | success |

三个主机报告均绑定实施提交与同一 runner：58 套件、2315 Python tests、0 failures/errors、2 optional skips、1 optional missing Torch module；6 个 CTest 项目/56 cases、136 Catalog tests、契约 51/0/87/0，源码前后干净一致。CI 的 Catalog 原始日志独立核对为 136/136，未把未解析的结构化计数字段当作证据。

| 环境 | Python | report SHA-256 |
| --- | --- | --- |
| ubuntu-24.04 py3.10 | Python 3.10.21 | `326ee58cb8531e9288ef0b05fc95195be77127f2d368789dec9939d5ed7a6ba9` |
| ubuntu-24.04 py3.12 | Python 3.12.14 | `403edb21f1b17ceb6bef746e15b75adb64ea6cf1b6faf49c168476ecf59f9347` |
| macos-15 py3.12 | Python 3.12.10 | `a81250fbd4e72b783242a6409a2690f905262cbf02f4285164de17e44789d20d` |

Catalog 数据包 SHA-256：`9059a3cd501eb00730432665b02c4eef9f001407c7be9c416a911a8612f6e389`，9171878 bytes；meta 与实际 payload 匹配。X5/S source refs 都是完整实施提交，X3 是固定历史 `6fcef2b87c12435e11fbd7327ea70d4efd917b1c`。57 条 Catalog 记录包含历史来源，与 51 个本仓 Sample 不是同一计数。

机器可读的需求矩阵、CI/源码摘要和可选范围见 [验收摘要 JSON](2026-10-06-develop-delivery-review.json)。GitHub Actions 中原始 artifact 按工作流保留期提供；外层本地原始日志不随仓库分发。

## 本轮发现与修复

保留了第一次干净克隆失败及实际 CI [37357258618](https://github.com/D-Robotics/rdk_model_zoo/actions/runs/37357258618) 的失败报告。干净克隆先后发现 runner 继承错误 cwd、Catalog 生成晚于依赖它的 Python 测试；分别以 `565ab9d8` 和 `24668498` 修复并增加回归。

`24668498` 的 Linux 3.10/3.12 CI 又定位到三处测试夹具问题：SAM 使用 Python 3.11 才有的 `contextlib.chdir`；OBB 标签断言误依赖临时目录数字；历史 YOLOv5 编译 stub 没有提供标准数学头。`545a3b58` 仅修复这三个测试文件，保留实际绘图、固定源摘要和完整翻译单元编译；没有修改运行时、模型行为、历史 pin、支持范围或增加 skip。新增 cwd 回归由本地独立隔离 Python 3.10/3.12 共享套件各 184 项验证，收据为外层 `local-execution/20261005-develop-delivery-readiness/independent-linux-fixture-fix-acceptance/receipt.json`；完整跨平台 CI 结论以本节前面的真实报告为准。

## 源码交付边界

活动 Git 树没有 `platforms/` 副本，原文可经固定历史提交访问。制品 URL、SHA、Benchmark、许可证与 VLA gitlink 保留。统一源码版本是 2.0.0 候选，平台制品与 Skills 独立版本不变。

本轮接受的是源码架构、可重复主机验收及上述实际 Develop CI。板卡推理、真实权重导出、校准、OE/Mapper 等工具链编译和真实模型精度按用户范围仍为 not-run，不构成本轮源码交付阻断。历史板测条件、MODNet manual 资产、S100P/SAM/ByteTrack 等已知缺口保留，不能外推为所有目标通过。

Main 提升可使用同一已验收源码，不需再次重写 Runtime，流程见 [统一源码候选与支持矩阵](unified-source-release.md)。本轮未创建 main、tag、Release，未切换默认分支或发布网站。
