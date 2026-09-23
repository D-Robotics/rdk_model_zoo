# B7 独立主机评审

本轮评审对象是 develop 工作树，HEAD `16c5d04d0c71ebd160400c14fb2be1c2eb2513f0`；新增实现尚未提交。评审与实现分工：root 审 FCOS、YOLOWorld、LPRNet、MODNet 和 YOLOv5 C++；其他 reviewer 审 root 编写的 YOLOv5 Python/ByteTrack。作者自检不替代独立复核，Board=not-run、Closed=no、delivery=not-ready。

## 首轮结论：changes-required（历史记录，不能覆盖为通过）

[独立反例记录](evidence/2026-09-23-b7-independent-initial-probes.json) 保存 UTC、反例、stdout/stderr/返回码和被审文件哈希；所有 SDK 均使用临时 fake，未加载或下载真实模型。另见 [FCOS 专项独立报告](2026-09-23-b7-fcos-independent-review.md)。

| Finding | 影响与复现 | 整改要求 |
| --- | --- | --- |
| B7-LM1 | LPRNet/MODNet 的 native `RuntimeModelRunner.load` 没有实际 target 和非空/发布 digest gate；把 gate 设为拒绝仍会调用假 SDK factory 并成功 load。`bind_model` 对伪造同形状 `evil.bin` publication row 也放行。 | 真实 API 入口在加载前完成 identity/file 检查；直接 binding 比对完整发布事实；保留明确的 host injection seam。 |
| B7-LM2 | `python /absolute/.../main.py --help` 在任意 cwd 因缺少 root bootstrap 报错；MODNet `--dry-run --target x5 --ref-size 256` 却返回成功并报告 512。 | 支持已约定的直接入口；dry-run 同样拒绝无法执行的参数；文档参数/default 与实现一致。 |
| B7-LM3 | LPRNet/MODNet evaluator 仅比较用户手工提供的任意 raw/matte 文件，没有可执行的源/统一捕获链或模型/部署/输入身份。 | 自包含 same-board capture+compare，保留完整 native 输入/输出、结果、身份、部署哈希及失败记录；主机 fixture 不算真实推理。 |
| B7-YW1 | YOLOWorld evaluator 只在成功后写少量 metadata；未存 native inputs、实际 metadata、统一代码哈希、UTC/argv/cwd 或失败 JSON；源 helper 还依赖环境中的 utils 导入。 | 完整独立捕获链，明确固定源 helper；缺文件、source失败、数值不一致都可追溯；真实源函数+注入SDK验证而非返回预设相等结果。 |
| B7-YW2 | YOLOWorld 模型文件未使用统一 verify_asset_file；绑定与 native输出形状未精确锁定；vocabulary 可能引用调用方可变数组，影响后续输入。 | 完整资产/实际 metadata 与非有限值校验；snapshot词表；保留上下文隔离。正常 RuntimeMetadata.from_runtime 已拒绝多模型，弱点主要在直接 binding 构造。 |
| B7-CPP1 | S C++ guard 持有比它更早析构的局部 vector 指针，异常退栈会访问悬空对象；X5 allocation 和 shape/dtype/容量边界不完整。 | 资源 ownership/异常释放与 native tensor metadata 检查，不能仅凭代码包含 RAII 名称判通过。 |
| B7-CPP2 | 原 C++ decoder 每个 anchor 可输出多个类别，源只选 argmax；X5 默认制品、top_k=300/阈值边界以及 scheduler 选项存在未声明差异。 | source 分支行为逐项保存或明确修复原因，并给出有意义的 portable 数值测试；不把平台差异强行抹平。 |
| B7-CPP3 | CMake S600 alignment 宏、nn_math 链接遗漏；native SDK 未编译；只有结果图而无机器可复核 native dump。 | 修复本机可判断的构建依赖，补板端可执行记录路径，实际 SDK 构建/缓存/stride/ABI 和模型数字留板测队列。 |

相关 NMS、CTC 和几何 helper 本身符合 inference-contract；不以“整文件只能四函数”判违规。原始数组的有声明复制也不改变数值语义，不作为单独阻断。文件 IO、模型加载适配、绘图/保存仍必须有明确职责边界。

## 本轮 root 自检发现（不是独立验收）

- YOLOv5 S quant scale 转 tuple 后会把原 float32 计算提升为 float64，已改成保留 dtype 的只读快照；per-channel source 数值及 mutation 回归通过。
- 源 S `score_thres or default` 吞掉显式零、resize 存在对象可变状态：统一任务分别用 `is None` 和冻结的 per-call context；两个方向的几何 A/B/A 检查。
- 源 ByteTrack 的直接 post_process 参数不完整；统一 `predict` 只组合三阶段并仅执行一次 tracker 更新。frame_rate 现在实际影响 track_buffer/30，默认30保持源行为。
- 完全落入 letterbox 填充的检测框经 clipping 后可能零面积；源 Kalman XYAH 初始化除零，实际 CPU fixture 复现 NaN。统一跟踪入口剔除非正面积 person，空检测仍更新一帧；有效框源/统一真实 CPU 序列相同。捕获器对源 NaN 留下失败 JSON，不把源缺陷改写为数值通过。
- YOLOv5/ByteTrack 模型比较增加发布 hash/非空 gate；ByteTrack 比较检查 capture metadata、代码身份、原始数组文件 digest，不只相信 JSON 中的相等声明。

## 复核关闭记录

整改和逐项独立复核正在进行；本节尚无全批通过结论。未连接远程电脑、未探测板卡、未执行 OE 转换或下载真实模型。主机 ByteTrack CPU 依赖安装如批次报告披露，不把包源访问说成全程离线。
