# B3 整改复核与主机侧推进记录

复核基点：`16c5d04`。原始独立评审保留在 `2026-09-22-b3-independent-review.md`，本记录追加当前结论，不改写历史。

## 结论

B3-R1–R4 作者整改已复核。主机侧可继续后续批次；B3 Board=not-run、Closed=no、delivery_readiness=not-ready。本次没有执行板端推理、下载模型或验证 OE 转换。依据用户最新要求，板卡不可用不再阻塞后续主机开发；发布验收仍须完成板端复验。

| finding | 本次核对 | 结果 |
| --- | --- | --- |
| B3-R1 | FasterNet parser/API 默认 s；FastViT parser/API 默认 s12；两个 parser→main 回归测试实际执行 | 原 finding 修复确认 |
| B3-R2 | 下载示例改为合法小写变体；FastViT 双语 evaluator 使用已有 bucket.JPEG，成功说明对应水桶 | 原 finding 修复确认 |
| B3-R3 | FastViT 双语表恢复 SA12/S12/T12/T8 与固定源历史数据，明确非本轮重测 | 原 finding 修复确认 |
| B3-R4 | 四份英文 evaluator 改为拟采用的 B3 判据与 B2 先例，结果仍 not-run | 原 finding 修复确认 |

额外检查发现 EdgeNeXt、FasterNet、FastViT 中文 evaluator 留有无关 EfficientFormer L1 的 67.72%/76.75% 说明；本次直接删除。该三处属于 Codex 作者修正，不伪称由另一个独立 reviewer 审核。原 R1–R4 的作者与复核者仍独立。

## 验证

本次实际执行：ConvNeXt 28、EdgeNeXt 26、FasterNet 28、FastViT 28，共 110；shared 71、checker 27，总计 208 tests，全部成功。CI 同命令 15 samples / 0 violations / 84 exemptions。完整命令、时间、返回码和 stdout/stderr 保存于 [证据](evidence/2026-09-22-b3-host-recheck.json)。未重跑其他 sample 套件；本次没有修改共享 runtime。HEAD 中另含 YOLO26 热修复，本记录不构成对该修复的验收。

## 下一步

按 B4→B11 推进源清点、逐 sample 重构、双语文档、主机验证和评审。板测与 OE 校准按 [主机推进与板端待办](2026-09-22-host-development-and-board-handoff.md) 留存，绝不把待办写成通过。最终旧目录删除仍受能力完整性与兼容窗口条件约束。

## 后续文档复核补记

继续 B4 时发现前述 R2 复核未覆盖所有运行说明：FastViT runtime 双语 prose 与 variant 描述仍写 S/T0/T1/T2，FasterNet 使用大写 CLI 变体，根 README 还保留 26 项旧计数。本轮已由 Codex 修正为实际 s12/sa12/t12/t8、s/t0/t1/t2 和 28 项；新增修正属于作者工作，不能以原独立复核结论替代其检查。此前“R2 修复确认”仅完整覆盖下载命令与图片路径，对全部 prose 的表述过宽，现以本节补正。
