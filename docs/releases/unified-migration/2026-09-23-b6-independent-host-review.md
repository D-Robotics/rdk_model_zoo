# B6 独立主机评审

review_decision=pass（host）；Board=not-run、Closed=no、delivery_readiness=not-ready。基于 develop 未提交工作树，base=`16c5d04d0c71ebd160400c14fb2be1c2eb2513f0`；[96 个文件的精确快照](evidence/2026-09-23-b6-working-tree-snapshot.json)固定本次被审版本。106 个源副本逐文件复核不变。

## 独立性与审阅范围

这是分组件交叉评审，不把实现作者自检称为独立验收。`dinov2_contract_audit` 实现 stages/tensor IO 后，由主任务对照四套源数值和公开 API 审阅；该 reviewer 反向独立检查主任务实现的 binding/runner/evaluator，另补转换数值回归。`siglip_docs` 实现 conversion 脚本，由主任务逐项核对源能力；该 reviewer 再独立审查主任务改写的 conversion README。`review_b4_first_four` 实现样例入口及首版客户文档，由主任务审阅；该 reviewer 独立复核主任务最后的双语文档、dry-run、prompt 边界和性能命令。主任务汇总以下实际返回结果。

## Findings 与关闭证据

| Finding | 修复及确认 |
| --- | --- |
| Torch 导出 wrapper 未注册子模型权重 | EfficientSAM S encoder 和 MobileSAM decoder 使用 Module 注册子模型；mock Module forward/注册契约验证，主任务源审查关闭。真实 Torch/ONNX 导出仍未运行。 |
| 转换迁移遗漏下载 checkpoint、clone 失败的 ZIP fallback、S embedding dump | 恢复真实源能力；下载仅 mock；ZIP 私有目录及路径检查；S dump 与源 fake ORT 输入/输出逐值相等；主任务复核关闭。 |
| X5 EfficientSAM builder 忽略自定义 checkpoint | 明确拒绝非默认 checkpoint 路径，S 保留显式参数；模拟回归通过。 |
| Frozen binding 的嵌套 metadata 可变 | 深冻结快照、数组只读，调用方修改不能改变契约；独立 reviewer 确认。 |
| 运行前模型摘要读取失败不保留 evidence | 进入执行范围后先创建证据目录，异常写 comparison.json/return_code=2；缺模型回归及独立 evaluator 复核通过。 |
| EfficientSAM README 误称 box、共享 API 静默忽略 box | 双语声明导出时固定双正点，predict 不接受 box；显式 decoder pre_process 也拒绝非空 box。四 target 拒绝回归先失败再通过；独立 reviewer 复跑包含绑定/阶段/evaluator/入口的 44 tests，确认关闭。MobileSAM 默认框数值保留。 |
| MobileSAM dry-run 将未加载的 X5 box shape 写死 | 改为候选形状与 requires runtime metadata；新增回归，独立 reviewer 实际 dry-run 与 9 项入口/README检查通过。 |
| 转换 README 单 config 未带 target，中文误写框扰动，缺逐配置输入 | 双语修正，精确 S100 config 示例；16 YAML 与四份逐配置 input_name/shape/type 表逐值核对，模拟编译分派为 hb_compile；独立 reviewer 关闭。 |
| evaluator 丢失源单模型性能测试方法/测量条件 | 恢复两 stage×四 target shell、线程/core、Params/FLOPs 和历史测量定义；独立 reviewer 用 fake hrt 捕获中英脚本，确认模型路径和所有 argv，未执行真实 SDK。 |

## 验证与文档路径

[回归原始记录](evidence/2026-09-23-b6-local-regression.json)：初轮全量 781 tests 通过；最终修改后 shared 101、EfficientSAM 19、MobileSAM 17 通过，当前覆盖 783。30 samples / 0 violations / 33 policy skips / 84 原有 B9 豁免。153 个本地 README 链接无缺失。没有增加宽泛豁免。

真实源 fixture 检查包括四套预后处理、阈值差异、原生输出 dtype 不变、无隐式 dequant、六阶段与 predict、A/B/A context 和 SDK buffer reuse、固定/运行时提示边界；转换 fixture 覆盖两样例×X5/S 校准、两套 S dump。fake ORT 不是实际模型执行；此前 reviewer 初次未看到新增 shared fixture 的判断已撤回，不是产品缺陷。

客户与 Agent 路径均可从根 README 定位准备、运行、Python API、转换和评估。命令区分主机/板端/工具链，历史结果与本轮证据分开。转换缺固定 upstream revision、checkpoint digest、代表性校准集，真实模型 native metadata 未读取，均明确披露。未验证条件没有升级为支持实测。

## 结论边界

无未关闭主机阻断项，可继续 B7 本地开发。按用户要求跳过板端环境验证，批次保持 Closed=no；B1/B2 历史证据不变。未执行板卡、远程、真实模型下载、OE 或真实 ONNX Runtime。未提交、push 或发布。实际板端复验、转换校准及源材料缺口补足仍在交接清单内。
