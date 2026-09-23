# B4 独立主机评审（持续记录）

范围：独立 reviewer `review_b4_first_four`；本地未提交工作树（基于 16c5d04d0c71ebd160400c14fb2be1c2eb2513f0），不是提交范围签署。板卡与网络未访问。八个 sample 的独立主机评审已完成，可继续主机迁移；delivery_readiness=not-ready，Board=not-run，Closed=no。

## 前四 sample：RepGhost / RepVGG / RepViT / MobileOne

独立复核确认以下原 findings 可关闭：

- 根 README 支持矩阵已按 target × variant × Python/C++ 展开，三态及双语一致，明确尚无板端证据。
- 算法说明、固定源论文和官方仓库链接恢复；只验证静态来源和格式，未进行在线存活探测。
- 根/runtime 的测试图、默认参数、API 示例及目录说明一致：ibex.JPEG / gooze.JPEG / yurt.JPEG / tiger_beetle.JPEG。
- 前提区明确实际主机依赖版本，未把未验证的板端镜像/SDK/资源写成承诺。

独立执行 `.venv/bin/python -m unittest discover -s <sample>/tests`：7/9/9/9 项通过；shared 71 项通过。reviewer 修正初始测试结论：共享层已覆盖 predict 与显式三阶段等价、交错输入 context 隔离；逐 sample 真实 binding 的同类增强为 P2 建议，非主机阻断，尚未新增。

复核另提两项 P2：三份根文档固定源 SHA 截断、中文 runtime target choices 未列全。作者已同步修正到八个 sample（完整 X5 SHA、auto/x5/s100/s100p/s600），本次作者整改不冒充 reviewer 的再次确认。

## 后四 sample

ResNeXt / VargConvNet / GoogLeNet / HGNetV2 独立评审已完成，没有 P0/P1 主机阻断。源能力、绑定、下载、runtime、五级双语文档、conversion/evaluator 均在范围内。独立套件 9/9/9/15 项通过，四 sample 规范检查各 0 violations；本地链接可解析、双语章节锚点一致、转换配方/导出脚本与固定源副本逐字节一致。

HGNetV2 evaluator 确认真实复用统一任务/运行器，CSV/BOM/嵌套路径、非法输入、成功分母、覆盖率、Top-K 命名、部分失败与退出码均有主机验证；evaluator 默认 resize=0 与 runtime=1 的差异已披露。未跑真实模型/数据集，不能证明精度与吞吐。

非阻断 P2：各 sample 专属 predict/context 场景增强仍未新增；四份绑定 docstring 源 SHA 缩写已由作者统一完整值。其他 sample、板端、真实模型下载/加载、OE 和数据集性能不在此次独立范围。

review_decision=pass（本地主机范围）；delivery_readiness=not-ready。Board=not-run、Closed=no。

## 逐样例 predict/context 增强收尾（2026-09-23）

八个 sample 各新增一个遍历全部发布变体的测试，校验显式三阶段与 predict 结果相同，A(17×31)→B(29×11)→A 的原图尺寸、transform/context、输出和先前输入张量不被后续调用覆盖。实际使用各 sample 的真实 binding，并非仅依赖共享层 fixture。没有改动生产代码。

`review_b4_first_four` 编写测试，主任务另行检查并完整运行八套测试：RepGhost 8、RepVGG/RepViT/MobileOne/ResNeXt/VargConvNet/GoogLeNet 各 10、HGNetV2 16（含原有 6 项 evaluator），合计 84 全部通过。该补充测试的作者不是其独立签署人；复核由主任务完成。原先“尚未新增”的 P2 描述保留为历史，在此关闭。命令、完整输出及测试文件摘要见 [增强证据](evidence/2026-09-23-b4-context-coverage.json)。

本次只增加主机覆盖，Board=not-run、Closed=no 不变。
