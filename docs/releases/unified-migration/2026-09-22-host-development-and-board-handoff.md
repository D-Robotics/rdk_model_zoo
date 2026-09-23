# 主机推进与板端交接（2026-09-22）

用户最新要求：当前没有板卡环境，由 Codex 接手基础开发、分支内容合并和 README 等文档评审，最终留下板端复验、校准任务。该要求覆盖历史“每批板测完成才可进入下一批”的排期限制，不改变已发布支持事实与最终验收标准。

## 执行规则

- 在当前 develop 工作树继续 B4–B11，逐 sample 深度重构，不机械复制两侧目录。每批保留源 SHA、旧新函数映射、能力/语言/制品清单、转换材料及对应事实。
- 每批先通过主机验证与代码/文档评审，再进入下一批。Board 未执行仍为 not-run，Closed 不因主机通过自动改 yes；主机推进与客户可交付状态分开。
- README 必须逐 sample 核对合法参数、默认值、输入、输出、平台、变体与真实基准来源；禁止复制相邻模型的数值和已验证声明。
- 任务模型只承载前处理、推理、后处理及 predict 组合；下载、标签加载、绘图、保存、SDK 适配分别归位。共享抽取要求至少两个真实消费者和一致协议。
- 用户进一步明确不再使用远程电脑：所有开发、评审和测试在本地进行；板端环境验证先跳过，不发远程任务、不探测板卡、不重试授权。代码留在本地；不 push、发布或切换默认分支。
- 同一作者新增实现的主机自审如实标为自审；不得冒充独立评审。保留最终独立复核待办。
- 尚未覆盖的板卡、OE 工具链和校准集明确标为缺少证据；不捏造可重现转换流程。删除旧兼容目录前必须完成能力与引用核对，不能因排期变化提前删除基线。

## 当前板端待办

| 批次 | 范围 | 用户恢复环境后的工作 | 当前状态 |
| --- | --- | --- | --- |
| B3 | X5 8GB/4GB；ConvNeXt atto；EdgeNeXt base/small/x_small/xx_small；FasterNet s/t0/t1/t2；FastViT s12/sa12/t12/t8 | 同板同模型同输入记录固定源与统一入口，完整 Top-K/分数与部署哈希；默认入口与明确变体均验证 | not-run，13 变体/板；无需 S 正向推理 |
| B4 RepGhost | X5 8GB/4GB；100/111/130/150/200 | [双语 evaluator](../../../samples/vision/repghost/evaluator/README_cn.md) 含默认入口、源 API 对照与 raw 数组保存命令；记录版本/制品/输入身份，按变体分别复测 | not-run；主机开发已完成，板端 metadata/数值待验证 |
| B4 RepVGG/RepViT/MobileOne | X5 双板；6/3/5 变体 | 各 sample evaluator 双语文档含源任务 API 与统一入口对照、完整 raw 向量保存；逐变体执行，记录部署/模型/图片哈希及完整输出 | not-run；主机侧开发完成 |
| B4 ResNeXt/VargConvNet/GoogLeNet/HGNetV2 | X5 双板；1/1/1/5 变体 | 各 sample evaluator 的源 API 对照；HGNetV2 另执行真实数据集评测，核对覆盖率和 resize=0；校准依据各 conversion 前提 | not-run；主机开发与评测接入完成 |
| B5 ViT | S100 / int8、int16 / Python | [evaluator](../../../samples/vision/vit/evaluator/README_cn.md) 提供同模型同图源/统一对照，保存完整 raw 数组；两变体分别跑默认及十张 CIFAR 图片，记录部署/制品/输入/标签身份 | not-run；主机实现 13 项通过，独立主机评审通过 |
| B5–B11 | 以各批次台账的 target×variant×language 为准 | 实现完成后逐批生成准确命令、模型/输入身份、预期输出、基线和接收判据 | pending；不能提前宣称开发完成 |
| 转换/校准 | 各 sample conversion 已发布材料与已披露前提 | 在匹配的 OE 环境补齐权重/导出图/校准数据后转换，核对 metadata 再进行数值/精度校验 | not-run；不是所有 sample 都需要重新量化 |

B1/B2 既有有效证据继续保留；只因行为或制品改变扩大复验范围。S600 MobileNetV2 C++ 仍按用户既定决定 not-run，不重新加入必测。

最终交接必须按实际实现逐项补齐精确命令与证据要求；此表目前是持续维护队列，不是完成声明。

### SigLIP 待板端复验

S100/S100P × 八变体 × pooler_output/last_hidden_state；核验同一发布 HBM 的板端 digest/SDK metadata、默认入口和显式子模型、完整 raw shape/dtype 与源 API 输出，并保存部署文件/输入摘要及完整日志。源协议允许 pooler 的两种记载形状，实际绑定后 shape 必须固定；384 patch14 hidden 为 729 tokens。待执行流程见 [SigLIP evaluator](../../../samples/vision/siglip/evaluator/README_cn.md)。当前全部 not-run，不以历史性能表或主机 fixture 代替。

### B5 其余板端复验队列

- **CLIP/X5 8GB与4GB**：精确img_encoder.bin+text_encoder.onnx，实际ONNX文本metadata/CPUExecutionProvider和BPUmetadata；源/unified同输入/BPE的两路raw、cosine和rank、默认与自定义文本、标注图。外部paths身份拒绝已主机测；真实资产/性能not-run。完整raw对拍步骤在sample evaluator。
- **DINOv2/S100、S100P、S600**：各march独立HBM，验证实际双输出dtype/quant descriptor，保存源/unified输入、cls_feat/patch_feat raw和反量化结果及JSON。对拍步骤会保留部署/模型/图像digest；历史ONNX精度不等于迁移一致性，仍须另外在OE准备浮点ONNX/真实校准集及精度对照。
- **3DResNet/S100**：同一video0.npy和400类JSON，实际5D输入及唯一400-score输出metadata，完整raw/softmax Top-K/archery历史预期；不存在源完整视频抽帧或可重现转换材料，补足后才能扩展承诺。

上述队列均 not-run。这里只记录后续任务，不连接板卡、不使用远程电脑。

## B6 SAM 待板端执行队列

本节只提供后续复验步骤，当前未连接板卡、未下载或运行模型。每种 sample 在 x5、s100、s100p、s600 各有一组独立 encoder/decoder 配对；X5 8GB/4GB 分开留证，共十个板位×样例组合。Host 证据与 board 证据不可混用。

1. 按对应 sample model README 显式准备该 target 两个制品（发布 SHA 均未知，保存观察到的 digest），部署本次完整仓库。
2. 实际 metadata 必须通过两个 stage 的 binding；记录 MobileSAM X5 box 的实际 rank，S 三种 HBM 的 mask H/W 与全部 native dtype。不能改 shape 或强制 reshape 放行。
3. 从仓库根目录运行 evaluator 的同板源实现→统一实现对拍；两个 sample 各跑一次。下面命令在选定板上执行，TARGET 必须改为该板实际值；首次使用新的唯一证据目录。

```bash
# cwd: repository root on the selected board; both artifacts prepared explicitly
TARGET=s100
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_BASE=/tmp/rdk-sam-${TARGET}-${STAMP}
python3 samples/vision/efficient_sam/evaluator/compare.py --target "$TARGET" \
  --output-dir "${OUT_BASE}-efficient" > "${OUT_BASE}-efficient.stdout" 2> "${OUT_BASE}-efficient.stderr"
EFFICIENT_RC=$?
printf '%s\n' "$EFFICIENT_RC" > "${OUT_BASE}-efficient.rc"
python3 samples/vision/mobile_sam/evaluator/compare.py --target "$TARGET" \
  --output-dir "${OUT_BASE}-mobile" > "${OUT_BASE}-mobile.stdout" 2> "${OUT_BASE}-mobile.stderr"
MOBILE_RC=$?
printf '%s\n' "$MOBILE_RC" > "${OUT_BASE}-mobile.rc"
```

保留两个完整目录、stdout/stderr/rc文件。`comparison.json` 保存 UTC、实际板身份、精确 argv/cwd、两侧 metadata、代码/模型/输入校验值、每项判据和返回码；EfficientSAM 14 个、MobileSAM 16 个数组覆盖输入/raw/result。通过要求 rc=0 且全部 checks=true；mask 像素或候选索引不一致均失败，不自动豁免阈值或平局差异。该判据不是数据集精度或延迟测量。自定义 box 还需至少一个边界内框对照；所有变更使用新目录保留记录。

转换另在相应 OE 环境按 conversion README 执行：源权重与版本前提、代表性校准图、真实 encoder embedding、目标配置、量化/编译、再按上述流程验证新制品。主机迁移期间未执行 Torch/ONNX/OE 导出，不能据源码合并认定配方已复现。
