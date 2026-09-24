# B7 整改作者报告（Codex 复审前）

本报告是 [GitHub 协作记录](2026-09-23-github-coordination.md) 第 15 节「B7 整改」的**作者侧**交付，记录本轮实现、验证、证据和未完成项。逐阶段落盘进度见 [整改进度记录](2026-09-23-b7-remediation-progress.md)。

**状态口径（不得被本报告改写）：**

- [B7 独立主机评审](2026-09-23-b7-independent-host-review.md) 的 **changes-required 结论原样保留**；[FCOS 专项独立报告](2026-09-23-b7-fcos-independent-review.md) 同样保留。本报告不声明独立评审通过。
- **Closed = no**；**Board = not-run**（本机无 X5/S 板卡、无目标 SDK、无模型二进制，未下载模型、未编译板端、未接板卡）。
- 全程**未 commit / 未 push / 未 merge**，未改写任何历史证据文件，未进入 B8。
- 作者自检通过不等于独立验收，也不等于交付可用。

基点：`d28d8c9b042e2b92823c8b57f77b80c2a4ac0fb2`（协调记录所述检查点），分支 `codex/b7-remediation-20260923`，独立 worktree。固定源不变：X5 `ac115717197920355fc390bb04299b20e6436864`，S `380e1a2bf42041af54be6f34935e50197cfadff9`。

---

## 1. YOLOv5 C++ 六项阻断

新增三个 **SDK 无关、主机可编译可运行** 的模块，使原「不能靠字符串断言证明」的校验获得真实行为测试；两个 adapter 与 CLI 相应整改。

| # | 阻断（交接文档 2026-09-23 第 59–73 行） | 整改 | 行为验证 |
| --- | --- | --- | --- |
| 1 | X5 `alignedByteSize` 仅查 `>0`，可能不足以读取 `count*sizeof(float)`；`alignedShape`/padded 布局未拒绝；紧凑 NV12 memcpy 未证明存储布局 | 新增 `yolov5_gate.*`：X5 输入要求 `NV12 + [1,3,640,640]`、aligned 与 valid 一致（未上报的 0 视为未上报而非 padding）、`alignedByteSize` 与分配均 ≥ 紧凑 NV12 帧；输出要求 F32/NONE/rank-4/`3*(5+classes)`、stride 唯一 8/16/32、aligned 无 padding、容量 ≥ `h*w*c*4` | `portable_checks x5_input` / `x5_head`：接受/拒绝各 7+9 例，含 undersized 分配、padded aligned、错误 dtype/量化 |
| 2 | S `dequantizeTensorS32` 缺 native dtype、scale 长度、stride/容量 gate；guard 对未分配 sysMem 盲 free | 反量化前新增 `check_s32_dequant`：SCALE 分支要求 native S32、scale 描述符长度覆盖通道、连续 NHWC 字节 stride、分配覆盖全部元素，否则拒绝；S guard 只释放 `virAddr != nullptr` 的张量 | `portable_checks s_dequant`：接受 2 例、拒绝 7 例（非 S32、无 scale、scale 过短、SHIFT、非连续 stride、容量不足、S32+NONE） |
| 3 | 固定 X5 C++ 源默认 `s-v2.0`，launcher 仍沿用 Python `n-v7.0`；X5 按类 NMS 的 top_k=300 与 OpenCV 严格 score 边界未保留 | `launcher.py` 在既无 `--variant` 也无 `--asset-id` 时采用 C++ 源默认 `s-v2.0`；`DecodePolicy` 显式区分平台：X5 = 严格 `>` 边界 + 每类 top_k=300（源 `cv::dnn::NMSBoxes`），S = `>=` 边界 + 无上限（源 `nms_bboxes`） | `test_cpp_launcher`：X5 默认变体与文件名断言；`portable_checks decode_boundary`：score 恰等于阈值时 X5 丢弃、S 保留；`decode_topk`：上限策略恰好保留 300、无上限保留全部 |
| 4 | X5 CLI 收 priority/bpu-core 但未应用；S100/S600 对齐宏影响构建，native binary 需拒绝与构建目标不一致的 target | X5 对非默认 `--priority/--bpu-core` **明确拒绝**（无经验证 HB-DNN 映射，不静默忽略）；S 应用调用方调度并声明与源「强制 priority=0」的差异；CMake 为两端定义 `YOLOV5_TARGET_NAME`，运行期 `check_target_matches_build` 拒绝目标不一致 | `portable_checks target_identity`：匹配接受、不匹配与无身份拒绝；`test_cpp_launcher`：X5 拒绝非默认调度、S 接受 |
| 5 | 只有渲染图，缺机器可比 native dump；dump 与可视化应独立模块 | 新增 `yolov5_dump.*`（含自实现 SHA-256，无 OpenCV/SDK 依赖）：`--dump-dir` 写出 `manifest.json`（schema/target/build_target/asset_id/model 与 image 的 SHA-256/输入输出 metadata/参数/UTC/argv/cwd/rc）与逐张量原始文件；失败也写 rc 与 error | `portable_checks dump` 校验 `sha256("abc")`/`sha256("")` 标准向量；Python 侧再独立 `json.load` 并重算 raw 文件 SHA-256、`struct.unpack` 校验浮点值 |
| 6 | 初版 tests 多为代码 substring 断言，不能作为行为证明 | 重写 `tests/test_cpp_contract.py`，改为编译 `tests/cpp/portable_checks.cpp` 并以真实元数据驱动 gate/解码边界/top_k/dump 的行为；仅保留文件存在性与 CMake 绑定的结构性断言 | 上述全部 `portable_checks` 子项 + Python 独立复核 |

声明与限制：

- **本机未编译两个 adapter TU**：主机无 OpenCV 开发头文件、无目标 SDK（`hb_dnn.h`/`hb_ucp.h`），因此 `x5_adapter.cpp`/`s_adapter.cpp` 的编译与 ABI、缓存、stride、模型数值仍为 **not-run**，留板端队列。
- `alignedShape`、`scale.scaleLen`/`zeroPointLen` 等字段名取自 SDK 公开结构并经源 `c_utils` 用法核对，板端编译需确认。
- 带 padding 的 aligned 布局当前是**明确拒绝**（源代码本身也按紧凑布局寻址）；若板端实测确有 padding，正确做法是按 aligned stride 寻址，列为后续项而非放宽 gate。
- 统一解码器丢弃非有限 score（源 S 会保留），属已声明的修复。
- 置信度乘积用 float32 计算，X5 源用 double；边界 1 ulp 内可能不同，已声明。

## 2. FCOS / LPRNet / MODNet / YOLOWorld

### FCOS（独立专项报告四项）

| 独立发现 | 状态 |
| --- | --- |
| P1 manifest 指向缺失 `download_model.sh` | 已存在；三个下载脚本同源 wrapper，测试断言不含 `.venv/bin/python` |
| P1 板端路径硬编码 `.venv/bin/python` | 三个脚本改用 `python3` |
| P1 `--resize-type 1` letterbox 逆几何错误 | 已按冻结 `ImageContext` 的 `resized_shape`/`pad` 反算并 clip；`_validate_context` 拒绝几何不一致；有非方图 A/B 测试 |
| P1 evaluator 只是手工配方 | 已实现自包含 `evaluator/compare.py`（gates x5、自行运行两侧、全 15 路 raw/metadata/argv/cwd/rc/失败记录、单一时戳变量） |
| P2 F32 输出仍走反量化 | 保留源语义并**补齐另一半分支测试** `test_non_scale_outputs_pass_through_in_both_helpers`（非 SCALE 描述符在统一与源 helper 中均为恒等） |

### LPRNet / MODNet（B7-LM1..3）

- **LM1**：两个 runner 在构造真实 SDK factory 前完成 `require_execution_target` + `verify_asset_file`，拒绝时不会调用 factory；`bind_model` 重新解析 manifest 并比对完整发布事实（伪造同形 publication row 被拒）；输入/输出 exact shape/dtype/finite 锁定。原实现的无条件 gate 会让约定的 host injection seam 不可用，已与 FCOS 参考实现对齐为「仅真实路径 gate，显式注入 factory 才跳过」，并加测试证明真实路径必过 gate、注入 seam 跳过。
- **LM2**：两个 `main.py` 支持任意 cwd 直接 `python /abs/main.py --help`；MODNet `--dry-run --ref-size 256` 返回 rc=2。**新发现并修复**：dry-run 原先不校验调度参数（`--priority 300/-1`、`--bpu-cores 0 -1` 返回 rc=0），现与真实路径同样拒绝。
- **LM3**：两个 evaluator 由「手工文件比较」重写为自包含 `compare.py`：gates x5、同一模型/输入/参数下自行运行固定源与统一实现、Recorder 捕获输入与 raw 输出、记录 metadata/argv/cwd/UTC/return_code/失败 JSON、绑定模型/输入/代码 SHA-256、逐数组落 `.npy` 并各自记录摘要。新增 `evaluator/source_reference.py` 固定源加载 seam。双语 evaluator README 同步重写。

### YOLOWorld（B7-YW1..2）

- **YW1**：evaluator 重写为自包含 `compare.py` —— 自行运行源与统一两侧，捕获预处理输入、raw score/box 与最终检测，记录模型/图像/词向量/代码 SHA-256、两边 runtime metadata、prompts/阈值/argv/cwd/UTC/rc，失败也写 `error`+`return_code: 2`；新增 `source_reference.py` 显式绑定固定平台 postprocess 并安装含 `QuantParams` 的注入 stub，去掉对调用方环境 `utils` 导入的依赖。
- **YW2**：runner 补 `verify_asset_file` 与输出有限值校验；`bind_model` 重新解析 manifest 并比对发布事实；**词表改为只读快照**（`np.array(copy=True)` + `writeable=False`），调用方事后改写自己的数组不再影响后续调用。

## 3. B7 双语 README 复核

- **14 项 checker 违规清零**：`runtime/cpp/README_{,cn}.md` 按 `docs/sample-standards/templates/runtime-cpp.{en,zh}.md` 重写，补齐 7 个 anchor，并写明默认变体、NMS 边界、letterbox、调度、dump 差异。**未新增豁免、未降级规则**，84 条 B9 豁免保持不变。
- **修正字面双反斜杠 shell 续行**（交接文档已标记）：4 个 B7 README 共 19 处 `\\` → `\`，修后 `bash -n` 通过。
- **修正 YOLOv5 evaluator 的运行位置表述**：由「在控制板卡的主机运行」改为「直接在目标板上运行，不是外部主机驱动板卡」（中英同步）。
- **机械交叉核对**：6 个 B7 sample 的 24 份 README 中所有 `` `--flag` `` 均存在于对应 `main.py`；LPRNet/MODNet/YOLOWorld 文档 shape 与 binding 常量一致；所有 B7 目录 README/README_cn anchor 集合完全一致。

## 4. 验证（主机与板端分开）

**CI 同命令**（`.github/workflows/sample-contract.yml`）：

| 步骤 | 命令 | 返回码 |
| --- | --- | --- |
| checker 单测 | `.venv/bin/python -m unittest discover -s tools/sample_contract/tests -v` | 0（27 OK） |
| migration checker | `.venv/bin/python tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json --report /tmp/sample-contract-report.json` | **0**，`36 samples, 0 violations, 39 skips, 84 exemptions applied`（整改前 rc=1 / 14 violations） |
| 空白检查 | `git diff --check` | 0 |

CI 差异仅两处：CI 用 python3 3.11（本机 3.12.14），report 路径为 `/tmp`（避免污染工作树）。

**全量主机回归**：906 tests，39 个套件。B7 与前一阶段相关套件：

| 套件 | 结果 |
| --- | --- |
| samples/vision/yolov5 | 32 OK（整改前 24） |
| samples/vision/bytetrack | 11 OK |
| samples/vision/fcos | 24 OK（+1 非 SCALE 分支） |
| samples/vision/lprnet | 12 OK（整改前 7） |
| samples/vision/modnet | 12 OK（整改前 7） |
| samples/vision/yoloworld | 18 OK（整改前 10） |
| samples/_shared | 101 OK |
| tools/sample_contract | 27 OK |

B4/B5/B6 既有套件（resnet 52、mobilenetv4 20、paddle_ocr 44、efficient_sam 19、mobile_sam 17 等）全部 OK。

**已知非本轮问题（原样报告，未修）：**

- `samples/vision/3dresnet/tests` 1 项失败：`test_board_recipe_keeps_evidence_and_never_auto_accepts_tied_ids` 期望抛出 `No automatic tie exemption` 断言而未抛出。相关文件在本 worktree **未被修改**，即基点行为，属 B5 范围，未在本轮处理（避免与 B5 既有独立评审结论冲突）。
- `samples/vision/ultralytics_yolo/tests` 2 项 **not-run**：需要 `tools/catalog-publisher/dist/catalog.json`（`npm --prefix tools/catalog-publisher run build`）。本机无 Node/npm，按 `AGENTS.md` 记为 not-run，不算通过也不算失败。
- `tools/catalog-publisher/tests` 定义 0 个用例（`Ran 0 tests` 非失败）。

**环境与哈希**：主机 `.venv` 为 Python 3.12.14，numpy 2.5.3 / opencv 5.0.0 / scipy 1.18.1 / pyyaml 6.0.3 / lap 0.5.12 / cython-bbox 0.1.5（另装 ftfy、regex 以跑通 CLIP 既有套件）。仅访问包源，**未下载任何模型、未加载 SDK、未接触板卡**。完整命令/返回码/版本/文件哈希见 [recon evidence](evidence/2026-09-23-b7-remediation-recon.json)、[regression evidence](evidence/2026-09-23-b7-remediation-regression.json) 与 [regression log](evidence/2026-09-23-b7-remediation-regression.log)。

**板端：全部 not-run。**

## 5. 未完成项 / 建议复审关注

1. 两个 C++ adapter TU 未在本机编译；板端需确认 `alignedShape` 与 `scale.*Len` 字段名、真实 stride/容量与对齐。
2. 带 padding 的 aligned 布局目前拒绝；若板端实测量产制品确有 padding，需改为按 stride 寻址。
3. X5 非默认 `--priority/--bpu-core` 当前拒绝；若 SDK 确认 `hbDNNInferCtrlParam` 支持对应字段，应改为应用。
4. 所有 evaluator 的**真实板端 source/unified 对照仍未执行**，`comparison.json` 的实际数值对照是待办。
5. 根目录 `evidence/2026-09-23-b7-yoloworld-host.json` 路径与批次目录不一致（历史遗留）。移动它会破坏 `evidence/2026-09-23-handoff-working-tree.json` 中的冻结路径记录，故**本轮未移动**，留待后续统一处理。
6. B8 未开始；`evidence/2026-09-23-b8-source-inventory.json` 仅为只读清点。
