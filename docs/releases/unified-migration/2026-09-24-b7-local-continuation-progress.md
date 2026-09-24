# B7 本地续作进度记录（2026-09-24，作者侧执行中）

本轮（本地 Claude Code + GLM 开发，Codex 负责 GitHub 同步与独立评审）承接
`5fc14a4`（HP 作者整改，经 GitHub 拉回）。开发仅在本 worktree；未 commit/push/merge；
不改写任何历史证据；独立评审结论（B7 changes-required、FCOS 专项、B5/B6 各自结论）
原样保留；板端执行由协调者负责，本工作树一律 **not-run**。

新增输入：[native SDK 独立核对](2026-09-24-b7-native-sdk-review.md) 与
[preflight 证据](evidence/2026-09-24-b7-native-sdk-preflight.json)（协调者在真实
X5/S100 读取）。固定源不变：X5 `ac115717197920355fc390bb04299b20e6436864`，S
`380e1a2bf42041af54be6f34935e50197cfadff9`。

环境：主机解释器 `rdk_model_zoo/.venv/bin/python`（Python 3.14.7，只读）；本机
Node v26.9.0 可用；编译器 Apple clang 21.0.0。

## 阶段进度

| 阶段 | 范围 | 状态 |
| --- | --- | --- |
| N1 | S100 原生编译阻断 + 按两侧真实 SDK 适配（含 padded 布局契约修正、dump v2、SDK-shape 主机回归测试） | 已完成，待协调者板端复验 |
| N2 | 3dresnet tie 测试跨平台失败：复现、定位与修复 | 执行中 |
| N3 | 原独立评审逐条作者复核（C++ 六项 + FCOS/LPRNet/MODNet/YOLOWorld + README） | 待执行 |
| N4 | Node 构建 catalog + 补跑 ultralytics_yolo 此前 not-run 测试 | 待执行 |
| N5 | 全量主机回归 + CI 同命令 checker + 证据 | 待执行 |
| N6 | 本轮作者报告 + 台账追加 | 待执行 |

## N1 S100 原生阻断（2026-09-24）

事实基线（全部来自协调者只读证据，非本地虚构）：X5 8GB 真实构建 rc=0、首次推理
rc=0；S100 构建在 `s_adapter.cpp:10` 失败（`<dnn/hb_dnn.h>` 不存在；真实头位于
`/usr/include/hobot/dnn/hb_dnn.h`）。真实 S100 `hbDNNTensorProperties` 无
`alignedShape`；`hbDNNQuantiType` 只有 NONE/SCALE；`dnn/hb_dnn_ext.h` 不存在；
`stride`/`alignedByteSize` 为 int64；`sysMem` 为单个 `hbUCPSysMem`；
`zeroPointData` 为 `int32_t*`。X5 侧这些字段/枚举存在（真实构建已证）。

整改（不删 gate、不虚构字段）：

1. CMake S 分支 include roots 改为固定 S 交付同款 `/usr/hobot/include`、
   `/usr/include/hobot/dnn`、`/usr/include/hobot`；s_adapter 改含
   `hobot/dnn/hb_dnn.h` + `hobot/hb_ucp.h`，删除不存在的 `hb_dnn_ext.h`。
2. S `project()` 不再读 `alignedShape`（S 无此字段）；S `quanti_code` 删 SHIFT；
   S `tensor_dtype_code` 只映射证据确认的 F32/S32，其余归 unknown 交 gate 拒绝
   （仓库内已证 S 代码不按名引用任何 `HB_DNN_TENSOR_TYPE_*`，S8/U8/S16 拼写无据）。
3. **padded 布局不再一概拒绝**：核对固定源 `dequantizeTensorS32` 的真实寻址
   （`(h*W + w)*stride[2] + c*stride[3]`），`check_s32_dequant` 改为接受真实受
   支持的行内 padding 与通道 padding，仅要求 `stride[1] == width*stride[2]`（该
   helper 隐含的均匀行距假设）与分配覆盖含 padding 的存储范围。X5 侧保持紧凑契约：
   固定 X5 源本身就是紧凑 memcpy/扁平读（main.cc:397/471/543/615），拒绝带精确
   原因并在 dump 中记录 `alignedShape`/`stride`/`alignedByteSize` 供板端跟进。
4. S raw dump 改为保留完整 `alignedByteSize` 范围（此前按 count*4 复制会把
   padded 布局错误地表示成紧凑前缀）；dump schema v2，逐张量记录
   aligned_byte_size/stride/aligned（未上报为 null）。
5. 新增三个主机行为测试：s_adapter 对**S100 形状** stub 头编译通过（修复前代码
   会失败——正是本次板端阻断的主机化重现）、对 X5 形状 stub 头必须编译失败、
   x5_adapter 对 X5 形状 stub 头编译通过。stub 全部来自 2026-09-24 板端证据。

主机验证：yolov5 35 tests OK（原 32）。详见
[evidence](evidence/2026-09-24-b7-native-s100-remediation.json)。

## N2 3dresnet tie 测试失败：复现与定位（2026-09-24）

- **本机复现结果**：macOS/arm64 + Python 3.14.7 + numpy 2.5.3 上该套件 16 tests
  OK（tie 测试通过）；HP 侧（Linux + Python 3.12.14 + numpy 2.5.3）同一测试失败
  （期望的 `No automatic tie exemption` 断言未抛出）。两侧证据均在案
  （B5 本机回归 evidence 第 561 行 vs HP B7 回归 evidence）。
- **机理（已实证）**：tie 用例把两侧 raw 置零。统一侧 top-k 用稳定
  `argsort`（并列取小 id），源侧 `get_topk_predictions` 用默认不稳定
  `argsort`。本机实测 400 个相等值下源侧序为 `[0, 272, 271, 270, 269]`，与稳定
  序 `[0,1,2,3,4]` 不同 → ids 不匹配 → 配方按契约失败 → 断言抛出 → 测试通过。
  在 HP 的 Linux 构建上，不稳定 argsort 对全并列键 evidently 返回与稳定序相同的
  前 5（`[0,1,2,3,4]`）→ ids 相等 → `passed=True` → 无断言 → 测试失败。numpy
  对 'quicksort' 并列序不作任何保证（docstring 明示 implementation varies），
  故该 fixture 依赖了未指定行为，跨平台必然不稳定。
- **判定**：被测配方/README 本身无缺陷——其契约是「ID 不匹配即失败、不允许任何
  tie 豁免」；在 HP 平台上两侧 ID 确实相等，按契约判 passed 是正确行为。缺陷在
  测试 fixture：它没有确定性地制造 ID 失配。

（修复与验证结果将在完成后写入。）

## N3-N6

待执行；完成后逐阶段补记。
