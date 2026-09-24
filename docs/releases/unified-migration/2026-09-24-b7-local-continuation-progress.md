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
| N1 | S100 原生编译阻断 + 按两侧真实 SDK 适配（含 padded 布局契约修正、dump v2、SDK-shape 主机回归测试） | 已完成（dc3aac5），待协调者板端复验 |
| N2 | 3dresnet tie 测试跨平台失败：复现、定位与修复 | 已完成（诊断与 fixture 修复随 dc3aac5 入库；机理见下） |
| N3 | 原独立评审逐条作者复核（C++ 六项 + FCOS/LPRNet/MODNet/YOLOWorld + README） | 暂缓（协调者统筹，本轮只做有界 native 任务） |
| N4 | Node 构建 catalog + 补跑 ultralytics_yolo 此前 not-run 测试 | 暂缓（catalog 计数已在 develop 9692699 由协调者修复） |
| N5 | 全量主机回归 + CI 同命令 checker + 证据 | 暂缓 |
| N6 | 本轮作者报告 + 台账追加 | 暂缓 |
| N7 | dc3aac5 板端发现有界整改（句柄名/dtype 映射/重叠布局 gate/标量描述符/调度掩码/dump 碰撞/dump 完整性） | 已完成，待协调者上板复验 |

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

修复（仅测试文件，随 dc3aac5 入库）：tie 用例改为确定性失配——legacy 侧保持
全零，统一侧在 `[100:105]` 加 `1e-7`（小于配方 atol=1e-5，raw_allclose 仍 True；
统一侧 softmax 在 float64 中区分该差值），统一侧 Top-5 在任意平台必为
`[100,101,102,103,104]`；并加前置断言把「两侧 ID 序确已不同」显式化。修正后
16 tests OK（本机复验）。该文件属 B5 批次，此为跨批次测试可移植性修复，未改
产品代码/README/历史结论，提请 Codex 复审。

## N3-N6

暂缓（协调者统筹原始 B7 总任务；本轮只做有界 native 任务）。

## N7 dc3aac5 板端发现有界整改（2026-09-24，仅 YOLOv5 native C++）

输入：协调者实板日志（S100 编译 rc=2：`hbPackedDNNHandle_t` 不存在，真实类型
`hbDNNPackedHandle_t`）、独立 gate probe（重叠像素布局被错误放行；标量 scale 被
转发给按通道索引的 helper）、独立 dump 碰撞 probe（raw/transformed 同名同文件，
后写覆盖原字节）、X5 实板 smoke 暴露的 dump 缺口，及 dtype/catalog/metadata 分工
（7f27c8c / develop 9692699 / 另一会话）。逐项：

1. **句柄名**：s_adapter 与测试内 S100 stub 全部改用 `hbDNNPackedHandle_t`。
   负向验证：把 s_adapter sed 回旧拼写对着修正后 stub 编译，精确复现板端错误
   （`unknown type name 'hbPackedDNNHandle_t'`）——stub 不再假通过。
2. **dtype 映射**：恢复实板证据确认的 `S8/U8/S16/F32/S32`，S 输入 dump 不再
   unknown；stub 枚举同步改为真实名。
3. **重叠布局**：`check_s32_dequant` 的 `stride[2]` 下界由 `width*stride[3]`
   改为 `channels*stride[3]`（一个完整像素），两条 stride 均须元素对齐，
   `stride[1] == width*stride[2]` 保持，最后被寻址字节用溢出检查算术精确计算。
   真实发布布局（84×84×255，stride 4/1024/86016，alignedByteSize 7225344）接受；
   probe 重叠布局（stride[2]=400）拒绝；`1LL<<62` 像素 stride 因溢出拒绝。
4. **标量描述符**：不再调用共享 `dequantizeTensorS32`（scale_len=1 时它会
   `scale_data[c]` 越界）；新增 SDK-free 私有 `dequant_s32_nhwc`（新模块
   `yolov5_s_native`），显式广播语义（标量→全通道；zero_point_len=0→无零点），
   短非标量描述符前置拒绝。gate 继续接受 scale_len=1 仅因该 helper 存在；
   发布制品 scale_len=255 走原逐通道路径，行为不变。
5. **调度**：SDK-free `bpu_core_to_backend` 显式映射索引到位掩码（-1→`1ULL<<7`，
   0..3→`1ULL<<n`），越界拒绝；s_adapter 赋掩码不赋索引；cli_main 拒绝
   `--bpu-core` 超出 -1..3。
6. **dump 碰撞**：张量负载按 `input/`、`raw/`、`transformed/` 分目录独立成文件，
   raw 与 transformed 不可能互相覆盖；portable dump 检查以「同名输出、不同字节」
   实测两文件各自字节与 digest。
7. **dump 完整性**：`input_tensors` 记录该次推理实际提交的输入缓冲（X5 紧凑
   NV12 负载；S 按行汇聚平面负载，未初始化 padding 不导出）；X5 NV12 输入 dtype
   记为 uint8（不再 unknown）；tensor info 增加 `quantize_axis` 与完整
   scale/zero-point 数值（上限 1024 项）；manifest 增加
   `binary_path`/`binary_sha256`（`/proc/self/exe` 优先，argv[0] 兜底）；失败
   路径维持 `return_code=2` 并同样记录二进制身份；README（双语）如实写明 dump
   不代表与固定源数值等价。

共享 `utils/c_utils` 未改动（广播反量化为 yolov5 私有 SDK-free 模块，不影响其他
sample）。主机验证：yolov5 37 tests OK（dc3aac5 为 35）、bytetrack 11 OK、
migration checker rc=0 / 0 violations、`git diff --check` 干净、cli_main 与两个
adapter 对各自修正后 stub 编译干净。详见
[evidence](evidence/2026-09-24-b7-native-bounded-round2.json)。板端复验由协调者
执行；本工作树板端仍 not-run。
