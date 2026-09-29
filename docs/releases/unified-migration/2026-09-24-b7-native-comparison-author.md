# B7 原生源/统一数值比较工具——作者报告（2026-09-24）

**作者侧交付，非独立评审。** 基点 `fbffddd`（协调者已推送并组为板测提交
`4d45f9a`：X5/S100 真实 SDK 构建 rc=0、两板统一 C++ 默认模型/图片 smoke rc=0；
数值一致性未做）。本任务为有界任务：只做 YOLOv5 native C++ 比较工具与指定
helper 边界修复；不改 `platforms/` 固定源、不改共享 `utils/c_utils`、不重做
dtype/metadata serializer/其他 sample。全程未 commit/push/merge/SSH。

## 1. 交付物（samples/vision/yolov5/evaluator/native/）

| 文件 | 作用 |
| --- | --- |
| `ycap_observer.hpp` | 单头、SDK-free 只读观测接口（标量/裸指针入参）。未设 `YOLOV5_CAPTURE_DIR` 时全部 no-op；逐 payload 即时落盘，`finish` 才写 `capture.json`——中途崩溃则无 manifest，比较工具据此拒绝 |
| `instrument.py` | 从 `ac11571`/`380e1a2` 经 `git cat-file` 取固定源并验证 SHA-256（失配拒绝）；逐 anchor 唯一计数插入观测 glue（X5 9 个、S 8 个；任一 anchor 匹配数 ≠1 拒绝）；X5 仅白名单重绑定 `MODEL_PATH`/`TESR_IMG_PATH` 两行（逐条记录 old/new）；生成适配 CMake + `instrumentation-audit.json`（原始 blob SHA、插桩后 SHA、每 anchor 匹配计数、重绑定、hooks/CMShA、工具 argv/UTC） |
| `compare_native.py` | 同板源捕获 vs 统一 dump 比较：两侧二进制 SHA、模型/图片 SHA 一致性、阈值一致性、按物理布局/stride 还原逻辑数组后比较；固定判据 inputs 精确 / raw atol1e-5 / scale·zero 精确 / boxes 1e-4 / scores 1e-5 / class id 精确（含已声明排序归一）；输出目录拒绝覆盖；任何缺料/非零运行码/hash·阈值不一致 → 非零退出且保留证据与全部 `.npy` |

**观测点**（源算法零改动，只读）：X5——提交给 SDK 的紧凑 NV12 字节、逐输出
metadata（dtype/quanti/stride/alignedByteSize/axis，按 SDK 枚举取值）、三个头
各自 INVALIDATE flush 后的 raw 字节（与源自身读取同一一致点）、逐类 NMS 后
模型空间 xyxy 检测；S——init 后逐输出 metadata + 完整 scale/zero 数组、
`bgr_to_nv12_tensor` 写入后的两平面完整分配字节（含 stride）、infer flush 后
逐输出 raw 完整范围、`nms_bboxes` 之后 `scale_letterbox_bboxes_back` 之前的
模型空间检测。检测都在模型空间比较（统一 dump 亦为模型空间），不把一侧结果
喂给另一侧。

## 2. 判据与证据形态

- 逻辑数组按 `(h*W+w)*stride[2] + c*stride[3]` 还原（向量化的字节 gather）；
  padding 字节保留在 `*-physical.bin` 证据中但从不参与字节相等断言。
- 检测排序归一（class_id, -score, x1, y1, x2, y2）在 `comparison.json` 中显式
  声明；数量不一致即失败并保存双侧完整数组。
- `comparison.json` 记录：判据全文、两侧二进制路径+SHA、源 cwd/argv0/板卡
  soc、模型/图片 SHA、UTC、逐 stage 通过/失败与首个差异位置和量级。

## 3. 主机验证（本机：Python 3.14.7 / Apple clang 21）

- yolov5 套件 **50 tests OK**（原 37 + 本轮 13）：插桩生成（双 target、锚点
  唯一、拒绝覆盖既有 workdir）、anchor 漂移 fail-closed（0 次与 2 次匹配均
  拒绝）、错误 pin SHA fail-closed、观测头独立编译 + no-op 与启用两模式实跑、
  **插桩后的固定源在 host stub 上编译通过**（X5 main.cc 与 S main.cpp/
  yolov5.cpp 全量 `-fsyntax-only`；这证明注入 glue 引用的是固定源真实变量，
  但**不代表真实 SDK/OpenCV 构建通过**——板端属协调者）、padded 布局比较
  通过（两侧 padding 字节不同而逻辑相等）、raw 篡改非零失败、缺 capture/
  统一非零返回码/模型 hash 不一致/缺 raw 文件/拒绝覆盖输出目录全部非零。
- SDK-free helper 新边界测试（portable_checks）：`dequant_s32_nhwc` 对
  `zero_point_len>0` 且指针 null 显式拒绝；`check_s32_dequant` 的
  `height*width*channels` 与 extent 中间乘积全链路溢出检查（`1LL<<21` 三维
  与 `1LL<<62` stride 负例）；精确 extent 改为
  `(H*W-1)*stride2 + (channels-1)*stride3 + 4`（不再多算尾像素的通道
  padding）；非标量 scale 要求 `quantizeAxis==3`（NHWC 通道轴），axis 未知/2
  拒绝、标量豁免；发布布局（84×84×255、stride 4/1024/86016、alignedByteSize
  7225344）接受复验。
- bytetrack 11 OK；migration checker rc=0 / 0 violations；`git diff --check`
  干净。

## 4. 指定边界修复清单（第 5 项）

1. `dequant_s32_nhwc`：声明了 zero point 长度但指针为 null → `invalid_argument`，
   绝不解引用。
2. `check_s32_dequant`：count 与 extent 全部走 checked 乘加；负例入库。
3. extent 公式改为 `(channels-1)*stride[3]+element_bytes`（原 `channels*
   stride[3]` 多算了最后元素之后的通道 padding）；发布制品布局仍接受。
4. 非标量 scale/zero 的 `quantizeAxis` 必须等于 3（未知/其他轴拒绝）；真实
   制品 axis 以板端 dump 为准，若板端 axis ≠3 会在 gate 处明确报错而非误读。
5. dump 量化数组：超 `kMaxQuantValues`（2^20）显式抛错、绝不静默截断；
   SCALE 张量缺 scale buffer / 声明了 zero 长度缺 buffer → 抛错，绝不以空
   数组冒充完整。真实头（≤255 值）完整保存。
6. CLI help `--bpu-core` 范围同步 `<-1|0..3>`。

## 5. 边界与未做

- 板端：本轮 **not-run**；按 evaluator README（双语）的 1–4 步由协调者在
  X5/S100 执行（X5 默认 s-v2 + bus.jpg；S 默认 x-672 + kite.jpg；阈值一致）。
  板端若 `quantizeAxis` 或其他 metadata 与假设不符，工具会显式失败，把报错
  带回即可。
- host stub 编译仅证明注入 glue 语法与变量引用正确，不证明真实 SDK 构建。
- 未触碰：`platforms/` 固定源、共享 `utils/c_utils`、Python evaluator
  （`compare.py` 指令在 README 保留）、dtype 别名（7f27c8c 已板验）、catalog
  （develop 9692699 已修）、metadata serializer（另会话）。
- 已知环境怪癖记录：本开发沙箱拒绝编译器打开文件名含 `hooks` 的头文件，
  故观测头命名 `ycap_observer.hpp`（内容无关命名；板端无此限制）。

## 6. 证据

- 命令/返回码/文件 SHA：[evidence](evidence/2026-09-24-b7-native-comparison.json)。
- 进度记录：[2026-09-24 本地续作进度](2026-09-24-b7-local-continuation-progress.md)
  N8 节。

---

## 7. 独立评审整改轮（2026-09-24 追记）

以上第 1–6 节为原始交付记录，**原样保留**。独立评审对 evaluator/native 提出
六项阻断；逐项先复现核实（含以真实板端 manifest 复现 parameters 解析错误），
确认全部适用后修复：

1. **parameters schema**：真实 v2 manifest 的 `parameters` 是 JSON object（以
   `evidence/2026-09-24-b7-native-round2/x5-manifest.json` 实测确认），原 pairs
   解析必坏。compare_native 改为仅接受 dict（pairs 明确拒绝），并以**真实
   manifest fixture** 做回归（ fixture 不可得时显式 skip）。
2. **空表绕过**：强制 x5 恰好 1 输入、s100/s100p/s600 恰好 2 输入、两侧恰好
   3 个形状唯一的输出头并构成完整双射（缺失/重复/多余均失败）；逐记录校验
   dtype ∈ {uint8,int32,float32}、值有限、量化类型/axis/scale/zero 一致；
   raw 比较要求 dtype 相同——浮点转换不再能掩盖类型错误。
3. **证据身份**：观测头在捕获时以内置 SHA-256 记录模型/图片 hash 与逐
   payload 尺寸/hash、完整 argv、cwd、起止 UTC、真实返回码，并 tee
   stdout/stderr；统一侧 hash **必填**（缺省即失败）且须与捕获时一致；比较
   阶段逐一验证两侧 payload 的执行时记录。target 必须与 build_target **完全
   相等**（s100 拒绝 s100p/s600；s600/s100p 为显式选项）。输出保留
   `originals/`（两侧 manifest + 物理 payload）与 `digests.json`。未完成的
   `capture-in-progress.json`/`capture-error.txt` 标记使捕获整体拒绝。
4. **观测头**：全部浮点 17 位有效数字序列化（box/scale 精度可 roundtrip，
   测试断言）；begin 拒绝非空捕获目录（留 capture-error.txt，防旧 capture
   假通过）；成功 finish 才移除 in-progress 标记；payload 打开/短写/空指针/
   重复/未注册张量均显式失败并持久化为 failed 捕获；非法 layout（负维度/
   stride）拒绝；**两侧 dtype 均读 SDK tensorType 枚举**（S/X5 glue 映射
   S32/F32/S8/U8/S16；quanti 不再用于推断 dtype）。
5. **闭包固定**：`S_CLOSURE_SHA256` 固定 yolov5.hpp + 7 个 c_utils 头 + 5 个
   c_utils 源 @ 380e1a2（与当前树字节一致，仍固定）；插桩把验证过的 blob
   **复制进 work dir**，生成的 CMake 只引用副本；audit 记录全部闭包 hash 与
   副本路径。浅克隆缺提交时 fail closed 并给出确切准备命令
   （`git fetch --depth=1 origin <sha>` 等）——工作树文件永不冒充固定源（有
   测试）。
6. **最终坐标**：捕获端新增 `detections_original`（X5 在渲染循环内源自身的
   最终坐标处；S 在 `scale_letterbox_bboxes_back` 之后）；统一 adapter 由
   **自己的**模型空间检测经 SDK-free `map_to_original`（与渲染完全相同的
   算法、不截断）计算，绝不从源结果派生；双侧具备时按同一 boxes 判据另比
   一组，否则报告显式记录「最终用户可见输出等价**未**建立（仅比较了模型
   空间）」。

判据不变（inputs 精确 / raw 1e-5 / scale·zero 精确 / boxes 1e-4 / scores
1e-5 / class id 精确）。主机验证：yolov5 **62 tests OK**（+12）、bytetrack 11
OK、migration checker rc=0/0 violations、`git diff --check` 干净。全部命令与
文件 SHA 见
[整改 evidence](evidence/2026-09-24-b7-native-comparison-remediation.json)。
板端仍 not-run，待 Codex 复审并按 README 0–4 步上板执行。

---

## 8. 第二轮独立复审整改（2026-09-24 追记；更正）

第 7 节的「全部六项已关闭」结论**不成立**，予以更正：独立复审以可执行
反例（[反例证据](evidence/2026-09-24-b7-native-tool-rereview.json) 与探针
脚本）证明七项新阻断。逐项先复现（七例全部复现，含以真实生成顺序触发观测
头 `payload without matching tensor metadata`），确认后修复：

- **A（双侧校验缺失）**：board_soc=S600 + --target s100、删除统一 payload
  digest 且 bytes=-123、翻转统一 quanti 三例原均 rc0。现统一侧 payload
  bytes+SHA 必填并核验、逐头 quanti 类型/axis/scale_len/scale 值双侧相等、
  输入 dtype+shape 双侧一致、板卡身份与 target 冲突即失败（三反例均 rc2，
  各有测试）。
- **B（真实插桩顺序）**：X5 输入 glue 先 payload 后 meta，观测头正确地
  fail（`payload without matching tensor metadata`）、inputs 空。glue 顺序已
  改 meta 在前；新测试**提取实际生成文件中的 ycap 调用序列**并原序重放到
  真实观测头，要求得到有效捕获（1 输入 + 3 输出）。
- **C（float32 语义）**：0.45f 的 17 位输出 0.44999998807907104 与统一
  "0.450000" 被 exact-double 拒绝；scale 17 位 vs 9 位被 list 相等拒绝。阈值
  与 scale 改为 **float32 位值比较**（f32 bits 相等即同值；位值不同即败），
  阈值未放宽；fixtures 全面改用非二进制可表示值（0.45、0.0041891187…），
  标准通过路径也如此。
- **D（失败报告缺失）**：缺 capture.json 时 rc2 但无 comparison.json；
  load_json/check 等异常在 bail 之外。现整个比较在单一 try/except 内，任何
  输入/解码/IO/意外错误都写完整失败报告并 rc2（缺文件与坏 JSON 两测）。
- **E（进程证据）**：新增外部 runner `run_capture.py`——真实 subprocess
  argv（含被 gflags 删除的参数，逐字）、cwd、**分离的** stdout/stderr 文件、
  真实退出码、起止 UTC、板卡身份、binary/model/image 运行前后 hash、插桩
  audit 逐文件校验；compare **必需** run-record 并交叉核验全部字段（含统一
  manifest binary_sha256 == 被背书二进制 hash）。C++ tee（混流复制两文件）
  已**移除**；观测头只保留进程外不可得的张量/阶段数据；`finish(0)` 不再被
  当作进程退出码。无大规模重构。
- **F（严格解码）**：restore_rows 拒绝 pitch<行字节与短 buffer；
  restore_logical 只接受两种已证实布局（均匀行距 strided：stride[1]==width*
  stride[2] 且 stride[0] 覆盖整面；精确尺寸 compact：全零 stride 且长度恰为
  count*itemsize），混合/null-当-compact 显式拒绝；`detections` 与
  `detections_original` 键双侧**必填**（不再缺键默认空表），最终坐标双侧
  必备、缺失或单侧缺失**阻断整体验收**（原「免责声明放行」模式删除）。
- **G（fixture 入库）**：真实 4d45f9a X5 板端 manifest 逐字节入库
  `tests/data/native/x5-board-manifest.json`（fec7e9b5…，PROVENANCE.md 记
  来源）——schema 回归在任意 checkout 必跑，不再依赖旁侧工作区/跳过。

主机验证：yolov5 **70 tests OK**（62→70）、bytetrack 11 OK、migration
checker rc=0 / 0 violations、`git diff --check` 干净。七项反例修后行为见
[evidence](evidence/2026-09-24-b7-native-tool-rereview-remediation.json)
（含 each 反例的新 rc 与错误信息）。README（双语）步骤 2 已改为 runner
流程并写明 float32 位值、最终坐标必备、布局白名单与板卡身份边界。板端仍
not-run；第 7 节中对身份/finite 规则「已满足」的表述以本节为准更正——其
满足范围仅限上列主机测试实际覆盖的内容。

---

## 9. 第三轮复审整改（2026-09-24 追记）

三项新反例（[证据](evidence/2026-09-24-b7-native-round3-remediation.json) 记
录复现输出）先逐一复现，确认后修复：

1. **板身份前缀匹配**：`board_conflict("s100", ["S100P"])` 与
   `["S100Whatever"]` 均返回 None——前缀不是身份。改为**精确别名契约**
   （镜像 `docs/release/platforms.json` 与 `samples/_shared/platforms.py`：
   soc 名 {x5|s100|s100p|s600} 精确匹配；x5 经 socinfo 名
   {x5,x5u,x5h,x5m} 解析），每个读数必须恰好解析为比较 target——S100P
   under s100、未知字符串、全空均失败。**统一侧同样要求真实物理进程证据**：
   `run_capture.py` 新增 `--role unified`（真实 argv/退出码/分离 stdout与
   stderr/UTC/cwd/板卡身份/前后 hash；无插桩 audit），compare 新增必填
   `--unified-run-record` 并交叉核验（role、rc、hash 与 manifest
   binary_sha256 及被背书二进制绑定、板卡身份）——build_target 与 manifest
   内部 return_code 不再单独作为物理证据。
2. **audit 协议**：空 `{}`/`null`/缺字段的 audit 原返回 `passed:true`。现按
   instrument.py 实际生成的协议全量校验：schema、target、pinned_commit 等于
   固定常量、**精确固定源集合**（多/少/重均拒）、每源 blob SHA==pin、每
   anchor 恰匹配一次、插桩文件在场且 hash 一致、**按 target 完整闭包**
   （s100 须 13 项、x5 不得有）、观测头与 CMake hash 对账 work dir。audit
   失败时 run_capture **不执行二进制**——先中止、持久化失败记录
   （return_code=null、error、空 stdout）并 rc2。
3. **通道 stride 边界**：`restore_logical` 原接受 `stride[3]=6`（未对齐）与
   `stride[3]=2`（重叠）。现拒绝 `stride[3] < itemsize`、`stride[3] %
   itemsize != 0`、`stride[2] < channels*stride[3]`（像素重叠），同时正例
   验证真实受支持形态（发布布局 84×84×255/4/1024/86016 与对齐通道 padding
   stride[3]=8）仍正确解码。阈值不变。

主机验证：yolov5 **76 tests OK**（70→+6）、bytetrack 11 OK、migration
checker rc=0/0 violations、`git diff --check` 干净。README（双语）板端步骤
更新为两侧均经 runner（第 2 步 source+audit、第 3 步 unified 角色、第 4 步
两个 run-record）。**声明：固定源 vs 统一的数值对照至今未在任何板卡运行
过；本报告不宣称任何对照通过。** 交 Codex 复审并安排真实板端数值验证。
