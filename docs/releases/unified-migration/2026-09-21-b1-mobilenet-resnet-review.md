# B1 批次评审 — mobilenetv1–v4 迁移 + resnet50/152 变体 + H5/H6（2026-09-21）

> **独立评审更新（Codex，2026-09-21，审阅 c218e86）：changes-required / not-ready。**
> 见 [B1 独立评审](2026-09-21-b1-independent-review.md)：B1-R1–R6 尚待修正。
> 下文 §5–§7 的 passed Review 为作者自评历史记录，不代表独立评审通过；
> “仅剩 x5-8g”的历史结论由本更新取代。板测事实保留，X5 8GB 由原执行者补测。
> B1 尚不可关闭；独立审阅必须逐批执行，不能推迟到最终收尾。

## 0. 预检（固定源与清点）

| 项 | 值 |
| --- | --- |
| rdk_x5 源 SHA | `ac115717197920355fc390bb04299b20e6436864`（tip = ac11571，x5-v1.1.3） |
| rdk_s 源 SHA | `380e1a2bf42041af54be6f34935e50197cfadff9`（tip = 380e1a2，s-v1.1.2） |
| mobilenetv1 文件数 x5/s | 17 / 16 |
| mobilenetv2 文件数 x5/s | 18 / 23（s 含 runtime/cpp，x5 python-only） |
| mobilenetv3 文件数 x5/s | 18 / 20 |
| mobilenetv4 文件数 x5/s | 19 / 22 |
| s resnet50 / resnet152 | 18 / 21（x5 无，resnet50/152 为 S 独有） |
| 提取方式 | `git archive <branch> <path>` 展开 + 手工重构（不整目录恢复已迁移 sample） |

两侧 `runtime/python` 均为标准化 wrapper（`MobileNetVNConfig` + `MobileNetVN`
五段接口 + `main.py`），结构同 resnet 试点审计过的源；差异在契约事实层。

## 1. 契约事实表（从源代码逐行核实，映射表的规范侧输入）

| sample / variant | target | resize 默认 | resize 插值 | letterbox 插值 | 输出策略 | 输出语义（源声明） | 几何 | class_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mobilenetv1 | x5 | 0（stretch） | linear | linear | none（不 softmax） | probabilities | 224×224 | 1000 |
| mobilenetv1 | s | 1（letterbox） | nearest | linear | none | probabilities | 224×224 | 1000 |
| mobilenetv2 | x5 | 1 | linear | linear | none | probabilities | 224×224 | 1000 |
| mobilenetv2 | s | 1 | nearest | linear | none | probabilities | 224×224 | 1000 |
| mobilenetv3 | x5 | 0 | linear | linear | softmax | logits | 224×224 | 1000 |
| mobilenetv3 | s | 1 | nearest | linear | softmax | logits | 224×224 | 1000 |
| mobilenetv4/small | x5 | 0 | linear | linear | softmax | logits | 224×224 | 1000 |
| mobilenetv4/small | s | 1 | nearest | linear | softmax | logits | 224×224 | 1000 |
| mobilenetv4/medium | x5 | 0 | linear | linear | softmax | logits | 224×224 | 1000 |
| mobilenetv4/medium | s | 1 | nearest | linear | softmax | logits | **256×256** | 1000 |
| resnet18（既有） | x5/s | 1 | linear/nearest | linear | legacy_softmax | unverified_score_vector | 224×224 | 1000 |
| resnet50（新，s-only） | s | 1 | nearest | linear | softmax | logits | 224×224 | 1000 |
| resnet152（新，s-only） | s | 1 | nearest | linear | softmax | logits | 224×224 | 1000 |

依据：x5 wrapper 显式 `cv2.INTER_LINEAR`；s wrapper 不传插值 → 其
`utils/py_utils/preprocess.py:resized_image` 默认 `INTER_NEAREST`（两侧该函数
实现一致，letterbox 填充均为 `(127,127,127)` BGR）。x5 v3/v4 `post_process`
用 `scipy.softmax`；s v3/v4 与 s resnet50/152 走
`visualize.get_topk_predictions`（数值稳定 softmax + argsort(-p)），数学一致。
x5 v1/v2 与 s v1/v2 源码不 softmax（docstring 声明输出已是概率），分数按原值
打印——canonical 以策略 `none` 逐位保留该行为。**v4-medium S 制品为 256×256**
（x5 medium 为 224）：几何由契约表逐 (variant,target) 给出，运行时再按元数据
校验，不得全局写死 224。

## 2. 旧→新函数映射表

### 2.1 x5 侧（mobilenetv1–v4 共用形态，N∈{1,2,3,4}）

| 源（`runtime/python/mobilenetvN.py` + `main.py` @ac11571） | 统一架构去向 |
| --- | --- |
| `MobileNetVNConfig(model_path,label_file,resize_type,topk)` | CLI `--asset-id/--variant/--resize-type/--top-k` + `resolve_selection`（路径优先 → 引用优先） |
| `__init__`: `HB_HBMRuntime(path)`、`model_names[0]`、H/W 取 shape `[2]/[3]` | `RuntimeModelRunner.load()`：懒加载 + `RuntimeMetadata.from_runtime`（显式选模型）+ `bind_model` 契约校验（几何来自契约表，不重算） |
| `set_scheduling_params(priority,bpu_cores)`（`{model:…}` 包装） | `RuntimeModelRunner.set_scheduling_params`（同一 API 形状） |
| `pre_process`: `resized_image(W,H,type,INTER_LINEAR)` + `bgr_to_nv12_planes` + 拼接 reshape `(1,3H/2,W,1)` | `prepare_nv12`：`resize_bgr`（linear）+ 共享 `bgr_to_nv12_planes` + `as_packed` **flat 1-D**（H2 canonical；字节同旧 4D 视图） |
| `forward`: `run(inputs)[model_name]` | runner `__call__`（包 `{model_name:…}`、扁平输出、容器校验） |
| `post_process`: squeeze→argsort（v1/v2）；softmax→argsort（v3/v4） | `ClassificationTask.post_process`：声明式 `output_transform`（raw_f32）→ 契约 score policy（none/softmax）→ `topk_from_scores` |
| `predict`/`__call__`（计时打印） | `ClassificationTask.predict/__call__`（计时归 CLI 层） |
| `main.py` 参数 `--model-path/--label-file/--priority/--bpu-cores/--test-img/--img-save-path/--resize-type/--topk` | 统一 CLI：`--target/--asset-id/--variant/--model-path/--test-img/--label-file/--top-k/--topk(旧拼写)/--resize-type/--priority/--bpu-cores/--img-save-path/--list-models/--dry-run`（对齐 resnet 试点） |
| 标签 `datasets/imagenet/imagenet_classes.names`（file_io.load_imagenet_labels） | 同文件，经仓库根 `datasets/`（A2 统一位置）读取 |
| `model/download.sh`（x5 命名） | `model/download.py` + `download.sh --target`（manifest 驱动 + sha256 校验） |
| `utils/py_utils` 引用 | sample 内不再引用；等价实现已在 `_shared`（image/preprocess 语义） |

### 2.2 s 侧（mobilenetv1–v4 + resnet50/152 共用形态）

| 源（`runtime/python/*.py` @380e1a2） | 统一架构去向 |
| --- | --- |
| `get_soc_name()` + `s600 else s100` **静默回退** | `samples/_shared/platforms.py` 显式解析；未知板卡报错；s100p 无清单行 → 明确 "no published asset"（不再回退 s100） |
| 默认模型 `/opt/hobot/model/{soc}/basic/*.hbm` | manifest 资产引用 `s:<id>:<file>` + `model/{s100,s600}/` 本地布局；系统路径在 README 记为手动准备选项 |
| `pre_process` 产出 split `{input0:y, input1:uv}`，H/W 取 shape `[1]/[2]` | `prepare_nv12` split 分支（`as_split` 平面）+ bind 按 Y/UV 形状识别角色 |
| `forward` 返回整个 `run()` dict；post 内 `outputs[model][out]` 延迟索引 | runner 统一扁平化输出；task 按 bound output name 提取 |
| `post_process`：argsort 原值（v1/v2）/ `visualize.get_topk_predictions`（v3/v4、resnet50/152） | 契约 score policy `none`/`softmax`（数学同源：稳定 softmax + argsort(-p)） |
| `main.py` 参数（v*：`--model-path/--priority/--bpu-cores/--test-img/--label-file`；resnet50/152 加 `--top-k`；resize 无 CLI，wrapper 默认 1） | 同 2.1 统一 CLI 集 |
| 标签 `test_data/imagenet1000_labels.txt`（v2）/`imagenet_classes.names`（v1/v3/v4）/ 根 `datasets/`（resnet50/152） | 统一根 `datasets/imagenet/imagenet_classes.names`；sample 本地标签文件保留在 test_data 供对照 |
| `model/download_model.sh`（soc 探测 + s100 回退） | `download.sh --target`（显式 target，无回退） |
| `runtime/cpp`（仅 mobilenetv2） | 原样保留于 `samples/vision/mobilenetv2/runtime/cpp/`，README 声明 S-only |

### 2.3 资产/清单映射（文件名保持发布事实不变）

| sample | x5 清单 id → 资产 | s 清单 id → 资产 |
| --- | --- | --- |
| mobilenetv1 | `mobilenetv1` → `mobilenetv1_224x224_nv12.bin` | `mobilenetv1` → `s100|s600/mobilenetv1_224x224_nv12.hbm` |
| mobilenetv2 | 同上形态 | 同上形态 |
| mobilenetv3 | → `MobileNetV3_224x224_nv12.bin`（注意大小写） | → `s100|s600/mobilenetv3_224x224_nv12.hbm` |
| mobilenetv4 | → `MobileNetV4_conv_{small,medium}_224x224_nv12.bin` | → `s100|s600/mobilenetv4_{small_224x224,medium_256x256}_nv12.hbm` |
| resnet（扩展） | 既有 `resnet`（resnet18）不动 | `resnet50`/`resnet152` 行 `sample_path` 改指 `samples/vision/resnet`、`download_scripts` 改统一 `download.sh`，加 notes（--variant 选择），**保留旧 id 与资产事实** |

variant 判定用每 sample 显式 `{filename: variant}` 表，不做文件名猜测。

### 2.4 转换能力映射（B1-R1 整改补记）

| 源（rdk_s @380e1a2） | 统一架构去向 | 事实 |
| --- | --- | --- |
| `samples/vision/resnet152/conversion/get_calibration_data.py`、`resnet152_config.yaml`、`x86_inference.py` | `samples/vision/resnet/conversion/` 同名文件 | 逐字节迁入（SHA-256 由 `tests/test_conversion_layout.py` 固定：d8a39491…、20eaf2cb…、f2c5738e…）；YAML 前缀与 Manifest 文件名一致性入测试 |
| `resnet152` 转换 README（含校准/编译/记录表） | `conversion/README(_cn).md` 按变体重写 | 152 全配方、50 指针式（OE `13_resnet50`，无源配方不虚构）、18 导出-only 的可复现范围逐变体声明 |
| 无对应源（resnet50 无配方） | 不新建 | 以 known-gap 记录，不编造 YAML/脚本 |

## 3. 本批架构决策

### 3.1 分类运行时提升（§5.1 第二消费者规则在批内成立）

B1 带来 5 个分类 sample（resnet 扩展 + mobilenetv1–4），B2–B4 还有 ~12 个。
逐 sample 复制 5 个运行时模块违背提升纪律；本批把 resnet 试点的
`tensor_io.py`/`model_runner.py`/`classification.py` 及 binding 机械部分提升为
`samples/_shared/` 共享模块（纯代码搬移 + 参数化，行为不变），resnet 原模块变为
再导出门面（公共面稳定，试点测试不改仍绿）。每个 sample 保留
`runtime/python/model_binding.py`（**契约表 = 平台差异的唯一归属**）与
`main.py`/`tests/`。score policy 集合扩为 `legacy_softmax | softmax | none`
（`softmax` 与 `legacy_softmax` 数学相同、认知状态不同：后者保留 resnet18 的
"未核实"警示；v3/v4/resnet50/152 源 docstring 明示 logits，用 `softmax`）。

### 3.2 H5 — 平台 Profile 契约（`samples/_shared/platform_profile.py`）

以 `ultralytics_yolo/runtime/python/yolo_platform.py` 的字段形状为参照升为通用
契约，但**只收平台不变事实**：`key/family/soc_names/march/model_suffix/
model_format/model_subdir/input_protocol(packed|split)/url_prefix/
cls_interpolation/supports_cpp` + `model_base_url()`。源码核实表明 resize 默认、
NMS 阈值、几何是 **per-sample/per-variant** 事实（x5 v1=0 vs v2=1 vs S 全 1），
留在各 sample 契约表，Profile 不越权。分类 sample 用共享工厂
`classification_profiles(url_prefix_s=…)` 生成 x5/s100/s100p/s600 四 Profile
（s100p 无已发布制品：Profile 存在使显式选择得到诚实的 "no published asset"
错误；其 `url_prefix=None` 且 `model_base_url` 拒绝构造）。yolo 本地模块**不动**
（B9 收编时再 lift）。

### 3.3 H6 — sample↔manifest 覆盖检查（`samples/_shared/tests/test_manifest_coverage.py`）

规则：`samples/*/**` 中含 `runtime/python/model_binding.py`（统一架构标记）的
每个目录，必须在 `docs/release/{x5,s}/models.yaml` 至少一个组有
`sample_path` 精确匹配的行；每行 `download_scripts` 指到的脚本必须存在。
仓库根不可定位时跳过（不误报）。清单 URL 与 Profile 布局的交叉核对记录为
后续可选扩展，本批不做。

### 3.4 遗留路径修正（随批）

resnet 试点 `--label-file` 默认指向 `platforms/x5/datasets/...`（快照路径，
收尾删除 platforms/ 时断链）；A2 后统一位置是根 `datasets/imagenet/…`。本批把
resnet（及全部新 sample）默认标签路径统一切到根 `datasets/`，快照路径在
README 不再出现。

## 4. 批内顺序

1. `_shared`：platform_profile（H5）+ 分类运行时提升 + policy 扩展；
2. resnet：变体扩展（resnet50/152）+ 标签路径修正 + 门面化；
3. mobilenetv1–v4：契约表 + main + tests + model/download + conversion/evaluator/test_data 合并 + 双语 README；
4. H6 覆盖测试；
5. 清单更新（s mobilenet 行、s resnet50/152 行）；
6. 全量主机验证 + Q3 检查器 + 台账 + 证据 + commit；
7. 板端冒烟：用户在 X5 8GB/4GB + S100 执行（H2 flat≡4D 等价性确认一并搭载）。

## 5. 进度区行（详表见 x5-s-migration-map.md 本轮进度区）

B1 十一个交付物（mobilenetv1–v4 ×{py}、mobilenetv2 cpp、resnet50/152 变体、
H5、H6）按实际执行回填：

| 交付物 | Mapping | Refactor | Docs | Host | Board | Review | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mobilenetv1（x5+s python） | done | done | done | passed（17 OK，板后修复回归） | passed（x5-8g+x5-4g+s100+s600 对照全等；s100p 负例 2/2） | passed（§6 自评 + §7 板后复核） | [b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json) |
| mobilenetv2（x5+s python + s cpp） | done | done | done | passed（24 OK；+7 为 B1-R2 整改的 cpp 启动器身份 fixture） | passed（python 四板对照全等（含 x5-8g 复测）；cpp s100 构建并运行 TOP-1 zebra，BUILD_JOBS=1） | passed（同上） | 同上 |
| mobilenetv3（x5+s python） | done | done | done | passed（17 OK） | passed（四板对照全等，含 x5-8g 复测） | passed（同上） | 同上 |
| mobilenetv4（x5+s python，small/medium） | done | done | done | passed（20 OK；+3 为 B1-R4 整改的转换 shape 一致性测试） | passed（四板（含 x5-8g 复测）；medium 下载缺陷经板测发现并修复复验，见 §7 B1-D2） | passed（同上） | 同上 |
| resnet50/152 变体（s-only python） | done | done | done | passed（套件 52 OK；+6 为 B1-R1 整改的转换 layout/provenance 测试） | passed（s100+s600 对照全等） | passed（同上） | 同上 |
| H5 Profile 契约 | done | done | done（模块 docstring + 测试表） | passed（11 OK） | not-applicable（纯主机契约） | passed | 同上 |
| H6 覆盖检查 | done | done | done（本文件 §3.3） | passed（4 OK） | not-applicable | passed | 同上 |

Closed=no：板测维度已随 2026-09-21 下午的 x5-8g 复测闭环（§7 末行：5/5 通过，
含 H2 flat≡4D 确认 v1/v2 逐位相等、v3/v4 ≤6e-8）；但独立评审（Codex，
[B1 独立评审](2026-09-21-b1-independent-review.md)）判定 changes-required
（B1-R1–R6），整改完成并由独立 reviewer 复核前 B1 不得关闭。原执行者整改记录见
[B1 整改记录](2026-09-21-b1-remediation.md)（2026-09-21；待独立 reviewer 复核），
台账拆行见 [x5-s-migration-map.md](x5-s-migration-map.md) 本轮进度区。上方本表 Board
列的 passed 仅覆盖板测维度；Review 列的 passed 是作者自评历史记录，已被独立
评审的 changes-required 取代（见文首更新说明）。

## 6. 批后自评（2026-09-21，B1 完成时）

**结论：B1 主机侧交付完成，全量验证绿；板端冒烟五个板位（x5-8g/x5-4g/s100/
s600-64g/s100p 负例）全部通过并回填（§7）。本自评不构成批次关闭：独立评审
changes-required（B1-R1–R6）整改中，Closed=no。**

### 6.1 做对了什么

- **第二消费者规则兑现**：分类运行时五模块升入 `_shared/`（labels/cls_binding/
  tensor_io/model_runner/classification），resnet 变再导出门面；试点 46 测试
  不改仍绿，4 个新 sample 各 14 测试首次生成即绿——提升没有引入行为回归。
- **策略语义诚实**：v1/v2 `none`（源声明输出已是概率，逐位保留）、v3/v4/
  resnet50/152 `softmax`（源 docstring 明示 logits）；`legacy_softmax` 仅保留
  resnet18 的"未核实"认知状态，数学与 `softmax` 一致（新测试
  test_policy_legacy_softmax_matches_softmax_numerically 固定该等式）。
- **H5 克制**：Profile 只收平台不变事实（march/format/subdir/url/caps），
  resize/NMS/几何留在 per-sample 契约表——源码核实表明这些是 per-variant
  事实（x5 v1=0 vs v2=1），收进 Profile 会制造假统一。
- **H6 可空转安全**：marker glob 有最小基数断言，防 glob 失效后两个覆盖断言
  空真；清单缺失时 skipTest 不误报。
- **静默回退清零**：v2 cpp 启动器 s100p/未知板显式报错；python 侧沿
  platforms.py 无回退原则。
- **缺口记录而非美化**：v4 X5-medium 导出 256 vs 发布 224、v1/v2 无导出器、
  S 侧无公开性能数据、SHA-256 未知——全部以 known-gap/not-run 形式落档。

### 6.2 本批自查发现并已修复的问题

1. 生成的单变体 download.py 用字符串而非元组作 key——15 个引用全数解析失败；
   修于生成器并重生成、逐一验证。
2. resnet 文档漂移（50/152 不在支持矩阵、旧清单路径、旧标签路径、"仅
   resnet18 发布"）：10 个 README + cpp run.sh 修正，Q3 复检 0 违规。
3. README 生成器首轮 30–42 锚点违规/样本（conversion/evaluator/cpp 层 section
   id 与 Q1 模板不符）：按模板重写三层后 0 违规。
4. 我对 mobilenetv2_config.yaml 的首次判断（X5 bayes-e）错误——grep 证实
   march nash-e（S 侧）；已按证据改正。
5. test_classification 合成绑定首版构造签名错误（facts=/model_path= 不存在
   的字段）；按数据类真实字段重写后 8/8 通过。

板端冒烟追加（详见 §7 与 board evidence）：

6. **B1-D1（板测发现）**：X5 mobilenet 制品输出为 F32 却仍带编译器量化
   描述符，`raw_f32` 契约在 `cls_binding.bind_model` 与
   `quantization.apply_output_transform` 两处以"存在描述符即拒绝"设卡，
   全部 5 个 x5 条目 bind 即败——旧实现直接吃 F32 值、忽略描述符。修复为
   **dtype 才是闸门**（float32 放行并把描述符快照进 binding 供记录、绝不
   应用；非 float32 仍拒绝），两处同步放宽，resnet 试点测试与新描述符
   回归测试按新契约改写。主机测试漏掉它是因为 fixtures 从不携带
   output_quants——这正是"host 绿 ≠ 板上能跑"的实证。
7. **B1-D2（板测发现）**：v4 `download_target` 解析引用时丢掉 variant，
   `--variant medium` 实际下载 small。根因是生成器多变体分支从未设置
   `variant_call`；主机侧此前只验证了 `asset_reference` 本身，从未走
   `download_target` 全链路。修复生成器 + 重生成（四样本全部字节一致校验），
   新增 `tests/test_download.py`（stub 掉下载器，断言每个组合转发的精确
   引用；v4 全 6 组合、v1-v3 各 3 target）。
8. 检查器自检顺序缺陷（自查发现）：B1 主机验证时先跑 checker 自检、后改
   进度区行，导致带描述性 sample 名/基础设施行的台账使 R-SCOPE 解析失败
   而未被当时发现。修复：台账约定 `~` 前缀基础设施行（legend 记载）、
   sample 行改写为可解析名，检查器支持该约定并有正反例测试。

### 6.3 遗留与风险

- **独立评审整改（当前阻断项）**：Codex 独立评审 changes-required
  （[B1 独立评审](2026-09-21-b1-independent-review.md)）：B1-R1 resnet50/152
  转换能力未迁入、B1-R2 v2 cpp 启动器 S100P 身份形式、B1-R3 v3/v4 校准文档、
  B1-R4 v4 medium S 验收 shape、B1-R5 板后证据未同步客户 README/台账拆行、
  B1-R6 CI 欠账基线。逐项整改后由独立 reviewer 复核，才可 Closed=yes。
- **s100 cpp 并行编译 OOM**：`cmake --build --parallel $(nproc)` 在 s100 上
  cc1plus 被杀；`BUILD_JOBS=1` 构建通过。启动器已支持 `BUILD_JOBS` 覆盖，
  README 补充建议（见 §7 备注）。
- **v4 medium 双几何**（x5 224 / s 256）已按目标分别记录并在两平台板测通过；
  若未来发现源分支发布过 x5 256 medium，需回改契约表而非运行时猜测。
- **平台身份检测路径**（boardinfo/socinfo/device-tree 顺序）沿用 `_shared/
  platforms.py` 现状，本批未动；B5+ 非视觉输入 sample 可能暴露其假设。
- 兼容 shim（platforms/{x5,s} 下旧入口）保留至收尾删除；evaluator README 的
  同板前后对照指引引用它们，属有意为之。
- `--scope migration` 全量模式报告 ultralytics_yolo 84 条 R-README-SECTIONS
  违规——试点文档债，台账 Docs=pending、按计划 B9 收编时按 Q1 门槛重写；
  非本批引入，不因检查器存在而视为已验收。
- 独立审阅不再推迟到收尾：B1 已由独立 reviewer（Codex）完成 change-review 并
  交付 changes-required 结论；本文件 §6/§7 是作者自评，与独立评审分开解读。

### 6.4 下一步

整改 B1-R1–R6（见 §6.3 首条与独立评审文件），完成后交独立 reviewer 复核
关闭 findings；通过后 B1 Closed=yes 并进入 B2
（efficientnet/efficientformer(v2)/efficientvit）。

## 7. 板端冒烟结果（2026-09-21，修复后）

方法：仓库子集打包（SHA-256 `2134411c…695c`；x5-4g 用此前
`8b81ffa4…fc94`，samples/ 内容字节一致）scp 至各板 `/tmp/rdk-b1-smoke`，
板上 runner 执行：manifest 驱动下载 → canonical 推理 → 板上状态隔离检查
（交错 31×47 零图后重推原像须逐位一致）→ importlib 载入 platforms/ 旧实现
对照（ids 全等 + scores allclose(1e-5,1e-6) + labels 相等）。逐条 JSON 证据
与制品/输入 SHA-256 见 [board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)。

| 板位 | 身份 | 条目 | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| x5-4g | socinfo X5U / RDK X5 V1.0 / py3.10.12 | v1-v3 + v4 small/medium（5） | 5/5 pass | v1/v2 maxdiff 0.0（H2 flat≡4D 逐位等价）；v3/v4 ≤6e-8；隔离稳定；bulbul/deerhound/kit fox/great grey owl 全部 top-1 正确 |
| s100 | soc_name S100 / py3.10.12 | v1-v4 small+medium + resnet50/152（7） | 7/7 pass | 全部 zebra top-1；maxdiff ≤1.2e-7；隔离稳定 |
| s100 | 同上 | mobilenetv2 runtime/cpp | pass | 首次并行编译 cc1plus OOM，`BUILD_JOBS=1` 后构建运行，TOP-1 zebra（launcher 自带 BUILD_JOBS 覆盖） |
| s600-64g | S600 / py3.12.3 | 同 s100（7） | 7/7 pass | **S600 SSH 恢复后完成复测**；maxdiff ≤1.2e-7 |
| s100p | S100P / RDK S100P V1P0 | 负例（2） | 2/2 pass | `--target s100` → `Target mismatch: requested s100, detected s100p`；s100p 资产 → `No published sample asset matches target='s100p'`；均 rc=2 无回退 |
| x5-8g | socinfo X5U / RDK X5 V1.0 / py3.10.12 / 7.25 GB / OS 3.5.0-beta（192.168.3.208） | v1-v3 + v4 small/medium（5） | 5/5 pass | **板卡重刷镜像后恢复**（旧 IP .207 失效、authorized_keys 重置——默认口令登录一次以恢复公钥后全程密钥认证；新镜像 hbrt 3.15.55 vs 模型构建 3.15.47 版本告警，对照两侧同板同 runtime，不影响等价结论）；v1/v2 maxdiff 0.0（H2 在第二块 X5 复证）；v3/v4 ≤5.96e-8；隔离稳定；制品 SHA-256 与 x5-4g 完全一致 |

板测发现的两个产品缺陷（B1-D1 原始 F32 拒绝量化描述符、B1-D2 v4 下载丢
variant）已修复、主机回归测试落地并在板上复验通过（§6.2 条 6/7）。
运行器自身的两处 harness 缺陷（canonical 侧漏传 labels、负例期望短语过时）
记录在 board evidence 的 harness_fixes_not_product，不计入产品缺陷。
