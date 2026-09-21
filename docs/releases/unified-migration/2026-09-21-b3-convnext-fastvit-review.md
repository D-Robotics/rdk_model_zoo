# B3 批次评审 — convnext / edgenext / fasternet / fastvit（x5）（2026-09-21）

> 作者：原执行者（Claude Code）。本文件按批次固定流程第 1 步在动手前写就：
> 预检、契约事实表、旧→新函数映射表、资产/清单映射、转换能力映射。
> **B2 关闭背景**：B2 独立关闭确认（Codex，21c833a 基点，[关闭报告]
> (2026-09-21-b2-independent-closure.md)，pass/ready/Closed=yes）；数值结论
> 限定为完整 Top-5 类别与 CLI 打印精度分数一致。用户指示 B3 完成后停在
> 等待独立评审，不进入 B4。

## 0. 预检（固定源与清点）

| 项 | 值 |
| --- | --- |
| rdk_x5 源 SHA | `ac115717197920355fc390bb04299b20e6436864`（tip = ac11571，x5-v1.1.3；本批执行前 `git rev-parse rdk_x5` 复核未漂移） |
| 执行基点 | develop@d91ac89（B2 关闭材料提交后 clean tree） |
| 文件数 convnext/edgenext/fasternet/fastvit | 24 / 21 / 22 / 21（含各 sample 双语 README ×5 级 + conversion YAML 3/4/4/4 + test_data） |
| 提取方式 | `git show ac11571:<path>` / `git archive` 展开 + 手工重构（不整目录恢复） |
| S 侧 | 四个 sample 均无 S 源（s models.yaml 无行，已核实） |
| C++ | convnext 有 `runtime/cpp/` 目录但为**显式占位**：CMakeLists 仅 `message(STATUS "No C++ runtime source is provided for this sample yet.")`，run.sh 打印未实现并 exit 1——语言覆盖仍为 python-only（P0 表 "C++ —" 口径一致）；其余三个无 cpp 目录 |

四个 wrapper 均为 B1/B2 已审定的 x5 标准化形态
（`XXXConfig(resize_type=1, topk=5)` + 五段接口 + packed NV12 +
`scipy.softmax` + `argsort`），逐行核实无差异；差异全部在契约事实层
（变体集、默认变体、默认测试图）与入口隐式行为（run.sh / 下载）。

## 1. 契约事实表（从源代码逐行核实）

| sample / variant | target | resize 默认 | 插值 | 输出策略 | 几何 | class_count | 源默认测试图 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| convnext/atto（唯一已发布变体） | x5 | 1（letterbox） | linear | softmax（scipy） | 224×224 | 1000 | cheetah.JPEG |
| edgenext/base（源默认） | x5 | 1 | linear | softmax | 224×224 | 1000 | Zebra.jpg |
| edgenext/small、x_small、xx_small | x5 | 1 | linear | softmax | 224×224 | 1000 | 同上 |
| fasternet/S（源默认） | x5 | 1 | linear | softmax | 224×224 | 1000 | drake.JPEG |
| fasternet/T0、T1、T2 | x5 | 1 | linear | softmax | 224×224 | 1000 | 同上 |
| fastvit/S12（源默认） | x5 | 1 | linear | softmax | 224×224 | 1000 | bucket.JPEG |
| fastvit/SA12、T12、T8 | x5 | 1 | linear | softmax | 224×224 | 1000 | 同上 |

依据（逐行核对 @ac11571）：

- 四个 wrapper 同构：H/W 取 shape `[2]/[3]`；`pre_process`
  `resized_image(..., INTER_LINEAR)` + `bgr_to_nv12_planes` packed；
  `post_process` `softmax(np.squeeze(outputs[out]))` → `argsort`
  （`[-topk:][::-1]`）。与 B2 四 sample 完全同形。
- main.py 全部 `--resize-type` default=1、`--topk` default=5（x5 拼写）；
  `--img-save-path` 默认：edgenext/fasternet/fastvit 为
  `test_data/result.jpg`（绝对拼接），**convnext 为相对路径
  `"result.jpg"`**——统一后按"仅 `--img-save-path` 时写图、默认不写"
  处理，差异如实记录。
- 标签默认均为仓库根 `datasets/imagenet/imagenet_classes.names`。
- 源默认变体（main.py DEFAULT_MODEL_PATH）：convnext **atto**、
  edgenext **base**、fasternet **S**、fastvit **S12**——统一后保持。

## 2. 旧→新函数映射表（x5 侧，四 sample 共用形态）

| 源（`runtime/python/<model>.py` + `main.py` @ac11571） | 统一架构去向 |
| --- | --- |
| `<Model>Config(model_path, label_file, resize_type=1, topk=5)` | CLI `--asset-id/--variant/--resize-type/--top-k/--topk(旧拼写)` + `resolve_selection` |
| `__init__`: `HB_HBMRuntime(path)`、H/W 取 `[2]/[3]` | `RuntimeModelRunner.load()` + `bind_model` 契约校验 |
| `set_scheduling_params(priority,bpu_cores)`（`{model:…}` 包装） | `RuntimeModelRunner.set_scheduling_params`（同一 API 形状） |
| `pre_process`: `resized_image(INTER_LINEAR)` + packed NV12 | `prepare_nv12` packed 分支（H2 canonical flat 1-D） |
| `forward`: `run(inputs)[model_name]` | runner `__call__`（容器校验 + 扁平输出） |
| `post_process`: `softmax(squeeze)`→`argsort` | `ClassificationTask.post_process`：`raw_f32` → softmax → `topk_from_scores` |
| main.py 参数（`--topk` 拼写、`--img-save-path` 默认写图） | 统一 CLI 集；默认不写图，仅 `--img-save-path` 时写 |
| `utils/py_utils` 引用（`sys.path.append`） | sample 内不再引用；等价实现已在 `_shared` |
| run.sh（edgenext/fasternet/fastvit：`/opt/hobot/model/x5/basic/` 优先 → sample 本地 → **自动 wget**；convnext：透传） | `run.sh` = `main.py "$@"` 纯透传（两个隐式行为均移除；下载显式化） |
| `model/download.sh`（edgenext/fastvit/fasternet 固定多条 wget；convnext 变量单条 `wget -c`；均无校验） | `model/download.py` + `download.sh --target/--variant`（manifest 驱动 + sha256 报告） |

## 3. 资产/清单映射（文件名保持发布事实不变）

| sample | 清单资产（docs/release/x5/models.yaml，sha256 均 null） | 变体表 |
| --- | --- | --- |
| convnext | `ConvNeXt_atto_224x224_nv12.bin`（**仅 1 行**） | atto（femto/nano 有转换配方、**无已发布资产**——如实披露，不虚构清单行） |
| edgenext | `EdgeNeXt_{base,small,x_small,xx_small}_224x224_nv12.bin`（4 行） | base/small/x-small/xx-small |
| fasternet | `FasterNet_{S,T0,T1,T2}_224x224_nv12.bin`（4 行） | S/T0/T1/T2 |
| fastvit | `FastViT_{S12,SA12,T12,T8}_224x224_nv12.bin`（4 行） | S12/SA12/T12/T8 |

清单动作：**无**。四行 `sample_path: samples/vision/<name>` 已是统一
路径，`download_scripts` 指向 `model/download.sh`（统一后同名同位），
均已核实不动。

## 4. 转换能力映射与已知缺口（逐 config 对照 YAML 与脚本）

全部 13 份 YAML 事实（逐文件核实，完全一致）：march `bayes-e`；
`input_type_rt nv12` / `input_type_train rgb`；`norm_type
data_mean_and_scale`，mean `123.675 116.28 103.53`，scale
`0.01712475 0.017507 0.01742919`；`cal_data_dir
./calibration_data_rgb_f32`。

| sample | config | 前缀 vs 清单文件名 | calibration | 缺口/锚定 |
| --- | --- | --- | --- | --- |
| convnext | atto/femto/nano | 前缀均 `ConvNeXt-deploy_224x224_nv12`（无变体）vs 清单 `ConvNeXt_atto_*` → **需重命名**（atto）；femto/nano 无清单对应 | default | 无导出脚本、无 rgb_f32 产出脚本；femto/nano 为**未发布配方**（转换 README 双语如实声明，不写成可发布流程） |
| edgenext | base/small/x_small/xx_small | 前缀逐字 = 清单基名（`EdgeNeXt_<variant>_224x224_nv12`）✓ **正向锚定**（同 B2 formerv2 形态） | max + `max_percentile 0.999` | 无导出脚本、无校准产出脚本；前缀全等无需重命名（测试钉住） |
| fasternet | S/T0/T1/T2 | 前缀均 `FasterNet_224x224_nv12`（无变体）vs 清单 `FasterNet_<variant>_*` → **需重命名** | default | working_dir 不对称：S=`model_output`、T0=`FasterNet_224x224_nv12_mix`、T1/T2=`FasterNet_224x224_nv12`——源分支原样保留并披露；无导出/校准产出脚本 |
| fastvit | S12/SA12/T12/T8 | 前缀均 `FastViT_224x224_nv12`（无变体）vs 清单 `FastViT_<variant>_*` → **需重命名** | default | working_dir 四者一致 `FastViT_224x224_nv12_mix`；无导出/校准产出脚本 |

转换 README（双语）按 B2 口径诚实声明：YAML 可用、ONNX 需按源自备、
`calibration_data_rgb_f32` 为缺失前提（改名不是修复）、前缀不匹配的
重命名步骤逐 config 列出、working_dir 不对称披露。不虚构可复现流程。

### 评估能力

四 sample evaluator 均为 README 记录型（x5 源分支的 Float/Quant Top-1
与延迟/FPS 表）：按"源分支记录，未重测"双语保留，functional check
命令更新为统一入口，比较口径写 Top-K + 容差 + 平局（沿 B2 复审后措辞）。

## 5. 本批架构决策

1. **无 `_shared` 变更**（预期）：纯消费 B1/B2 已就位的共享面；若执行
   中发现第二个真实消费者需求再按纪律提升。B2-R1 教训内建：变体默认
   按 per-target 声明（本批全部 x5-only，单字符串默认即可；**无 S 变体
   集交错**，B2-R1 形态不触发，仍以省略变体解析测试钉住每 sample 默认）。
2. **变体命名**与源一一对应（atto；base/small/x-small/xx-small；
   S/T0/T1/T2；S12/SA12/T12/T8），CLI `--variant` choices 显式枚举。
3. **convnext cpp 占位保留**：源 cpp 目录为显式"未实现"占位（CMake
   message + run.sh exit 1），统一后 python-only；占位文件随源保留并
   在 README 声明（不删除源事实、不宣称双语言）。
4. **run.sh 统一纯透传**；`/opt/hobot` 优先与自动 wget 两个隐式行为
   移除（沿 B2 efficientvit 先例，README 记为手动选项）。
5. **test_data 合并**：保留各源素材（cheetah/Zebra/drake/bucket 各自
   默认图 + 架构图 + inference.png）；统一默认测试图沿用各源默认
   （板测对照时按源默认图执行）。
6. **转换材料逐字节迁入 + SHA 固定**（13 YAML）；缺口披露见表。

## 6. 批内顺序

1. convnext（唯一多配方少资产行为 + cpp 占位 + 相对 img-save 默认，先
   行验证本批全部特异形态）；
2. edgenext（前缀正向锚定样板）；
3. fasternet → fastvit（套模板）；
4. 每 sample：主机套件 + checker；批次末：全量回归（B1+B2+`_shared`+
   checker 单测）+ CI 同命令；
5. 台账 B3 行回填、本文件补执行结果、证据 JSON、逐 sample 提交；
6. 板端冒烟（既有授权）：x5-8g/x5-4g 四 sample 全变体同板 legacy 对照
   + CLI + 默认入口（省略 variant）当场完整留证（沿 B2-R1-E 教训：
   部署文件哈希板上计算、完整 stdout/stderr、时间戳/argv/cwd/rc）；
   S 板不涉及（无 S 资产）；无法执行记 not-run 及原因。

## 7. 执行结果（2026-09-21 回填；板测待执行）

### 7.1 提交清单

| sample | commit | 文件数 | 要点 |
| --- | --- | --- | --- |
| convnext | `e843005` | 35 | atto 单已发布变体；femto/nano 为无资产配方（绑定表断言不含）；ONNX 引用互为错位（atto→femto.onnx、femto→外部 atto、nano→pico.onnx）+ 源 README 列了未交付的 pico.yaml——钉住披露；前缀无变体需重命名；基准表不含 atto（唯一已发布变体）如实披露 |
| edgenext | `b4f8aff` | 36 | 4 变体全发布；**前缀正向锚定**（前缀=清单基名，无重命名，测试钉住）；xca-Softmax int16 3/3/3/16 |
| fasternet | `69052d5` | 37 | 表内 id 小写 s/t0/t1/t2（共享绑定小写请求；文件名保留大写，双侧钉住）；无变体前缀 + working_dir 三种形态不对称披露；仅 T0 有 int16 摆放；源表 Params 与上游论文不符按发布记录 |
| fastvit | `f33167a` | 36 | 表内 id 小写 s12/sa12/t12/t8；ONNX 全指外部 01_common 路径（披露）+ 无变体前缀；int16 5/6/4/10 |

### 7.2 主机验证（cwd：仓库根，`.venv/bin/python`）

- B3 四套件：convnext 28 / edgenext 26 / fasternet 27 / fastvit 27 OK。
- 回归：B2 四 sample 106、B1+pilots 233 全 OK；`_shared` 71、checker 27。
- 逐 sample checker 4/4 零违规；**CI 同命令（台账 Refactor=done 后）：
  15 samples / 0 violations / 18 skips / 84 exemptions / rc=0**（范围
  11→15；84 条 B9 基线未动）。

### 7.3 执行中发现并修复的问题（作者自检记录）

1. 生成顺序缺陷（工具层，被套件当场捕获）：变体替换在模型名改名
   之后执行导致 fastvit 文件名/ id 残留旧拼写——逐一修正；受控断言
   （每次替换 count==1）纪律保持。
2. fasternet 变体大小写：表内 id 大写 S/T0 与共享绑定的小写化请求
   冲突（"Unknown sample variant 'S'"）——统一为小写 id，测试双侧钉住。
3. edgenext 清单计数沿用模板的 2 变体断言（4 != 2）——核对清单后修正
   为 4（发布资产数先核实再改测试）。
4. edgenext 转换测试初版断言"无 Softmax 摆放"（沿用 convnext 惯性）
   ——EdgeNeXt 实有 xca-Softmax（每份 3 处），改为钉住真实结构。
5. convnext 中文转换 README 用了中文锚点——checker 要求 CN 文件使用
   规范英文锚点（9 violations）——按 B2 约定修正。

### 7.4 板测（待执行）

按 B2-R1-E 证据标准：x5-8g/x5-4g 四 sample 全变体同板 legacy 对照 +
CLI + 省略变体默认入口；部署文件哈希板上计算、完整 stdout/stderr、
UTC 时间戳/argv/cwd/rc。S 板不涉及（无 S 资产）。数值结论将限定
Top-5 类别与 CLI 打印精度分数一致，不宣称原始张量逐位相同或数据集
精度。

### 7.5 状态

作者自检（主机侧）完成；板测未执行；独立评审未开始。全部 B3 行
Closed=no。板测与证据回填后停在 B3 等待独立评审，不进入 B4。
