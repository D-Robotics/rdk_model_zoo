# B2 批次评审 — efficientnet（x5+s 合并）+ efficientformer/efficientformerv2/efficientvit（x5）（2026-09-21）

> 作者：原执行者（Claude Code）。本文件按批次固定流程第 1 步在动手前写就：
> 预检、契约事实表、旧→新函数映射表、资产/清单映射、转换能力映射。
> **B1 推进背景（用户决定，如实记录）**：B1 独立复审基点 dd60911 判定
> R2–R6 closed、R1a/R1b 文档待修正；R1a/R1b 修复已完成（commit b835e7f，
> 待独立确认，B1 各行 Closed=no / Review=changes-required 不变）。用户
> 于 2026-09-21 授权：R1a/R1b 文档修正后直接开启 B2，无需等待 B1 下一次
> 独立复审；S600 MobileNetV2 C++ 保持 not-run 不阻断。见迁移台账
> "B1 R1a/R1b 修复与 B2 推进决定"节。

## 0. 预检（固定源与清点）

| 项 | 值 |
| --- | --- |
| rdk_x5 源 SHA | `ac115717197920355fc390bb04299b20e6436864`（tip = ac11571，x5-v1.1.3；本批执行前复核 tip 未漂移） |
| rdk_s 源 SHA | `380e1a2bf42041af54be6f34935e50197cfadff9`（tip = 380e1a2，s-v1.1.2；同上） |
| efficientnet 文件数 x5/s | 20 / 34（s 含 5 个变体导出器 + timm2onnx_local.py + x86_inference.py；x5 python-only，无 cpp） |
| efficientformer 文件数 x5 | 20（仅 x5；s 无此 sample，已核实 s models.yaml 无行） |
| efficientformerv2 文件数 x5 | 20（仅 x5，同上） |
| efficientvit 文件数 x5 | 20（仅 x5，同上） |
| 执行基点 | develop@b835e7f（B1 收尾后 clean tree） |
| 提取方式 | `git archive <SHA> <path>` 展开 + 手工重构（不整目录恢复） |

两侧 `runtime/python` 均为标准化 wrapper（`XXXConfig` + 五段接口 +
`main.py`），与 B1 审计过的 mobilenet 形态一致；差异全部在契约事实层
（变体集、几何、标签默认）。四个 sample 均无 runtime/cpp（x5 源无、
s 源 efficientnet 无——cpp 首现于 B7/B8 批）。

## 1. 契约事实表（从源代码逐行核实）

| sample / variant | target | resize 默认 | resize 插值 | letterbox 插值 | 输出策略 | 输出语义（源声明） | 几何 | class_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet/b2 | x5 | 1（letterbox） | linear | linear | softmax | logits（wrapper docstring "single logits output"，post `scipy.softmax`） | 224×224 | 1000 |
| efficientnet/b3 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientnet/b4 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientnet/lite0 | s100+s600 | 1 | nearest | linear | softmax | logits（post `visualize.get_topk_predictions` = 稳定 softmax） | 224×224 | 1000 |
| efficientnet/lite1 | s100+s600 | 1 | nearest | linear | softmax | logits | **240×240** | 1000 |
| efficientnet/lite2 | s100+s600 | 1 | nearest | linear | softmax | logits | **260×260** | 1000 |
| efficientnet/lite3 | s100+s600 | 1 | nearest | linear | softmax | logits | **300×300** | 1000 |
| efficientnet/lite4 | s100+s600 | 1 | nearest | linear | softmax | logits | **380×380** | 1000 |
| efficientformer/l1 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientformer/l3 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientformerv2/s0 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientformerv2/s1 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientformerv2/s2 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |
| efficientvit/m5 | x5 | 1 | linear | linear | softmax | logits | 224×224 | 1000 |

依据（逐行核对源码）：

- x5 四个 wrapper（efficientnet/efficientformer/efficientformerv2/
  efficientvit @ac11571）形态相同：`Config(resize_type=1, topk=5)`；
  `pre_process` `resized_image(..., INTER_LINEAR)` + `bgr_to_nv12_planes`
  + 拼接 reshape `(1,3H/2,W,1)`（packed）；H/W 取 shape `[2]/[3]`；
  `post_process` `softmax(np.squeeze(outputs[out]))` → `argsort`。
- s efficientnet wrapper @380e1a2：`_resolve_model_soc()` 静默回退
  （soc_name 非 s600 一律 s100，含 s100p/未知板——统一架构改为显式
  解析）；默认模型 `/opt/hobot/model/{soc}/basic/efficientnet_lite0_
  224x224_nv12.hbm`；`pre_process` 不传插值 → `resized_image` 默认
  `INTER_NEAREST`，产出 split `{input0:y, input1:uv}`；H/W 取 shape
  `[1]/[2]`；`post_process` `visualize.get_topk_predictions`（数值
  稳定 softmax + argsort(-p)）。与 x5 侧 softmax 数学一致（B1 已锚定）。
- **lite0–lite4 为逐变体几何**（224/240/260/300/380，清单文件名与 YAML
  前缀均如此）；b2/b3/b4 与其余 x5 sample 全部 224。几何进契约表逐
  (variant,target) 登记，运行时再按元数据校验，不全局写死 224
  （与 B1 v4-medium 256 教训同源）。
- 两 s 侧标签默认 `test_data/imagenet_classes.names`，x5 侧
  `datasets/imagenet/imagenet_classes.names`——统一到根 `datasets/`
  （A2 位置），sample 本地副本保留在 test_data 供对照。

## 2. 旧→新函数映射表

### 2.1 x5 侧（四个 sample 共用形态）

| 源（`runtime/python/<model>.py` + `main.py` @ac11571） | 统一架构去向 |
| --- | --- |
| `<Model>Config(model_path, label_file, resize_type=1, topk=5)` | CLI `--asset-id/--variant/--resize-type/--top-k` + `resolve_selection`（路径优先 → 引用优先） |
| `__init__`: `HB_HBMRuntime(path)`、H/W 取 shape `[2]/[3]` | `RuntimeModelRunner.load()`（懒加载）+ `bind_model` 契约校验（几何来自契约表） |
| `set_scheduling_params(priority,bpu_cores)`（`{model:…}` 包装） | `RuntimeModelRunner.set_scheduling_params`（同一 API 形状） |
| `pre_process`: `resized_image(INTER_LINEAR)` + `bgr_to_nv12_planes` + reshape `(1,3H/2,W,1)` | `prepare_nv12` packed 分支：`resize_bgr`(linear) + 共享 `bgr_to_nv12_planes` + `as_packed` flat 1-D（H2 canonical） |
| `forward`: `run(inputs)[model_name]` | runner `__call__`（容器校验 + 扁平输出） |
| `post_process`: `softmax(squeeze)`→`argsort` | `ClassificationTask.post_process`：`raw_f32` → softmax → `topk_from_scores` |
| `main.py` 参数（x5 命名含 `--topk`） | 统一 CLI：`--target/--asset-id/--variant/--model-path/--test-img/--label-file/--top-k/--topk(旧拼写)/--resize-type/--priority/--bpu-cores/--img-save-path/--list-models/--dry-run` |
| 标签 `datasets/imagenet/imagenet_classes.names`（file_io.load_imagenet_labels） | 同文件（仓库根 `datasets/`） |
| `model/download.sh`（固定 3 条 wget，无校验） | `model/download.py` + `download.sh --target/--variant`（manifest 驱动 + sha256 报告） |
| `utils/py_utils` 引用（`sys.path.append("../../../../../")`） | sample 内不再引用；等价实现已在 `_shared` |

### 2.2 s 侧（efficientnet）

| 源（`runtime/python/efficientnet.py` + `main.py` @380e1a2） | 统一架构去向 |
| --- | --- |
| `_resolve_model_soc()` + 非 s600 一律 s100 **静默回退** | `samples/_shared/platforms.py` 显式解析；未知板/s100p 无清单行 → 显式报错（无回退） |
| 默认模型 `/opt/hobot/model/{soc}/basic/efficientnet_lite0_*.hbm`（import 时求值） | manifest 资产引用 `s:efficientnet:s<soc>/<file>` + `model/s100|s600/` 本地布局；系统路径在 README 记为手动准备选项 |
| `pre_process` split `{input0:y, input1:uv}`，H/W 取 shape `[1]/[2]` | `prepare_nv12` split 分支（`as_split`）+ bind 按 Y/UV 形状识别角色 |
| `forward` 返回整个 `run()` dict；post 内 `outputs[model][out][0]` 延迟索引 | runner 统一扁平化输出；task 按 bound output name 提取 |
| `post_process`：`visualize.get_topk_predictions` | 契约 score policy `softmax`（数学同源：稳定 softmax + argsort(-p)） |
| `main.py` 参数（`--model-path/--priority/--bpu-cores/--test-img/--label-file/--top-k/--resize-type`，默认模型 soc 探测） | 同 2.1 统一 CLI 集（多 `--variant` 区分 lite0–4） |
| `model/download_model.sh`（soc 探测 + s100 回退 + `/opt/hobot` 安装） | `download.sh --target/--variant`（显式 target，无回退；输出 sample 本地，`/opt` 记为手动选项） |
| `run.sh`（soc 探测回退 + 自动 wget） | `run.sh` = `main.py "$@"` 透传（无隐式下载；模型准备显式化） |

### 2.3 资产/清单映射（文件名保持发布事实不变）

| sample | x5 清单 id → 资产 | s 清单 id → 资产 |
| --- | --- | --- |
| efficientnet | `efficientnet` → `EfficientNet_{B2,B3,B4}_224x224_nv12.bin`（sha256 null） | `efficientnet` → `s100|s600/efficientnet_lite{0..4}_{224,240,260,300,380}_*.hbm`（10 行，sha256 null） |
| efficientformer | `efficientformer` → `EfficientFormer_{l1,l3}_224x224_nv12.bin` | —（s 无） |
| efficientformerv2 | `efficientformerv2` → `EfficientFormerv2_{s0,s1,s2}_224x224_nv12.bin` | —（s 无） |
| efficientvit | `efficientvit` → `EfficientViT_m5_224x224_nv12.bin` | —（s 无） |

清单动作：s `efficientnet` 行 `download_scripts` 由 `download_model.sh`
改指统一 `download.sh`（`sample_path` 已是 `samples/vision/efficientnet`
不动）；x5 四行 `sample_path`/`download_scripts` 均已正确不动。variant
判定用每 sample 显式 `{filename: variant}` 表（b2/b3/b4、lite0–4、
l1/l3、s0/s1/s2、m5），不做文件名猜测。s100p：无已发布资产 → 显式
"no published asset" 错误（沿 B1 负例语义）。

## 3. 转换能力映射与已知缺口（逐 config 对照 YAML 与脚本）

### 3.1 x5 四 sample（全部缺口同形，B1-R3 教训适用）

全部 YAML 事实（逐文件 grep 核实）：march `bayes-e`；`input_type_rt
nv12` / `input_type_train rgb` / `input_layout_train NCHW`；`norm_type
data_mean_and_scale`，mean `123.675 116.28 103.53`，scale
`0.01712475 0.017507 0.01742919`；`cal_data_dir ./calibration_data_rgb_f32`
（float32）。

| sample | config | onnx | 前缀 vs 清单文件名 | 缺口 |
| --- | --- | --- | --- | --- |
| efficientnet | B2/B3/B4 | `./efficientnet_b{2,3,4}.onnx` | 前缀均为 `EfficientNet_224x224_nv12`（**无 B2/B3/B4 区分**）vs 清单 `EfficientNet_B{2,3,4}_224x224_nv12.bin` → 需重命名对齐 | 无导出脚本（README 仅 4 步通用指引）；`calibration_data_rgb_f32` 无产出脚本；B3 `working_dir: model_output`（B2/B4 为 `EfficientNet_224x224_nv12`）；三份 YAML 均带 `debug_mode: dump_calibration_data`、B2/B4 另有 B3 没有的 `node_info` int16 配置——源分支原样保留的不一致，如实披露 |
| efficientformer | l1/l3 | `./efficientformer_l{1,3}.onnx` | 前缀均 `EfficientFormer_224x224_nv12`（无 l1/l3）vs 清单 `EfficientFormer_l{1,3}_*.bin` → 需重命名 | 同上（无导出脚本、无校准产出脚本） |
| efficientformerv2 | s0/s1/s2 | `./efficientformerv2_s{0,1,2}.onnx` | 前缀 `EfficientFormerv2_s{0,1,2}_224x224_nv12` **与清单一致** ✓ | 无导出脚本、无校准产出脚本；s0 `working_dir` 为 `int16_model_output` 后缀（s1/s2 不同）——原样保留 |
| efficientvit | m5 | `./efficientvit_m5.onnx` | 前缀 `EfficientViT_msra_224x224_nv12` vs 清单 `EfficientViT_m5_*.bin` → msra≠m5 需重命名 | 同上 |

x5 转换 README（双语）重写为按 sample×variant 的诚实声明：YAML 可用、
ONNX 需用户按源指引自备（timm 导出流程记录自源 README）、
`calibration_data_rgb_f32`（float32 RGB NCHW）为**缺失前提**（无产出
配方，改名不是修复）、前缀不匹配的重命名步骤逐 config 列出、源
working_dir/debug_mode 不一致披露。不虚构可复现流程。

### 3.2 s efficientnet（完整配方，逐项对照一致）

| 项 | 值（脚本 vs YAML 逐项核对） |
| --- | --- |
| config × 5 | `efficientnet_lite{0..4}_config.yaml`：march `nash-e`；nv12/rgb/NCHW；mean `127 127 127`，scale `0.007843×3`；`cal_data_dir ./calibration_data_rgb`（float32）；前缀 = 清单文件名逐字一致 ✓ |
| 导出器 × 5 | `get_efficientnet_lite{N}_onnx.py` + `timm2onnx_local.py`（timm `tf_efficientnet_lite{N}` checkpoint → ONNX） |
| 校准脚本 | `get_calibration_data.py`：`ScaleTransformer(255)` + `MeanTransformer([127,127,127])` + `ScaleTransformer(0.007843)`——**与 YAML mean/scale 数值一致**（非 resnet152 的 0.017 差异形态）；输出 `./calibration_data_rgb/` = YAML `cal_data_dir` ✓ |
| x86 对照 | `x86_inference.py` 保留 |
| S600 | YAML 默认 `nash-e`（S100）；S600 重编译改 `nash-p`（README 记录，与 resnet152 同形） |

### 3.3 评估能力

两侧 evaluator 均为 README 记录型（无执行实现）：x5 为 Float/Quant
Top-1 + 延迟/FPS 表（源分支记录，未在本仓重测）；s 为功能检查 +
性能记录表。合并后 evaluator README（双语）按 target 分表保留两份源
记录并注明"源分支记录，未重测"；functional check 命令更新为统一入口。

## 4. 本批架构决策

1. **无 `_shared` 变更**：四个 sample 的运行时面（labels/cls_binding/
   tensor_io/model_runner/classification/platform_profile）B1 已全部
   就位，本批是纯消费——每个 sample 只带 `model_binding.py` 契约表 +
   门面 re-export + `main.py` + tests。第二消费者规则不触发新提升。
2. **变体命名**：`b2/b3/b4`（x5）、`lite0..lite4`（s）、`l1/l3`、
   `s0/s1/s2`、`m5`——与源一一对应，CLI `--variant` choices 显式枚举。
3. **geometry 进契约表**：efficientnet 8 个 (variant,target) 组合几何
   逐项登记（224×8 中 lite1–4 为 240/260/300/380）。
4. **转换材料合并策略**：efficientnet/conversion/ 同时收 x5 B2/B3/B4
   YAML（+重写 README）与 s 全套 13 文件（5 YAML + 5 导出器 + timm
   helper + 校准 + x86）；s 侧 5 个脚本按逐字节迁入处理（源分支文件，
   迁移不改内容），README 重写为统一双语契约。
5. **test_data 合并**：保留双侧素材（x5 redshank + 架构图，s
   deerhound/zebra + 架构图 + 标签副本 + imagenet_1k.json）；统一默认
   测试图 `Scottish_deerhound.JPEG`（B1 样例一致的选择；板测对照时按
   各源默认图显式传参）。

## 5. 批内顺序

1. efficientnet（x5+s 合并，唯一双源 sample，先行验证模板）；
2. efficientformer / efficientformerv2 / efficientvit（x5 单源，套模板）；
3. 清单更新（s efficientnet download_scripts）；
4. 全量主机验证（新旧 sample 全套 + `_shared` + checker 单测）+ Q3
   检查器（scope 增至 11 samples）+ CI 同命令；
5. 台账 B2 行回填、本文件补执行结果、证据 JSON、commit；
6. 板端冒烟（既有授权）：x5-8g/x5-4g 四 sample、s100/s600
   efficientnet（先同板源实现基线，后迁移对照）；s100p 负例；
   无法执行记 not-run 及原因。

## 6. 执行结果（2026-09-21 回填）

### 6.1 提交清单

| sample | commit | 文件数 | 说明 |
| --- | --- | --- | --- |
| efficientnet（x5+s） | `ac09101` | 54 | 8 变体契约表（x5 B2/B3/B4 224 + s lite0–4 逐变体几何 224/240/260/300/380）；s 侧 13 个转换文件逐字节迁入并锚定校准链；x5 3 YAML SHA 固定 + 缺口披露 |
| efficientformer | `27fea1a` | 35 | l1/l3 双变体，缺省 l3 保持源默认；无变体前缀重命名缺口以 assertNotEqual 钉住 |
| efficientformerv2 | `a8a9ace` | 35 | s0/s1/s2，缺省 s0 保持源默认；**正向锚定**：`output_model_file_prefix + '.bin' == Manifest 基名`（三变体全等，无需重命名）；s0 独有 debug/optimization 不对称钉住 |
| efficientvit | `4f08785` | 35 | 单变体 m5；`msra` 无变体前缀 vs `m5` 基名缺口钉住；0.99999 分位与 28 节点 Softmax int16 摆放锚定 |
| manifest + 台账 + 证据 | （本提交） | — | s `efficientnet` 行 `download_scripts` → `download.sh`；B2 台账行回填；本文件 §6；evidence JSON |

### 6.2 主机验证（cwd：仓库根，`.venv/bin/python`）

- B2 新套件：efficientnet 25 OK / efficientformer 25 OK /
  efficientformerv2 26 OK / efficientvit 26 OK（共 102）。
- B1 回归：resnet 52、mobilenetv1 17、v2 24、v3 17、v4 20、
  ultralytics_yolo 59、paddle_ocr 44——全部 OK（233）。
- `_shared` 71 OK；checker 单测 27 OK。
- 逐 sample Q3 检查器：4/4 均 0 violations、1 skip（main.py CLI 层
  R-STAGE-PURITY 策略跳过）。
- CI 同命令（migration scope）：**11 samples、0 violations、
  84 exemptions、rc=0**——84 条 B9 延期基线原样未动；范围由 7 增至
  11 系 B2 台账行 Refactor=done 后按进度区解析所得。

### 6.3 执行中发现并修复的问题（作者自检记录）

1. model_binding 委托 API 首次写错（虚构 build_* 函数，导入即
   TypeError）→ 更正为显式 `list_assets/resolve_selection/bind_model`
   委托（efficientformer）。
2. `classification_profiles()` 缺必填 `url_prefix_s`（x5-only sample
   同样必填）→ 以惰性值 + 如实 docstring 修复（无 S 清单行，该前缀
   永不被消费）。
3. efficientformer test_conversion_layout.py 出现无意义断言
   （对死字符串 assertNotIn）→ 换成有意义的
   `assertFalse((CONVERSION/'calibration_data_rgb_f32').exists())`。
4. 本文件 §3.1 事实错误：efficientnet 的 `debug_mode` 最初记为 B3 独有
   ，实际三份 YAML（B2/B3/B4）均有 → §3.1 已在 efficientnet 提交内
   更正，并补记 B2/B4 有而 B3 没有的 `node_info` 不对称。
5. 不可验证引用：efficientnet 根 README 曾以 arXiv 1904.01146 指称
   EfficientNet-Lite 原始论文 → 改为按源交付引用 tensorflow/tpu 仓库
   （双语）。
6. efficientformerv2 sed 改名残留：文件名字面量（错误大小写/旧变体名
   、双变体列表）首批补丁静默未命中、第二批中途退出 → 全部经显式
   Edit 逐一核对面修复；教训：sed 产物先扫残留字面量再跑测试。
7. efficientvit Softmax 摆放数初判 27，grep 实数 28 → 测试锚定 28。

### 6.4 材料披露

- **test_data 大小写冲突（efficientnet，x5 合并）**：x5 侧
  `EfficientNet_architecture.png` 与 s 侧 `efficientnet_architecture.png`
  仅文件名大小写不同；两侧字节**完全一致**（sha256 `f0c7ccbe…`，
  170885 bytes，双源核验）；APFS 单目录无法并存两种大小写，保留 s 侧
  小写名一份——零内容损失，仅名称大小写差异。
- legacy 生成物（result.jpg/result.png）不迁移：均为运行期副作用文件
  而非输入素材；统一运行时仅在 `--img-save-path` 时写图。
- efficientvit 源 `run.sh` 的 `/opt/hobot/model/x5/basic/` 优先 + 自动
  wget 回退：两个隐式行为均已移除（下载显式化、run.sh 纯透传）。
- 已发布基准表（x5-v1.1.3）按"源分支记录、未重测"如实复制；
  efficientvit 源表未声明延迟线程条件，照实记录。
- 论文引用核实：EfficientNet 1905.11942；EfficientFormer 2206.00171；
  EfficientFormerV2 2212.08059（snap-research/EfficientFormer）；
  EfficientViT 2305.07027（microsoft/Cream）；EfficientNet-Lite 维持
  源交付引用（tensorflow/tpu，原始论文链接在源中不可验证）。

### 6.5 板测矩阵（既有授权；执行前为 not-run）

| 板位 | 范围 | 状态 |
| --- | --- | --- |
| x5-8g | 四 sample 全变体默认图冒烟 + 同板 legacy 对照 | not-run（待执行） |
| x5-4g | 同上 | not-run（待执行） |
| s100 | efficientnet lite0–4 冒烟 + legacy 对照 | not-run（待执行） |
| s600 | efficientnet（nash-p 制品） | not-run（待执行） |
| s100p | 拒绝负例（无已发布资产） | not-run（待执行） |

主机测试不替代板测；各项执行后在此回填实测值，无法执行记 not-run
及原因。

### 6.6 状态

作者自检完成；独立评审未开始。全部 B2 行 Closed=no。完成后停在 B2，
不进入 B3（用户指示）。
