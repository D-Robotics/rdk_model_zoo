[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive Python 推理

<a id="environment"></a>
## 环境

实际推理需要 S100P 或 S600、匹配的 `hbm_runtime`、Python、NumPy 和 OpenCV。两个目标使用独立发布模型，不存在 S100/X5 资产。[显式下载器](../../model/README_cn.md)负责准备并校验 HBM；推理不安装依赖、不下载模型。本次迁移未运行真实 SDK/板卡，主机测试显式注入运行夹具。

以下命令均从仓库根目录执行，Shell 包装入口也会切换到该目录。源任务类混合 SDK 资源管理、量化和绘图；新任务使用共享具名数组 runner 传输，将数据 IO、变换和可视化分别放入独立模块。

<a id="usage"></a>
## 单案例与批量运行

显式选择目标检查，不导入板端 SDK、不打开 HBM：

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --dry-run
```

准备好对应板端环境和模型后，使用新目录执行随附案例：

```bash
bash samples/vision/diffusiondrive/runtime/python/run.sh --target s600 --input-npz samples/vision/diffusiondrive/test_data/case_017/inputs.npz --output outputs/diffusiondrive_case017
```

检查或执行源五案例：

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s100p --output outputs/diffusiondrive_cases --dry-run
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s100p --output outputs/diffusiondrive_cases
```

批量 dry-run 校验全部五份输入并打印命令，不执行 SDK、不创建输出。实际运行复用单案例 CLI，与源实现一样每案例加载一次模型，在首个非零返回处停止，`batch-report.json` 保留已完成返回码、剩余案例及可用报告摘要。部分完成不等于五例通过。批量输出改用 `--output`，替代源包装入口的位置参数。

<a id="parameters"></a>
## 参数

下表为单案例解析器的字面默认值，解析后的路径另行说明。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` / `--platform` | `auto` | 显式目标或可识别本机身份；未知主机不回退 S600 |
| `--asset-id` | `null` | 按目标推断；auto 下可由它确定目标 |
| `--model-path` | `null` | 解析至 sample model 目录的目标 HBM；外部路径须提供资产 ID |
| `--input-npz` | `samples/vision/diffusiondrive/test_data/reference_inputs.npz` | 精确四份逻辑特征数组 |
| `--output` | `outputs/diffusiondrive` | 新正式结果目录 |
| `--output-npz` | `null` | 可选额外解码归档副本，正式归档始终保留 |
| `--img-save-path` / `--output-image` | `null` | 可选额外 PNG/JPEG/BMP，按扩展名编码 |
| `--agent-score-thres` | `0.5` | 有限 [0,1]，沿用源 sigmoid 概率 >= 比较 |
| `--priority` | `0` | SDK 整数优先级 0..255 |
| `--bpu-cores` | `[0]` | 一个或多个非负核 ID，实际支持取决于运行库 |
| `--list-models` | `false` | 只列清单身份/校验和，不执行 |
| `--dry-run` | `false` | 只解析并打印，不执行 |

检查模式互斥。批量入口复用目标/资产/模型/阈值/调度/检查参数，另有 `--cases-root`（默认 `samples/vision/diffusiondrive/test_data`）与 `--output`（默认 `outputs/diffusiondrive_cases`）。它没有单案例 `--input-npz` 或额外输出文件参数。案例顺序为000、017、042、073、099。

输出目录及额外路径必须不存在。额外目标须互不相同，不得覆盖正式数组、图片或报告。保留源图片别名与 `--platform`，原输出文件默认位置改为每次运行独立目录；移除环境目标覆盖及隐式下载。直接 Python 命令的相对路径按当前工作目录解析。

<a id="results"></a>
## 保存的结果

| 文件 | 契约 |
| --- | --- |
| `physical_inputs.npz` | 四份实际量化/转换输入，保留名称、形状、类型 |
| `raw_outputs.npz` | 反量化前的四份物理原始输出 |
| `outputs.npz` | 解码轨迹、Agent 状态/概率/掩码、BEV logits/标签 |
| `result.png` | 相机/BEV/LiDAR/轨迹/Agent 可视化 |
| `report.json` | 资产与文件摘要、含量化信息的实际输入/输出元数据、调度、阈值、UTC 时间段及处理边界 |

解码数组：float32 轨迹 `[1,8,3]`、Agent 状态 `[1,30,5]`、概率 `[1,30]`、BEV logits `[1,7,128,256]`；bool Agent 掩码 `[1,30]`；uint8 BEV 标签 `[1,128,256]`。运行库提供版本时记录，否则为 `unknown`。UTC 字段界定工作时间，不是延迟基准。不预热、不重新生成噪声、不执行控制。

额外解码归档为字节副本；额外图片单独编码，JPEG 有损，不要求与 PNG 像素一致。写入失败可能留下不完整目录，使用前检查返回码与报告。返回 0 表示处理/输出完成，不代表驾驶质量。[离线评估器](../../evaluator/README_cn.md)读取解码 `outputs.npz`，不读取原始张量。

<a id="integration-example"></a>
## 应用集成

在准备好的 S600 上从仓库根目录执行，与 CLI 使用相同任务路径，不写文件、不绘图：

```python
from samples.vision.diffusiondrive.runtime.python.model_binding import resolve_selection
from samples.vision.diffusiondrive.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.diffusiondrive.runtime.python.data_io import load_features
from samples.vision.diffusiondrive.runtime.python.diffusiondrive import DiffusionDriveTask

selection = resolve_selection("s600")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
features = load_features("samples/vision/diffusiondrive/test_data/reference_inputs.npz")
task = DiffusionDriveTask(runner, binding, agent_score_threshold=0.5)
result = task.predict(features)
assert result["trajectory"].shape == (1, 8, 3)
assert result["bev_labels"].shape == (1, 128, 256)
```

需要保留物理输入与原始输出时分阶段调用。结果独立持有数组，使用任务时须保持 SDK runner 存活；未建立同步机制前不应并发共享。任务不再持有 SDK，也不提供 `__call__` 别名，请使用 `predict`。它不接受原始相机/LiDAR 传感器数据替代准备后的特征张量。

<a id="stage-io"></a>
## 阶段 IO 与量化

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `pre_process` | 精确四份有限 float32 逻辑数组 | 按绑定类型/量化生成的平坦名称→物理数组映射 |
| `forward` | 物理映射 | 独立原始具名输出，不做语义解码 |
| `post_process` | 精确匹配元数据的四份原始数组 | 独立解码六数组结果 |
| `predict` | 逻辑特征 | 组合上述三阶段 |

逻辑输入：camera `[1,3,256,1024]`、lidar `[1,1,256,256]`、status `[1,8]`、noise `[1,20,8,2]`。源输出名称/形状：trajectory `[1,8,3]`、agent_states `[1,30,5]`、agent_labels `[1,30]`、bev_semantic_map `[1,7,128,256]`。绑定要求单模型和精确名称集合，名称顺序不影响绑定。

物理类型可为 int8/uint8/int16/uint16/int32/uint32/float16/float32，须通过真实元数据和变换校验。整数要求显式正且有限的 SCALE 描述，输入 scale 只能是逐张量。浮点空量化信息表示类型转换/直通，NONE 描述允许值为零的零点占位；非空浮点 SCALE 保持源仿射行为。输出支持逐轴 scale，长度须匹配轴；单个零点广播到全部通道，修正源 reshape 错误。整数零点须为范围内整数。绑定保留变换值快照，不深拷贝 SDK 描述对象。

输入量化保留源 float32 `rint(x/scale + zero)`；最终整数转换前用 float64 边界裁剪，避免 int32/uint32 上界回绕。缺少整数 scale、元数据错误、非有限输入/结果均失败。输出先反量化，再执行源 sigmoid（logits 裁剪 [-60,60]）、Agent 阈值和通道轴 BEV argmax。不插入未说明的传感器归一化、随机噪声或新规划算法。本次主机迁移尚未观察真实 HBM 元数据。

<a id="troubleshooting"></a>
## 排障

- auto 无法识别主机：检查/准备可显式选目标；真实执行仍要求匹配物理身份。
- 摘要不符：使用正确发布文件，不能重命名自定义模型绕过验证。
- 特征错误：保留精确 float32 形状/名称，不增加归档字段或重新生成噪声。
- 量化拒绝：先记录实际元数据、解决契约差异，再考虑修改校验。
- BEV 近乎全灰：灰色表示道路，先检查 logits/标签与参考指标，不能直接判定色表错误。
- 批量失败：检查各例报告与 `remaining_cases`，修正原因后使用新目录重跑。

主机测试将六份浮点案例与真实源后处理/绘图逐项对照，覆盖量化边界，并用注入 SDK 运行真实 CLI。它们不证明物理 HBM 兼容、板端相等、NAVSIM 精度或性能。历史表格和验证边界见评估说明。
