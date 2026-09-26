[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 确定性示例

本目录逐字节保留源分支六组输入/参考对及六幅历史结果图。源文档将图片描述为 S600 结果；它们不是本次迁移新生成的图片或板端验证证据。原说明保留于 `platforms/s/samples/vision/diffusiondrive/test_data`。

## 输入契约

| 张量 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `camera` | `[1,3,256,1024]` | float32 | 已准备的左/前/右相机 RGB 全景 |
| `lidar` | `[1,1,256,256]` | float32 | 已准备的 LiDAR BEV 直方图 |
| `status` | `[1,8]` | float32 | 已准备的自车状态与驾驶指令 |
| `noise` | `[1,20,8,2]` | float32 | 固定截断扩散噪声 |

它们是逻辑浮点特征，不是 HBM 物理量化缓冲区。Sample 按实际运行元数据量化，不从原始传感器重建 NAVSIM 特征。未提供原始数据集样本 ID、准备脚本或完整传感器来源，不应超出源描述猜测 status 各分量含义。确定性对照应保持噪声不变；固定输入本身不能证明每个 SDK 上的运行都确定。

## 文件与参考输出

默认数据为 `reference_inputs.npz`、`reference_outputs.npz`，`reference_result.png` 为历史显示图。每个案例目录包含 `inputs.npz`、`reference_outputs.npz`、`result.png`。

| 浮点参考张量 | 形状 | 含义 |
| --- | --- | --- |
| `trajectory` | `[1,8,3]` | 八个自车位姿 `[x,y,heading]` |
| `agent_states` | `[1,30,5]` | 三十个候选的 `[x,y,heading,length,width]` |
| `agent_labels` | `[1,30]` | Agent logits，不是概率 |
| `bev_semantic_map` | `[1,7,128,256]` | 七类 BEV logits，不是标签 |

源文档称其为 PyTorch 浮点输出，不是标注真值。源元数据与已知 HBM 摘要不能固定完整的上游 checkpoint/导出历史。随附数组均为有限 float32，运行结果另有原始与解码两种契约。

## 五个源案例

下表数值均保留自历史 S600 记录，不是本次主机迁移测量。

| 案例 | 场景 | 预测 Agent 数 | BEV 像素一致率 | BEV 平均 IoU |
| --- | --- | ---: | ---: | ---: |
| `case_000` | 宽阔信号灯路口 | 7 | 0.944061 | 0.868425 |
| `case_017` | 附近有交通参与者的信号灯路口 | 7 | 0.944000 | 0.761217 |
| `case_042` | 密集多车道城市交通 | 13 | 0.966736 | 0.728740 |
| `case_073` | 开阔直行大道 | 6 | 0.966156 | 0.876931 |
| `case_099` | 检测到较多 Agent 的宽阔路口 | 14 | 0.958862 | 0.899669 |

平均 IoU 包含任一预测中出现的类别。case_017/case_042 少量 class-4 像素会显著影响宏平均 IoU，因此同时保留像素一致率。[评估说明](../evaluator/README_cn.md)明确定义当前检查与指标。

| case_017 | case_042 |
| --- | --- |
| ![历史路口结果](case_017/result.png) | ![历史密集交通结果](case_042/result.png) |
| case_073 | case_099 |
| ![历史大道结果](case_073/result.png) | ![历史宽阔路口结果](case_099/result.png) |

另保留[默认历史结果](reference_result.png)和 [case_000 历史结果](case_000/result.png)。

## 运行示例

在准备好的 S600 上，从仓库根目录执行：

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --input-npz samples/vision/diffusiondrive/test_data/case_017/inputs.npz --output outputs/diffusiondrive_case017
```

指定目标和新的批量目录运行五个案例：

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s600 --output outputs/diffusiondrive_cases
```

主机检查可在批量命令追加 `--dry-run`，它验证五份输入并打印命令，不执行 SDK、不创建输出目录。批量运行在首个失败案例处停止，保留已完成记录，不将后续案例记为通过。具备 S100P 板卡时使用 `--target s100p` 和对应独立模型。不自动下载，不回退到其他板卡目标。

## 坐标与显示解释

轨迹/Agent 的 x 为自车前方、y 为左方，单位米。绘图使用 0.25 m 栅格像素及源旋转/裁剪，让前方朝上显示。蓝色框为自车，橙色为规划轨迹，红框为超过 Agent 阈值的候选。显示不产生控制指令，不执行轨迹。

| 类别 ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 含义 | 背景 | 道路 | 人行道 | 中心线 | 静态物体 | 车辆 | 行人 |

道路为灰色；近乎全灰的 BEV 图可能表示道路预测占绝大多数，不代表缺失色表。图片用于可视化，不能替代原始数组对照。DiffusionDrive/NAVSIM 资产仍受其原始条款约束，本目录不提供完整的已授权评估数据集。
