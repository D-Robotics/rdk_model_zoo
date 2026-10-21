# 按任务的字段

任务代码定义在 `model_zoo_web/task-registry.json`。`facet_id` 是网页筛选项，`label_zh` 是卡片上的任务名，`accuracy_profile` 指向 `accuracy-metrics.json` 里的精度字段。`null` 表示这种任务还不能标成完整。

所有任务在待补状态下的必填、可选摘录和禁止项相同，见 [registration-states.md](registration-states.md)。摘录了基准或下载链接也不会把记录变成完整发布。下表只增加“变成完整记录时，这种任务还要有什么”。

共同的完整字段：身份、许可证、`sample_path`、每个规格的参数量与 GFLOPs、源输入与 Runtime 输入分开记录、每个已发布板卡的制品、OE 报告、Runtime 性能和 C++ 端到端。精度列是额外要求。

| 代码 | 筛选项 | 完整时的精度 | 备注 |
| --- | --- | --- | --- |
| `cls` | 图像分类 | `top1`、`top5` | 浮点和 Runtime 都要有。数据集按实际评测集填写，不要把 ImageNetV2 写成 ILSVRC |
| `detect` | 目标检测 | `map_50_95`、`map_50`、`map_75` | 检测框指标 |
| `seg` | 实例分割 | `box_ap`、`mask_ap` | 掩码是分割任务的主指标 |
| `pose` | 姿态估计 | `keypoints_ap` | |
| `obb` | 旋转框检测 | `map_50` | `dataset` 为 `DOTA val`，`evaluation_scope` 为 `local_dota_val_single_scale`。筛选项 id 仍是 `object-detection`，卡片文案是“旋转框检测” |
| `semseg` | 语义分割 | 尚无字段表 | 像素类别任务。补表之前保持待补 |
| `prompt-seg` | 提示式分割 | 尚无字段表 | 提示（点或框）属于输入合同，完整记录要写清提示如何进入模型 |
| `depth` | 深度估计 | 尚无字段表 | 相对深度和绝对深度不能混写。没有评测协议就保持待补 |
| `track` | 目标跟踪 | 尚无字段表 | 跟踪器如果只消费检测框、自己没有 BPU 模型，完整记录要写清输入是检测结果而不是图像 |
| `embed` | 视觉特征提取 | 尚无字段表 | 写清输出是全局嵌入、图像块特征，还是两者都有 |
| `ocr-det` | 文字检测 | 尚无字段表 | 与 `ocr-rec` 分开建档 |
| `ocr-rec` | 文字识别 | 尚无字段表 | 词表和 CTC 解码属于模型合同，不把检测指标抄到识别记录 |
| `similarity` | 图文匹配 | 尚无字段表 | 图像编码器和文本编码器都存在时才使用这个代码。只有视觉编码器时用 `embed`，并用 `facet_tasks` 加上图文筛选项 |
| `ov-detect` | 开放词汇检测 | 尚无字段表 | 词表或文本嵌入是输入的一部分，不要写成普通类别文件 |
| `lpr` | 车牌识别 | 尚无字段表 | |
| `matting` | 人像抠图 | 尚无字段表 | |
| `video-cls` | 视频动作分类 | 尚无字段表 | 输入通常是片段而不是单张图。完整记录要改输入合同，不能硬套四维静态图校验 |
| `avatar` | 语音驱动数字人 | 尚无字段表 | 多段模型组成一条流水线时，完整记录逐段列出，不合并成一个虚构文件 |
| `text-gen` | 文本生成 | 尚无字段表 | 现有完整校验假定视觉四维输入。文本模型在扩展校验器之前保持待补 |
| `vlm` | 视觉语言模型 | 尚无字段表 | 同时有文本和图像入口。不要只填视觉输入 |
| `manip` | 机械臂策略 | 尚无字段表 | 子模块示例用 `sample_kind: gitlink`。策略权重、归一化统计和机器人标定未随仓库展开时，不要编造下载地址 |
| `loco` | 足式运动控制 | 尚无字段表 | 观测向量和动作维度属于输入合同 |
| `drive` | 自动驾驶 | 尚无字段表 | 写清相机、激光雷达和自车状态里哪些真的进入模型 |
| `lane` | 车道线检测 | 尚无字段表 | 分割分支和嵌入分支都要在完整记录里出现 |
| `point-seg` | 点云分割 | 尚无字段表 | 输入不是图像。扩展校验器之前保持待补 |
| `asr` | 语音识别 | 尚无字段表 | 采样率、分块和词表属于合同，不是可省略的说明 |
| `kws` | 关键词唤醒 | 尚无字段表 | 写明关键词和决策方式。不要把一次样例音频的置信度当成正式精度 |

## 待补页的精度瓦片

已发布任务（`cls`、`detect`、`seg`、`pose`、`obb`）的待补页使用该任务已发布页的精度瓦片。下面这些任务还没有已发布对照，待补页在 Runtime 延迟、端到端延迟、吞吐量之后固定再放两格，名称如下。没有原文时数值是“数据待补充”，不要写成笼统的“精度”，也不要编造数字。

| 代码 | 两格精度名 |
| --- | --- |
| `semseg` | mIoU、像素准确率 |
| `prompt-seg` | mIoU、Dice |
| `depth` | AbsRel、δ1 |
| `track` | MOTA、IDF1 |
| `embed` | k-NN Top-1、线性探测 Top-1 |
| `ocr-det` | 检测精确率、检测召回率 |
| `ocr-rec` | 识别准确率、字符错误率 |
| `similarity` | CLIP zero-shot Top-1、图文检索 R@1 |
| `ov-detect` | 开放词汇 mAP50–95、开放词汇 mAP50 |
| `lpr` | 车牌准确率、字符准确率 |
| `matting` | SAD、MSE |
| `video-cls` | Top-1 准确率、Top-5 准确率 |
| `avatar` | LSE-D、FID |
| `text-gen` | 困惑度、生成吞吐 |
| `vlm` | VQA 准确率、图文检索 R@1 |
| `manip` | 任务成功率、轨迹误差 |
| `loco` | 速度跟踪误差、存活时间 |
| `drive` | 规划误差、碰撞率 |
| `lane` | 车道线 F1、车道线准确率 |
| `point-seg` | 点云 mIoU、点云准确率 |
| `asr` | 词错误率 WER、字错误率 CER |
| `kws` | 唤醒准确率、误唤醒率 |

## 现有完整示例

这些记录已经完整，登记其他模型时不要改它们，也不要再建同 `id` 的待补文件：

| 示例 | 网页记录 |
| --- | --- |
| `samples/vision/mobilenetv1` | `vision/mobilenetv1/mobilenetv1/cls.yaml` |
| `samples/vision/mobilenetv2` | `vision/mobilenetv2/mobilenetv2/cls.yaml` |
| `samples/vision/mobilenetv3` | `vision/mobilenetv3/mobilenetv3/cls.yaml` |
| `samples/vision/mobilenetv4` | `vision/mobilenetv4/mobilenetv4/cls.yaml` |
| `samples/vision/resnet` | `vision/resnet/resnet/cls.yaml` |
| `samples/vision/ultralytics_yolo` | `vision/ultralytics_yolo/yolov8/{cls,detect,pose,seg}.yaml`、`yolo11/{cls,detect,pose,seg}.yaml`、`yolo26/{cls,detect,obb,pose,seg}.yaml` |

`samples/vla/guides` 是 ACT 和 Pi0 的说明，不是模型目录，不建记录。
