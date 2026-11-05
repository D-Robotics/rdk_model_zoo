# 模型广度：剩余阻塞项（2026-11-05）

本批优先补齐此前没有新目录下载的模型条目，先交付 S100。ViT ImageNet-1K、FCOS ResNet50-FPN 和 UNet VGG16/VOC21 是独立规格，旧的 ViT CIFAR-10、FCOS EfficientNet 和 UNet ResNet50 规格仍保留原状态。

Wav2Vec2 ASR 与 SigLIP 的导出阻塞已解决。两者按负责人批准的独立 FP32 源合同核验全部原始输出，并完成 S100 编译、Native 样例与 Archive 校验；详见 `speech-siglip-conversion-acceptance-20261105.json`。Wav2Vec2 保留真实 attention mask 和全部3503类 logits；SigLIP 同时保留学习池化及196×768特征，因两次 INT16 诊断偏差较大，发布独立 FP16 内部计算产物。两个条目的独立封面来自真实主机模型输出，均未声称板端或数据集精度通过。

以下11个条目需要先解决源模型、导出或数据问题，再进行芯片扩展。不能用随机权重、其他任务模型或缺失输出的图替代。

| 条目 | 当前缺口 |
| --- | --- |
| ACT | 匹配的已训练策略权重、机械臂自由度/相机定义、归一化统计和成对回放数据。 |
| Pi0 | 匹配策略、机器人动作/观测定义、归一化统计和成对输入数据。 |
| HIMLoco | 匹配 Go2 的已训练权重及真实观测/动作回放。 |
| DiffusionDrive | 匹配四输入接口的 checkpoint、可核验导出及成对 NAVSIM 校准记录。 |
| MDTC KWS | 已有训练权重；缺 HeySnips 真实正、负类校准音频，导出接口仍需验证。 |
| PointNet | 匹配 chair 四部件分割的训练权重和真实点云；分类权重不能替代。 |
| UNetMobileNet | 匹配 Cityscapes 19 类、1024×2048 的原架构和训练权重。 |
| VargConvNet | 匹配的原架构与已训练权重。 |
| LiteAvatar | 完整四阶段权重、配套输入及可核验的分发条款。 |
| YOLO-World | 完整图像和 32×512 文本嵌入接口、训练权重与成对校准输入；不能用 YOLOE 或删去文本输入替代。 |
| HobotMot | CPU 关联组件不应伪造独立神经网络模型下载；需要以配套检测器组合或源码样例交付。 |

本批数据均先用于转换与主机检查：ViT/FCOS/UNet 使用冻结的 COCO train200；LaneNet 使用 100 张 TuSimple 训练片段真实末帧；3D ResNet 使用公开镜像中 8 个动作类别的 100 段真实 UCF101 训练视频，属于试验性跨数据集 PTQ。Faster R-CNN 跟踪另有 30 帧真实有序视频的主机流水线检查。

这些检查不能替代板端精度与性能测量。板端 smoke、精度、Runtime 1/2 和 C++ 1/2 均保持 not-run。S100P/S600 后续需要独立 nash-m/nash-p 转换、对应 ABI、资产登记及验收，不能直接复用 S100 HBM。
