# 2026-10-08 Sample Runtime 接口调整

本轮以 `7af1d56f546d39d8a6019949fddeb87b09eeff3d` 参考工作快照为对照，
在 `codex/runtime-simplification-20261008` 整合 Runtime 和中英文 README。
适用 49 个 Python Runtime；两项 LLM 保留原生接口，ACT/Pi0 上游子仓库不变。
编写要求见 [Runtime 代码规范](../sample-standards/runtime-code.md)。
49 个 Runtime 直属 Python 文件从 315 个减少到 221 个，其中 32 个 Sample 使用三个 Python 文件；
这些计数不包含独立的跟踪器等子包和 `utils/` 公共库。

## 入口与模型构造

普通 Sample 的主要文件是 `main.py`、`cli.py` 和模型文件。
入口选择参数、构造具名模型、调用 `predict`，再交付结果。
模型通过构造方法或 `from_model` / `from_models` 建立 Runtime；
业务集成无需先创建 runner 或手动调用 `load`。

模型阶段、发布制品契约和各板卡默认值沿用原有语义；板卡选择仍在模型执行前验证。
`predict` 返回模型结果，展示和保存由 CLI 或调用者处理。
测试注入 runner 的低层接口保留在需要的模型中，参数名以各构造方法为准。

## 分类模型

ResNet 及其余 21 个分类 Sample 使用 `classify.py` 中的具名分类类。
原 Sample `model_binding.py` 的发布表、选择和列表能力移入 `cli.py`；
本地转发 runner 和空包文件不再需要。共享加载实现位于 `utils/py_utils/model_runner.py`。

```python
from samples.vision.vit.runtime.python.cli import resolve_selection
from samples.vision.vit.runtime.python.classify import ViTClassifier

selection = resolve_selection("s100", variant="int8")
contract = selection.contract
model = ViTClassifier(
    selection.model_path,
    target=selection.target,
    input_size=(contract.input_height, contract.input_width),
    class_count=contract.class_count,
    resize_type=contract.resize_type,
    resize_interpolation=contract.resize_interpolation,
    score_policy=contract.output_score_policy,
    output_transform=contract.output_transform,
)
result = model.predict("samples/vision/vit/test_data/airplane_0000.png")
```

实际输入文件与发布参数以各 Runtime README 的完整示例为准。自定义分类模型直接传
路径、目标板卡、输入尺寸、类别数和输出策略；无需注册发布清单。

## 具名模型 API

下表列出主要模型构造入口；`selection` / `pair` 由对应 Sample CLI 的选择函数生成。
可选参数和结果字段见同目录模型文件及中英文 README。

| Sample | 主要构造入口 |
| --- | --- |
| 3dresnet | `classification.R3D18Classifier(selection, ...)` |
| bytetrack | `tracking.ByteTrackTask.from_model(selection, ...)` |
| clip | `matching.CLIPMatcher(selection, ...)` |
| depth_anything_v2 | `depth_anything_v2.DepthEstimator(selection, ...)` |
| diffusiondrive | `diffusiondrive.DiffusionDrivePlanner(selection, ...)` |
| dinov2 | `embedding.DINOv2Embedder(selection, ...)` |
| efficient_sam | `pipeline.EfficientSAMPipeline.from_models(selection, ...)` |
| lanenet | `lanenet.LaneNetSegmenter(selection, ...)` |
| lprnet | `lprnet.LPRNetRecognizer(selection, ...)` |
| mobile_sam | `pipeline.MobileSAMPipeline.from_models(selection, ...)` |
| modnet | `modnet.MODNetMatting(selection, ...)` |
| paddle_ocr | `pipeline.OCRPipeline.from_models(pair, ...)` |
| pointnet | `pointnet.PointNetSegmenter(selection, ...)` |
| pp_liteseg | `pp_liteseg.PPLiteSegSegmenter(selection, ...)` |
| siglip | `embedding.SigLIPEmbedder(selection, ...)` |
| unet | `unet.UNetSegmenter(selection, ...)` |
| unetmobilenet | `unetmobilenet.UnetMobileNetSegmenter(selection, ...)` |
| yolov5 | `detection.YOLOv5Task(selection, ...)` |
| yoloworld | `yoloworld.YOLOWorldTask(selection, vocabulary, ...)` |
| fcos | `fcos.FCOSTask(selection, ...)` |
| yolo26_depth | `yolo26_depth.Yolo26DepthTask(selection, ...)` |
| yoloe | `yoloe.YOLOE(selection, config, ...)` |
| asr | `asr.ASR.from_model(selection, vocabulary, config, ...)` |
| kws | `kws.KWS.from_model(selection, config, ...)` |
| paraformer | `pipeline.ParaformerPipeline.from_models(selections, vocabulary, ...)` |
| himloco | `policy.HimLocoTask.from_model(selection, ...)` |

Ultralytics 的 `yolo_dispatch.prepare_runtime_model` 返回模型类型与配置，
入口显式执行 `Model(config)` 和 `model.predict`。原 `yolo_detect.py`、`yolo26_cls.py`
转发模块及 `legacy.py` 元组适配移除，分别使用实际任务模块 `detect.py`、`yolo_cls.py`
和结构化前处理结果。任务族选择仍使用既有 family / task / platform 规则。

## 导入位置与保留模块

- 已删除的选择模块使用对应 `cli.py`；模型绑定和张量准备使用对应模型文件。
- YOLOv5、YOLOWorld、FCOS、YOLOE、PaddleOCR 等保留承载实际复杂契约或资源管理的模块，
  不将所有 `model_binding` / `model_runner` 名称机械替换为 CLI。
- ASR/KWS 的输出处理归入模型文件；Paraformer 和 HIMLoco 的 CLI 应用辅助归入 `cli.py`。
- Paraformer 的原 `RuntimeBundle` / `load_runtime` 调用改用模型类 `from_models`。
  三模型的实际元数据校验及加载保留在 `runtime.load_model_runners`。
- UNet、UNetMobileNet、MODNet、PP-LiteSeg、ByteTrack、CLIP、EfficientSAM、MobileSAM
  的简单结果绘制归入 `cli.py`。UNetMobileNet 的 `PALETTE_BGR` 和 PP-LiteSeg 的
  `CITYSCAPES_PALETTE_BGR` 改为静态 tuple，颜色值不变；需要数组索引时使用
  `np.asarray(palette, dtype=np.uint8)`。
- 分词器、语音前端、CIF、跟踪器、复杂解码和编译/运行共同使用的图像准备保持独立职责。
- Shell 保持参数转发和退出码传播；Ultralytics 的位置式 task 参数继续可用。

README 使用当前接口直接介绍交付物；本文件记录变更，测试结果另记。
