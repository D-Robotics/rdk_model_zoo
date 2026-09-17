# X5 / S P0 人工迁移表

日期：2026-09-16  
源码锚点：cd74a2b241075bb21036d8d0855d0403f8e8c963  
对应基线：[2026-09-16-baseline.md](2026-09-16-baseline.md)

这是一份基于源码和发布清单的人工清点表，不是已完成迁移清单。候选目标路径沿用旧规划表
docs/superpowers/specs/2026-09-16-unified-model-zoo-migration-map.md，只作为人工核对起点。
P0 没有填写旧函数到新函数的映射，也没有把同名目录标成算法等价；后续必须按真实符号、
输入输出张量、模型制品和结果证据补齐。

后续证据按批次追加，表内 P0 状态保持原始清点口径：

| 批次 / 范围 | 维护位置与符号映射 | 当前证据边界 |
| --- | --- | --- |
| P1 ResNet18 | `samples/vision/resnet/`；该 Sample README 的绑定、Runner、任务映射 | [P1 报告](2026-09-16-pilot-validation.md)：X5 8GB/4GB、S100、S600 固定输入旧新对照通过；不包含 ResNet50/152 或 S100P |
| P1 YOLO DFL 检测 | `samples/vision/ultralytics_yolo/DETECTION_CONTRACT.md` 的旧新符号表 | 同上：YOLOv8n 五板对照通过；不代表所有模型尺寸、其他任务或数据集精度 |
| P2 YOLO26 LTRB | 同一 Sample 的协议、绑定、Runner 与任务模块；独立审查闭环 | [P2 报告](2026-09-16-p2-validation.md)：95 项主机测试和五板固定输入对照通过；不扩大到其他 YOLO26 任务 |
| P2 OCR 组合 | [源码审计](p2-ocr-source-audit.md) 的函数映射与差异表 | [P2 报告](2026-09-16-p2-validation.md)：31 项 OCR 主机测试，X5 两板和 S100 两种长宽比逐阶段对照通过；C++、转换、评估保留原入口 |
| P2 分割 / 姿态 | 现有 `yolo_seg.py` / `yolo_pose.py`，后续显式绑定改造 | YOLOv8n 五板原版基线已采集；新契约尚未实施 |

其余条目仍需逐项迁移与验收；P1/P2 试点不改变本表记录的完整能力范围。

## 表内口径

* 入口：Py=runtime/python/main.py，sh=runtime/python/run.sh；C++ 后列出真实 main 文件。
  两个 LLM 样例的入口在 runtime/cpp，ACT/PI0 入口在未初始化的 gitlink 中，不能从本仓
  的空目录判断。
* Conv / Eval：HEAD git tree 中 conversion / evaluator 下除 README 外的文件数，只表示
  能力文件存在，不代表转换或评估已经执行。
* A：Manifest assets 行数和格式摘要；这些是发布引用，不是工作树中的模型字节。
  B：对应 benchmarks.yaml 的原始行数。S 的发布器应用 errata 后可能产生派生行，基线文件另记。
* N：第一方 Notebook 数；X5/S 递归扫描和 HEAD 均为 0。
* 状态 S/F/H：源码和清单静态核对 / 旧新函数映射未核定 / 板端验证未执行。候选目标不等于
  迁移完成。M-：源码存在但缺少当前平台 Manifest 行。

## X5（37 个源 Sample，37 个 Manifest 行）

| 当前源路径 | Manifest ID / 能力 | 入口与配套文件（HEAD） | A / B | 候选维护路径 | P0 状态 |
| --- | --- | --- | ---: | --- | --- |
| platforms/x5/samples/robotics/himloco | himloco / legged-locomotion-control | Py+sh；C++ src/main.cpp；Conv 3；Eval 4 | 1 bin / 5 | samples/robotics/himloco | S/F/H |
| platforms/x5/samples/vision/clip | clip / image-text-similarity | Py+sh；C++ —；Conv 0；Eval 0 | 2 bin+onnx / 0 | samples/vision/clip | S/F/H |
| platforms/x5/samples/vision/convnext | convnext / image-classification | Py+sh；C++ —；Conv 3；Eval 0 | 1 bin / 4 | samples/vision/convnext | S/F/H |
| platforms/x5/samples/vision/edgenext | edgenext / image-classification | Py+sh；C++ —；Conv 4；Eval 0 | 4 bin / 8 | samples/vision/edgenext | S/F/H |
| platforms/x5/samples/vision/efficient_sam | efficient_sam / promptable-image-segmentation | Py+sh；C++ —；Conv 10；Eval 0 | 2 bin / 6 | samples/vision/efficient_sam | S/F/H |
| platforms/x5/samples/vision/efficientformer | efficientformer / image-classification | Py+sh；C++ —；Conv 2；Eval 0 | 2 bin / 4 | samples/vision/efficientformer | S/F/H |
| platforms/x5/samples/vision/efficientformerv2 | efficientformerv2 / image-classification | Py+sh；C++ —；Conv 3；Eval 0 | 3 bin / 6 | samples/vision/efficientformerv2 | S/F/H |
| platforms/x5/samples/vision/efficientnet | efficientnet / image-classification | Py+sh；C++ —；Conv 3；Eval 0 | 3 bin / 6 | samples/vision/efficientnet | S/F/H |
| platforms/x5/samples/vision/efficientvit | efficientvit / image-classification | Py+sh；C++ —；Conv 1；Eval 0 | 1 bin / 2 | samples/vision/efficientvit | S/F/H |
| platforms/x5/samples/vision/fasternet | fasternet / image-classification | Py+sh；C++ —；Conv 4；Eval 0 | 4 bin / 8 | samples/vision/fasternet | S/F/H |
| platforms/x5/samples/vision/fastvit | fastvit / image-classification | Py+sh；C++ —；Conv 4；Eval 0 | 4 bin / 8 | samples/vision/fastvit | S/F/H |
| platforms/x5/samples/vision/fcos | fcos / object-detection | Py+sh；C++ —；Conv 3；Eval 0 | 3 bin / 9 | samples/vision/fcos | S/F/H |
| platforms/x5/samples/vision/googlenet | googlenet / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 2 | samples/vision/googlenet | S/F/H |
| platforms/x5/samples/vision/hgnetv2 | hgnetv2 / image-classification | Py+sh；C++ —；Conv 10；Eval 1 | 5 bin / 5 | samples/vision/hgnetv2 | S/F/H |
| platforms/x5/samples/vision/lprnet | lprnet / license-plate-recognition | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 2 | samples/vision/lprnet | S/F/H |
| platforms/x5/samples/vision/mobile_sam | mobile_sam / promptable-image-segmentation | Py+sh；C++ —；Conv 8；Eval 0 | 2 bin / 6 | samples/vision/mobile_sam | S/F/H |
| platforms/x5/samples/vision/mobilenetv1 | mobilenetv1 / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 1 | samples/vision/mobilenetv1 | S/F/H |
| platforms/x5/samples/vision/mobilenetv2 | mobilenetv2 / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 1 | samples/vision/mobilenetv2 | S/F/H |
| platforms/x5/samples/vision/mobilenetv3 | mobilenetv3 / image-classification | Py+sh；C++ —；Conv 1；Eval 0 | 1 bin / 1 | samples/vision/mobilenetv3 | S/F/H |
| platforms/x5/samples/vision/mobilenetv4 | mobilenetv4 / image-classification | Py+sh；C++ —；Conv 2；Eval 0 | 2 bin / 2 | samples/vision/mobilenetv4 | S/F/H |
| platforms/x5/samples/vision/mobileone | mobileone / image-classification | Py+sh；C++ —；Conv 5；Eval 0 | 5 bin / 10 | samples/vision/mobileone | S/F/H |
| platforms/x5/samples/vision/modnet | modnet / portrait-matting | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin (manual) / 2 | samples/vision/modnet | S/F/H |
| platforms/x5/samples/vision/paddleocr | paddleocr / ocr-text-detection, ocr-text-recognition | Py+sh；C++ —；Conv 2；Eval 0 | 2 bin / 4 | samples/vision/paddleocr | S/F/H |
| platforms/x5/samples/vision/pp_liteseg | pp_liteseg / semantic-segmentation | Py+sh；C++ —；Conv 5；Eval 1 | 1 bin / 1 | samples/vision/pp_liteseg | S/F/H |
| platforms/x5/samples/vision/repghost | repghost / image-classification | Py+sh；C++ —；Conv 5；Eval 0 | 5 bin / 10 | samples/vision/repghost | S/F/H |
| platforms/x5/samples/vision/repvgg | repvgg / image-classification | Py+sh；C++ —；Conv 6；Eval 0 | 6 bin / 12 | samples/vision/repvgg | S/F/H |
| platforms/x5/samples/vision/repvit | repvit / image-classification | Py+sh；C++ —；Conv 3；Eval 0 | 3 bin / 6 | samples/vision/repvit | S/F/H |
| platforms/x5/samples/vision/resnet | resnet / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 1 | samples/vision/resnet | S/F/H |
| platforms/x5/samples/vision/resnext | resnext / image-classification | Py+sh；C++ —；Conv 1；Eval 0 | 1 bin / 2 | samples/vision/resnext | S/F/H |
| platforms/x5/samples/vision/ultralytics_yolo | ultralytics_yolo / object-detection, instance-segmentation, pose-estimation, image-classification | Py+sh；C++ classify/detect/pose/segment main.cc；Conv 8；Eval 5 | 67 bin / 67 | samples/vision/ultralytics_yolo | S/F/H；已有统一入口 |
| platforms/x5/samples/vision/ultralytics_yolo26 | ultralytics_yolo26 / object-detection, instance-segmentation, pose-estimation, oriented-bounding-box-detection, image-classification | Py+sh；C++ —；Conv 6；Eval 5 | 25 bin / 8 | samples/vision/ultralytics_yolo | S/F/H；已有 family=yolo26 wrapper |
| platforms/x5/samples/vision/unet | unet / semantic-segmentation | Py+sh；C++ —；Conv 10；Eval 1 | 5 bin / 8 | samples/vision/unet | S/F/H |
| platforms/x5/samples/vision/vargconvnet | vargconvnet / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 0 | samples/vision/vargconvnet | S/F/H |
| platforms/x5/samples/vision/yolo26_depth | yolo26_depth / monocular-depth-estimation | Py+sh；C++ src/main.cpp；Conv 10；Eval 3 | 5 bin / 10 | samples/vision/yolo26_depth | S/F/H；独立深度流程 |
| platforms/x5/samples/vision/yoloe | yoloe / instance-segmentation | Py+sh；C++ —；Conv 2；Eval 0 | 3 bin / 3 | samples/vision/yoloe | S/F/H |
| platforms/x5/samples/vision/yolov5 | yolov5 / object-detection | Py+sh；C++ main.cc；Conv 2；Eval 0 | 1 bin / 9 | samples/vision/yolov5 | S/F/H |
| platforms/x5/samples/vision/yoloworld | yoloworld / open-vocabulary-object-detection | Py+sh；C++ —；Conv 0；Eval 0 | 1 bin / 0 | samples/vision/yoloworld | S/F/H |

## S（37 个源目录，35 个 Manifest 行）

| 当前源路径 | Manifest ID / 能力 | 入口与配套文件（HEAD） | A / B | 候选维护路径 | P0 状态 |
| --- | --- | --- | ---: | --- | --- |
| platforms/s/samples/llm/gemma4-e2b | gemma4-e2b / vision-language-model | C++ src/main.cpp；Conv 21；Eval 0 | 7 hbm4+bin1+json2 / 0 | samples/llm/gemma4-e2b | S/F/H |
| platforms/s/samples/llm/minicpm5-2b | Manifest 缺行；README 为 MiniCPM5-2B text generation | C++ cpp/src/main.cc + legacy/src/main.cc；Conv 13；Eval 5 | — / — | samples/llm/minicpm5-2b | M-/F/H；需外部 OELLM SDK |
| platforms/s/samples/speech/asr | asr / speech-recognition | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 1 | samples/speech/asr | S/F/H |
| platforms/s/samples/speech/kws | kws / keyword-spotting | Py+sh；C++ —；Conv 0；Eval 0 | 1 hbm / 1 | samples/speech/kws | S/F/H |
| platforms/s/samples/speech/paraformer | paraformer / speech-recognition | Py+sh；C++ src/main.cpp；Conv 15；Eval 0 | 6 hbm3+json+mvn+yaml / 1 | samples/speech/paraformer | S/F/H；三 HBM + CPU CIF |
| platforms/s/samples/vision/3dresnet | 3dresnet / video-action-classification | Py+sh；C++ —；Conv 0；Eval 0 | 1 hbm / 1 | samples/vision/3dresnet | S/F/H |
| platforms/s/samples/vision/bytetrack | bytetrack / multi-object-tracking | Py+sh；C++ —；Conv 0；Eval 0 | 3 hbm / 1 | samples/vision/bytetrack | S/F/H；依赖 YOLOv5 detector |
| platforms/s/samples/vision/depth_anything_v2 | depth_anything_v2 / monocular-depth-estimation | Py+sh；C++ —；Conv 0；Eval 0 | 1 hbm / 1 | samples/vision/depth_anything_v2 | S/F/H |
| platforms/s/samples/vision/diffusiondrive | diffusiondrive / autonomous-driving | Py+sh；C++ —；Conv 2；Eval 1 | 2 hbm / 2 | samples/vision/diffusiondrive | S/F/H |
| platforms/s/samples/vision/dinov2 | dinov2 / image-embedding | Py+sh；C++ —；Conv 2；Eval 0 | 3 hbm / 4 | samples/vision/dinov2 | S/F/H |
| platforms/s/samples/vision/efficient_sam | efficient_sam / promptable-image-segmentation | Py+sh；C++ —；Conv 12；Eval 0 | 6 hbm / 3 | samples/vision/efficient_sam | S/F/H |
| platforms/s/samples/vision/efficientnet | efficientnet / image-classification | Py+sh；C++ —；Conv 13；Eval 0 | 10 hbm / 5 | samples/vision/efficientnet | S/F/H |
| platforms/s/samples/vision/lanenet | lanenet / lane-detection | Py+sh；C++ src/main.cpp；Conv 1；Eval 0 | 1 hbm / 1 | samples/vision/lanenet | S/F/H |
| platforms/s/samples/vision/mobile_sam | mobile_sam / promptable-image-segmentation | Py+sh；C++ —；Conv 12；Eval 0 | 6 hbm / 3 | samples/vision/mobile_sam | S/F/H |
| platforms/s/samples/vision/mobilenetv1 | mobilenetv1 / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/mobilenetv1 | S/F/H |
| platforms/s/samples/vision/mobilenetv2 | mobilenetv2 / image-classification | Py+sh；C++ src/main.cpp；Conv 1；Eval 0 | 2 hbm / 1 | samples/vision/mobilenetv2 | S/F/H |
| platforms/s/samples/vision/mobilenetv3 | mobilenetv3 / image-classification | Py+sh；C++ —；Conv 4；Eval 0 | 2 hbm / 1 | samples/vision/mobilenetv3 | S/F/H |
| platforms/s/samples/vision/mobilenetv4 | mobilenetv4 / image-classification | Py+sh；C++ —；Conv 6；Eval 0 | 4 hbm / 2 | samples/vision/mobilenetv4 | S/F/H |
| platforms/s/samples/vision/paddle_ocr | paddle_ocr / text-detection, text-recognition | Py+sh；C++ src/main.cpp；Conv 2；Eval 0 | 2 hbm / 0 | samples/vision/paddleocr | S/F/H；组件待核 |
| platforms/s/samples/vision/pointnet | pointnet / point-cloud-segmentation | Py+sh；C++ —；Conv 0；Eval 0 | 1 hbm / 1 | samples/vision/pointnet | S/F/H |
| platforms/s/samples/vision/resnet152 | resnet152 / image-classification | Py+sh；C++ —；Conv 3；Eval 0 | 2 hbm / 1 | samples/vision/resnet | S/F/H；规格待保留 |
| platforms/s/samples/vision/resnet18 | resnet18 / image-classification | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/resnet | S/F/H；规格待保留 |
| platforms/s/samples/vision/resnet50 | resnet50 / image-classification | Py+sh；C++ —；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/resnet | S/F/H；规格待保留 |
| platforms/s/samples/vision/siglip | siglip / vision-embedding, image-text-similarity | Py+sh；C++ —；Conv 0；Eval 0 | 8 hbm / 16 | samples/vision/siglip | S/F/H；两个输出语义待保留 |
| platforms/s/samples/vision/ultralytics_yolo | ultralytics_yolo / object-detection, instance-segmentation, pose-estimation, image-classification | Py+sh；C++ —；Conv 6；Eval 4 | 186 hbm / 368 | samples/vision/ultralytics_yolo | S/F/H；已有统一入口 |
| platforms/s/samples/vision/ultralytics_yolo26 | ultralytics_yolo26 / object-detection, instance-segmentation, pose-estimation, oriented-bounding-box-detection, image-classification | Py+sh；C++ —；Conv 6；Eval 5 | 75 hbm / 95 | samples/vision/ultralytics_yolo | S/F/H；已有 family=yolo26 wrapper |
| platforms/s/samples/vision/unetmobilenet | unetmobilenet / semantic-segmentation | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/unetmobilenet | S/F/H；不替换 X5 unet |
| platforms/s/samples/vision/vit | vit / image-classification | Py+sh；C++ —；Conv 2；Eval 0 | 2 hbm / 1 | samples/vision/vit | S/F/H |
| platforms/s/samples/vision/yolo11 | yolo11 / object-detection | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/ultralytics_yolo | S/F/H；候选归并 |
| platforms/s/samples/vision/yolo11_pose | yolo11_pose / pose-estimation | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/ultralytics_yolo | S/F/H；候选归并 |
| platforms/s/samples/vision/yolo11_seg | yolo11_seg / instance-segmentation | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/ultralytics_yolo | S/F/H；候选归并 |
| platforms/s/samples/vision/yolo26_depth | yolo26_depth / monocular-depth-estimation | Py+sh；C++ —；Conv 29；Eval 2 | 15 hbm / 20 | samples/vision/yolo26_depth | S/F/H；独立深度流程 |
| platforms/s/samples/vision/yoloe11_seg | Manifest 缺行；README 为 4585 类开放词表实例分割 | Py+sh；C++ src/main.cpp；Conv 8；Eval 0 | — / — | samples/vision/yoloe | M-/F/H；仅 S100，S600 明确退出 |
| platforms/s/samples/vision/yolov13_imoonlab | yolov13_imoonlab / object-detection | Py+sh；C++ —；Conv 4；Eval 0 | 4 hbm / 8 | samples/vision/ultralytics_yolo | S/F/H；候选归并 |
| platforms/s/samples/vision/yolov5 | yolov5 / object-detection | Py+sh；C++ src/main.cpp；Conv 0；Eval 0 | 2 hbm / 0 | samples/vision/yolov5 | S/F/H |
| platforms/s/samples/vla/act | act / robot-manipulation-policy | 外部 gitlink；本仓无入口文件 | 0 / 0 | samples/vla/act | S/F/H；gitlink 326ea043be204de25223d95c7d918efe8672dc66 |
| platforms/s/samples/vla/pi0 | pi0 / robot-manipulation-policy | 外部 gitlink；本仓无入口文件 | 0 / 0 | samples/vla/pi0 | S/F/H；gitlink a32de276bc1681a2b1531012de111eaa1c16acb6 |

## 明确保留的未核定项

1. 逐函数映射：本表没有声称旧 X5 的某函数进入某个新函数，也没有声称 S 的同名实现可以替代 X5。
2. 平台支持：Manifest 的 asset 路径只说明发布引用。尤其 S 的 nash-e/nash-m/nash-p、旧 s100/s600 路径和
   README 中的限制需要按具体模型与实际板卡分别验证。
3. 制品身份：Manifest 中大多数资产没有 SHA-256，工作树也没有这些模型字节；A 列不等于可下载成功或运行通过。
4. 外部依赖：ACT/PI0 的上游目录未初始化；Gemma4/MiniCPM 的 LLM/OELLM SDK 与版本、许可证、运行边界需要单独取证。
5. 第一方 Notebook：X5/S 为 0；X3 的 20 个 Notebook 是历史资源，不能在本表中改动或算入 X5/S 迁移完成度。
