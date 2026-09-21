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
6. **S 侧 cls 文件名真伪未裁定**（B9 收编 ultralytics 家族时裁定）：S 快照的
   `tests/test_yolo_cls_resolution.py` 断言 cls 制品名为 `*_cls_<march>_224x224_nv12.hbm`
   （且 `model_url` 亦返回 224 URL）；s tip（380e1a2）删除了该测试，sample 代码与
   `model/download_model.sh` 改为构造 **640x640** 文件名与 URL，清单 filename 键为
   640、但清单 URL 仍为 224——即 tip 处于 224→640 改名中途（文件名已改、URL 未跟）。
   develop 目录侧 errata（`applySCatalogErrata`）按 URL 证据把展示名归一为 224 并注明
   "Legacy 640 URLs remain compatible"。三方（catalog 展示 224 / 清单字节 640 键 /
   sample 按 manifest 键解析）各自自洽、互不改写；真名需网络实测裁定（用户门禁），
   裁定前 B9 不得合并或改写任何一侧。
7. **platforms/s 快照落后 s tip**（A7 勘定，2026-09-21）：快照相对 380e1a2 缺 53 个
   文件——`samples/vision/yoloe26_seg`（30）、`samples/llm/minicpm5-2b`（13，
   legacy evaluator + results + test_data）、`samples/vla/{act,pi0}` gitlink 与
   `.gitmodules`、`docs/manifests/*`（清单在 tip 已由 docs/release 迁至 docs/manifests；
   develop 侧 A4/A5 已按 tip 内容落位 `docs/release/s` 并补齐 tip 新增资产 382=368+14）、
   空的 `docs/tros/README.md` 与占位 `skills/README.md`（A6 已裁定不迁）。全部去向：
   B9（yoloe26_seg）、B11（minicpm5、vla gitlink+.gitmodules 路径改指 samples/vla），
   本表各行 source SHA 一律为 tip。快照多出的文件（旧 docs/release 清单、历史 release
   notes、`__pycache__`、`.gitattributes`、`tests/test_yolo_cls_resolution.py`、根
   `tros/`）均为 tip 已取代或废弃内容，不构成迁移义务。platforms/x5 快照对
   ac11571 的 sample 目录**零漂移**（差异仅 docs/catalog、skills、workflows 等
   ADR-0001/A3 范围设施）。

---

## 本轮 B1–B11 迁移进度区（2026-09-20 起）

本区独立记录本轮迁移进度，供人工审阅和 Q3 检查器读取。上方历史 P0 表的 `S/F/H` 是清点口径（S=源码/清单静态核对，F=旧新函数映射未核定，H=板端验证未执行），不是顺序状态机；历史列不因批次工作翻转，不得把 F 再解释为 mapping verified，也不得用历史 P0 状态推断新代码已验收。Q3 检查器纳入范围按本区 Refactor 列（in-progress/done 均纳入）确定。

- Mapping / Refactor / Docs：`pending / in-progress / done / not-applicable`。
- Host / Board / Review：`not-run / passed / failed / not-applicable`。
- Closed：默认 `no`，迁移代码存在不等于 closed；仅当全部 required 维度完成/通过且无未解决阻断项时为 `yes`。
- `not-applicable` 必须引用适用范围和理由，不能替代 required 检查；可选未运行项仍披露，不扩大支持或实测声明。
- 基础设施交付行（工具/契约类交付，无对应 `samples/` 目录，如 H5 Profile 契约、H6 覆盖检查）在 Sample 列以 `~` 开头；Q3 检查器的迁移范围解析跳过这些行，不当作 sample 解析。sample 行不得使用该标记规避解析。
- Evidence 绑定实际代码 SHA、源 SHA、目标、制品与输入身份、命令/cwd、结果和评审来源；阶段没有执行时不生成虚构成功记录。
- 一个 sample 的目标、变体或语言完成状态不同必须拆行；下表 pending 行先按 sample 粒度登记，批次执行时按 target×variant×语言细化拆分。

| Batch | Sample / source SHA | Target / variant / language | Mapping | Refactor | Docs | Host | Board | Review | Closed | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | ultralytics_yolo（yolov8n/yolo26n 检测）/ develop@9f17f2a | x5+s / yolov8n、yolo26n / python | done | done | pending（Q1–Q5 合规随 B9 收编验收） | passed | passed（x5 双板+s100+s100p 适用项；S600 not-run） | not-run | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md) |
| pilot | resnet（resnet18）/ develop@9f17f2a | x5+s100 / resnet18 / python + s100 cpp | done | done | done（2026-09-21 Q4 参照改造，检查器 0 violations） | passed | passed（x5 双板+s100；s100p 无已批准资产，未验证） | passed（2026-09-21 双视角评审） | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md)、[2026-09-21-phase05-q4-review.md](2026-09-21-phase05-q4-review.md) |
| pilot | paddle_ocr（PP-OCRv3/v6）/ develop@9f17f2a | x5+s100 / det+rec 两阶段 / python + s100 cpp | done | done | done（2026-09-21 Q4 参照改造，检查器 0 violations） | passed | passed（x5 双板+s100；s100p/s600 not-run） | passed（2026-09-21 双视角评审） | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md)、[2026-09-21-phase05-q4-review.md](2026-09-21-phase05-q4-review.md) |
| B1 | mobilenetv1 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（x5-8g 复测过；与 x5-4g/s100/s600 对照全等，v1 maxdiff 0.0） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv1 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（对照全等，maxdiff ≤1.2e-7） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv1 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（SSH 恢复后复测；对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv2 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK；+7 为 R2 整改的 cpp 启动器身份 fixture） | passed（x5-8g 复测过；四板 python 对照全等，v2 maxdiff 0.0） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv2 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK） | passed（对照全等，maxdiff ≤1.2e-7） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK） | passed（SSH 恢复后复测；对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s100 / cpp | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（身份 gate fixture 7 项随 python 套件执行（R2 整改）） | passed（R2 整改后板端复测（2026-09-21）：gate 放行、增量构建、TOP-1 zebra prob=9.30961 复现基线、rc=0；见 board evidence `r2_launcher_board_recheck` 节） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s600 / cpp | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（身份 gate fixture 7 项随 python 套件执行（R2 整改）） | not-run（不在 B1 冒烟集，不扩大声明） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv3 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（x5-8g 复测过；四板对照全等，v3 ≤5.96e-8） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv3 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（对照全等，maxdiff ≤1.2e-7） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv3 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（SSH 恢复后复测；对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv4 / x5:ac11571 | x5 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK；+3 为 R4 整改的转换 shape 一致性测试） | passed（x5-8g 复测过；四板对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv4 / x5:ac11571 | x5 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（x5-8g 复测过；按发布 224 几何运行） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s100 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s100 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（按发布 256 几何运行；对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s600 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（SSH 恢复后复测） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s600 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（medium 下载缺陷板测发现→修复→复验（B1-D2）） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | resnet（resnet50 变体）/ s:380e1a2 | s100+s600 / python | done | done（R1 整改：转换配方逐字节迁入并 SHA 固定） | done（R5 证据/文档同步整改后，待独立复核） | passed（套件 52 OK；+6 为 R1 整改的转换 layout/provenance 测试） | passed（s100+s600 对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | resnet（resnet152 变体）/ s:380e1a2 | s100+s600 / python | done | done（R1 整改：转换配方逐字节迁入并 SHA 固定） | done（R5 证据/文档同步整改后，待独立复核） | passed（套件 52 OK） | passed（s100+s600 对照全等） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（待独立评审复核关闭 findings） | 同上 |
| B1 | ~ s100p 负例（B1 全部 sample 共享身份/资产拒绝语义，`_shared/platforms.py`+manifest；R2 后含 v2 cpp launcher）/ develop | s100p / any / python | not-applicable（无 s100p 制品，负例即全部可验证面） | not-applicable | done | not-applicable | passed（python 负例 2/2：target mismatch 与 no-published-asset 均显式报错 rc=2，无回退；R2 整改后 cpp launcher 板端复测 4/4：真实 `S100P`、`s100`+`s100p`、`s100`+`RDK S100P` 三种身份形态显式拒绝 rc=2，对照例 gate 放行停于模型准备提示——s100p 实板 2026-09-21） | failed（dd60911 独立复审：R2–R6 closed；R1a/R1b 文档待修正） | no（随批） | [board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json) |
| B1 | ~ H5 平台 Profile 契约 / develop | `_shared/platform_profile.py`（纯主机契约） | done | done | done | passed（11 OK） | not-applicable（主机契约，无板端面） | passed（Codex 本轮范围核对；批级门槛仍未过） | no（随批：独立评审整改未闭环） | 同上 evidence；[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | ~ H6 sample↔manifest 覆盖检查 / develop | `_shared/tests/test_manifest_coverage.py` | done | done | done | passed（4 OK；7 个统一样本全覆盖） | not-applicable（静态清单核对） | passed（Codex 本轮范围核对；批级门槛仍未过） | no（随批：独立评审整改未闭环） | 同上 evidence；[独立评审](2026-09-21-b1-independent-review.md) |
| B2 | efficientnet / x5:ac11571 + s:380e1a2 | x5+s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B2 | efficientformer / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B2 | efficientformerv2 / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B2 | efficientvit / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B3 | convnext / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B3 | edgenext / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B3 | fasternet / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B3 | fastvit / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | repghost / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | repvgg / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | repvit / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | mobileone / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | resnext / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | vargconvnet / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | googlenet / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B4 | hgnetv2 / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B5 | clip / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B5 | siglip / s:380e1a2 | s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B5 | dinov2 / s:380e1a2 | s / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B5 | vit / s:380e1a2 | s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B5 | 3dresnet / s:380e1a2 | s / python（视频输入） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B6 | efficient_sam / x5:ac11571 + s:380e1a2 | x5+s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B6 | mobile_sam / x5:ac11571 + s:380e1a2 | x5+s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | yolov5 / x5:ac11571 + s:380e1a2 | x5+s / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | fcos / x5:ac11571 | x5 / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | yoloworld / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | lprnet / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | modnet / x5:ac11571 | x5 / python（资产 manual） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B7 | bytetrack / s:380e1a2 | s / python（依赖 yolov5） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | unet / x5:ac11571 | x5 / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | unetmobilenet / s:380e1a2 | s / python + cpp（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | pp_liteseg / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | yolo26_depth / x5:ac11571 + s:380e1a2 | x5+s / python + x5 cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | depth_anything_v2 / s:380e1a2 | s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | lanenet / s:380e1a2 | s / python + cpp（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | pointnet / s:380e1a2 | s / python（点云输入） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | diffusiondrive / s:380e1a2 | s / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B9 | ultralytics_yolo 收编 yolo11/yolo11_pose/yolo11_seg/yolov13_imoonlab / s:380e1a2 | s / 新 family/variant / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | —（B9 必须删除 B1-R6 欠账基线 `tools/sample_contract/baselines/ultralytics-readme-debt.json` 及 workflow 中的 `--exemptions` 旗标，见 checker README） |
| B9 | yoloe（x5 yoloe + s yoloe11_seg + s tip yoloe26_seg）/ x5:ac11571 + s:380e1a2 | x5+s / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B10 | himloco / x5:ac11571 | x5 / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B10 | asr / s:380e1a2 | s / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B10 | kws / s:380e1a2 | s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B10 | paraformer / s:380e1a2 | s / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B11 | gemma4-e2b / s:380e1a2 | s / cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B11 | minicpm5-2b / s:380e1a2（快照无，tip 新增） | s / cpp + evaluator（legacy 脚本 + results/*.json）+ test_data | pending | pending | pending | not-run | not-run | not-run | no | — |
| B11 | vla/act / s:380e1a2（gitlink 326ea043） | s / gitlink | pending | pending | pending | not-run | not-run | not-run | no | — |
| B11 | vla/pi0 / s:380e1a2（gitlink a32de276） | s / gitlink | pending | pending | pending | not-run | not-run | not-run | no | — |

勘误（相对批次 kickoff 表）：depth_anything_v2、lanenet、pointnet 为 S-only 源（kickoff 表 B8 行未单独标注）；B2 的 efficientformer 与 efficientformerv2 为两个独立源 sample；S 侧 yolov13 实际目录为 yolov13_imoonlab；yoloe26_seg、minicpm5-2b、vla gitlink 仅存在于 s tip（快照缺失，见未核定项 7）；B9 收编 ultralytics 家族时先裁定未核定项 6 的 cls 文件名真伪。

### B1 独立审阅覆盖说明（2026-09-21）

以 [独立报告](2026-09-21-b1-independent-review.md) 的矩阵区分板位、语言及正/负例。上方旧聚合 Board 文本仅保留历史证据摘要，不等同整行通过；R5 修正须拆分不同状态的目标/变体/语言，才能用于机器闭环。当前所有 B1 Closed=no。

### B1 整改独立复审更新（dd60911）

> **最新独立复审（Codex，HEAD dd60911）：changes-required / not-ready，Closed=no。** R2–R6 已关闭；R1 文件保留已通过，双语转换 README 尚有执行目录错误与 scale 一致性错误。见 [整改独立复审](2026-09-21-b1-independent-rereview.md)。历史内容和原始结论保留。

331 项主机测试通过；带限界基线的 migration checker 通过。板测门槛已满足既定范围，当前阻断仅为 ResNet152 双语转换 README 的 R1a/R1b。所有 B1 Closed=no；B2 pending。
