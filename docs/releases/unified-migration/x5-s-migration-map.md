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
| pilot | ultralytics_yolo（yolov8n/yolo26n 检测）/ develop@9f17f2a | x5+s / yolov8n、yolo26n / python | done | done | done（2026-09-26 全层双语 README；原 84 条豁免清零；实现职责及 B9 收编另验收） | passed | passed（x5 双板+s100+s100p 适用项；S600 not-run） | not-run | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md) |
| pilot | resnet（resnet18）/ develop@9f17f2a | x5+s100 / resnet18 / python + s100 cpp | done | done | done（2026-09-21 Q4 参照改造，检查器 0 violations） | passed | passed（x5 双板+s100；s100p 无已批准资产，未验证） | passed（2026-09-21 双视角评审） | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md)、[2026-09-21-phase05-q4-review.md](2026-09-21-phase05-q4-review.md) |
| pilot | paddle_ocr（PP-OCRv3/v6）/ develop@9f17f2a | x5+s100 / det+rec 两阶段 / python + s100 cpp | done | done | done（2026-09-21 Q4 参照改造，检查器 0 violations） | passed | passed（x5 双板+s100；s100p/s600 not-run） | passed（2026-09-21 双视角评审） | no | [2026-09-17-integration-review.md](2026-09-17-integration-review.md)、[2026-09-21-phase05-q4-review.md](2026-09-21-phase05-q4-review.md) |
| B1 | mobilenetv1 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（x5-8g 复测过；与 x5-4g/s100/s600 对照全等，v1 maxdiff 0.0） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv1 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（对照全等，maxdiff ≤1.2e-7） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv1 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（SSH 恢复后复测；对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv2 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK；+7 为 R2 整改的 cpp 启动器身份 fixture） | passed（x5-8g 复测过；四板 python 对照全等，v2 maxdiff 0.0） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv2 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK） | passed（对照全等，maxdiff ≤1.2e-7） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（24 OK） | passed（SSH 恢复后复测；对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s100 / cpp | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（身份 gate fixture 7 项随 python 套件执行（R2 整改）） | passed（R2 整改后板端复测（2026-09-21）：gate 放行、增量构建、TOP-1 zebra prob=9.30961 复现基线、rc=0；见 board evidence `r2_launcher_board_recheck` 节） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv2 / s:380e1a2 | s600 / cpp | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（身份 gate fixture 7 项随 python 套件执行（R2 整改）） | not-run（不在 B1 冒烟集，不扩大声明） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv3 / x5:ac11571 | x5 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（x5-8g 复测过；四板对照全等，v3 ≤5.96e-8） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv3 / s:380e1a2 | s100 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（对照全等，maxdiff ≤1.2e-7） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv3 / s:380e1a2 | s600 / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（17 OK） | passed（SSH 恢复后复测；对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv4 / x5:ac11571 | x5 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK；+3 为 R4 整改的转换 shape 一致性测试） | passed（x5-8g 复测过；四板对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | mobilenetv4 / x5:ac11571 | x5 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（x5-8g 复测过；按发布 224 几何运行） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s100 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s100 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（按发布 256 几何运行；对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s600 / small / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（SSH 恢复后复测） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | mobilenetv4 / s:380e1a2 | s600 / medium / python | done | done | done（R5 证据/文档同步整改后，待独立复核） | passed（20 OK） | passed（medium 下载缺陷板测发现→修复→复验（B1-D2）） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | resnet（resnet50 变体）/ s:380e1a2 | s100+s600 / python | done | done（R1 整改：转换配方逐字节迁入并 SHA 固定） | done（R5 同步 + R1a/R1b 第二轮修正：校准命令 cwd 修复、scale 差异如实声明；待独立复核确认） | passed（套件 52 OK；+6 为 R1 整改的转换 layout/provenance 测试） | passed（s100+s600 对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [b1 评审](2026-09-21-b1-mobilenet-resnet-review.md)、[b1 evidence](evidence/2026-09-21-b1-mobilenet-resnet-evidence.json)、[board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)、[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | resnet（resnet152 变体）/ s:380e1a2 | s100+s600 / python | done | done（R1 整改：转换配方逐字节迁入并 SHA 固定） | done（R5 同步 + R1a/R1b 第二轮修正：校准命令 cwd 修复、scale 差异如实声明；待独立复核确认） | passed（套件 52 OK） | passed（s100+s600 对照全等） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 |
| B1 | ~ s100p 负例（B1 全部 sample 共享身份/资产拒绝语义，`_shared/platforms.py`+manifest；R2 后含 v2 cpp launcher）/ develop | s100p / any / python | not-applicable（无 s100p 制品，负例即全部可验证面） | not-applicable | done | not-applicable | passed（python 负例 2/2：target mismatch 与 no-published-asset 均显式报错 rc=2，无回退；R2 整改后 cpp launcher 板端复测 4/4：真实 `S100P`、`s100`+`s100p`、`s100`+`RDK S100P` 三种身份形态显式拒绝 rc=2，对照例 gate 放行停于模型准备提示——s100p 实板 2026-09-21） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | [board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json) |
| B1 | ~ H5 平台 Profile 契约 / develop | `_shared/platform_profile.py`（纯主机契约） | done | done | done | passed（11 OK） | not-applicable（主机契约，无板端面） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 evidence；[独立评审](2026-09-21-b1-independent-review.md) |
| B1 | ~ H6 sample↔manifest 覆盖检查 / develop | `_shared/tests/test_manifest_coverage.py` | done | done | done | passed（4 OK；7 个统一样本全覆盖） | not-applicable（静态清单核对） | passed（Codex 716bdca 最终复核；R1–R6 closed） | yes（既定范围；S600 cpp 仍 not-run） | 同上 evidence；[独立评审](2026-09-21-b1-independent-review.md) |
| B2 | efficientnet / x5:ac11571 + s:380e1a2 | x5 / B2、B3、B4 / python | done（提交 ac09101） | done | done（双语 5 级 README，检查器 0 violations） | passed（28 OK；+3 为 B2-R1 整改：省略变体逐 target 默认解析回归测试） | passed（x5-8g+x5-4g b2/b3/b4 同板 legacy 对照 ids 全等，maxdiff ≤1.19e-7；两板逐 case 数值一致；CLI b2 rc=0） | passed（Codex 21c833a；R1/R2/R3/R1-E closed，N1/N2 确认） | yes | [B2 评审](2026-09-21-b2-efficient-review.md)、[B2 evidence](evidence/2026-09-21-b2-efficient-evidence.json)、[board evidence](evidence/2026-09-21-b2-board-smoke-evidence.json) |
| B2 | efficientnet / x5:ac11571 + s:380e1a2 | s100+s600 / lite0–lite4 / python | done（提交 ac09101） | done（S 转换配方校准链核实；calibration 脚本遗留 src_image_dir 缺口如实披露） | done（双语 5 级 README，检查器 0 violations） | passed（28 OK；+3 为 B2-R1 整改回归测试） | passed（s100+s600 lite0–4 同板 legacy 对照全等 maxdiff ≤1.19e-7，逐变体几何 224/240/260/300/380 解析正确；CLI rc=0 双板；s100p 负例 3/3；R1 整改后板端复验省略变体默认入口：s100/s600 → lite0，完整 Top-5 与 lite0 记录逐 rank 一致，显式路径不变；B2-R1-E 完整输出+部署哈希记录见 [r1e-default-entry](evidence/2026-09-21-b2-r1e-default-entry/)） | passed（Codex 21c833a；R1/R2/R3/R1-E closed，N1/N2 确认） | yes | 同上 |
| B2 | efficientformer / x5:ac11571 | x5 / l1、l3（缺省 l3 保持源默认）/ python | done（提交 27fea1a） | done | done（双语 5 级 README，检查器 0 violations） | passed（25 OK） | passed（x5 双板 l1/l3 对照全等，maxdiff ≤2.38e-7；CLI l3 rc=0） | passed（Codex 21c833a；R1/R2/R3/R1-E closed，N1/N2 确认） | yes | 同上 |
| B2 | efficientformerv2 / x5:ac11571 | x5 / s0、s1、s2（缺省 s0 保持源默认）/ python | done（提交 a8a9ace） | done（output 前缀复现 Manifest 基名的正向锚定；s0 独有 debug/optimization 不对称如实保留） | done（双语 5 级 README，检查器 0 violations） | passed（26 OK） | passed（x5 双板对照；s0/s2 ids 全等，s1 两板各一例精确平局裁定 tie_resolved——794/851 各自实现内部分数相等（gap 0.0），跨实现差 1.4e-9 非逐字节相同，系 softmax 舍入+排序噪声非行为差异（独立评审裁定接受），完整 per-id 证据在 review-inputs 板端 record；CLI s0 rc=0） | passed（Codex 21c833a；R1/R2/R3/R1-E closed，N1/N2 确认） | yes | 同上 |
| B2 | efficientvit / x5:ac11571 | x5 / m5（缺省 m5 保持源默认）/ python | done（提交 4f08785） | done（`msra` 无变体输出前缀 vs Manifest `m5` 基名缺口钉住；0.99999 分位与 28 节点 Softmax int16 摆放锚定） | done（双语 5 级 README，检查器 0 violations） | passed（27 OK；+1 为复审 B2-N1 采纳后的无默认映射单资产拒绝边界测试） | passed（x5 双板 m5 对照全等，maxdiff ≤1.86e-9；CLI rc=0） | passed（Codex 21c833a；R1/R2/R3/R1-E closed，N1/N2 确认） | yes | 同上 |
| B3 | convnext / x5:ac11571 | x5 / atto / python | done（提交 e843005） | done | done（双语 5 级 README，检查器 0 violations） | passed（28 OK） | not-run（用户决定本轮跳过：网络环境切换、SSH 不可达；不作为通过声明，待后续板卡可达时补测） | passed（原 R1–R4 整改复核；208 主机 tests；见 2026-09-22-b3-host-recheck.md；板测待办） | no | [B3 评审](2026-09-21-b3-convnext-fastvit-review.md)、[B3 evidence](evidence/2026-09-21-b3-convnext-fastvit-evidence.json) |
| B3 | edgenext / x5:ac11571 | x5 / base、small、x_small、xx_small / python | done（提交 b4f8aff） | done（输出前缀正向锚定：前缀复现清单基名，无需重命名） | done（双语 5 级 README，检查器 0 violations） | passed（26 OK） | not-run（用户决定本轮跳过：网络环境切换、SSH 不可达；不作为通过声明，待后续板卡可达时补测） | passed（原 R1–R4 整改复核；208 主机 tests；见 2026-09-22-b3-host-recheck.md；板测待办） | no | 同上 |
| B3 | fasternet / x5:ac11571 | x5 / s、t0、t1、t2（表内小写 id；文件名保留大写）/ python | done（提交 69052d5） | done（无变体前缀需重命名 + working_dir 三种形态不对称，钉住披露） | done（双语 5 级 README，检查器 0 violations） | passed（28 OK；+1 为 B3-R1 整改的 parser→main→下载委托测试） | not-run（用户决定本轮跳过：网络环境切换、SSH 不可达；不作为通过声明，待后续板卡可达时补测） | passed（原 R1–R4 整改复核；208 主机 tests；见 2026-09-22-b3-host-recheck.md；板测待办） | no | 同上 |
| B3 | fastvit / x5:ac11571 | x5 / s12、sa12、t12、t8（表内小写 id）/ python | done（提交 f33167a） | done（onnx 全指外部 01_common 路径 + 无变体前缀，钉住披露） | done（双语 5 级 README，检查器 0 violations） | passed（28 OK；+1 为 B3-R1 整改的 parser→main→下载委托测试） | not-run（用户决定本轮跳过：网络环境切换、SSH 不可达；不作为通过声明，待后续板卡可达时补测） | passed（原 R1–R4 整改复核；208 主机 tests；见 2026-09-22-b3-host-recheck.md；板测待办） | no | 同上 |
| B4 | repghost / x5:ac11571 | x5 / 100、111、130、150、200 / python | done（源 SHA 清点） | done（统一任务/绑定/下载；见 B4 记录） | done（双语 5 级 README） | passed（8 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | repvgg / x5:ac11571 | x5 / a0/a1/a2/b0/b1g2/b1g4 / python | done（源清点与职责映射） | done（共享分类任务，独立制品契约） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | repvit / x5:ac11571 | x5 / m0_9/m1_0/m1_1 / python | done（源清点与职责映射） | done（共享分类任务，独立制品契约） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | mobileone / x5:ac11571 | x5 / s0/s1/s2/s3/s4 / python | done（源清点与职责映射） | done（共享分类任务，独立制品契约） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | resnext / x5:ac11571 | x5 / 50_32x4d / python | done（源清点与职责映射） | done（统一分类；HGNetV2 保留评测/导出能力） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | vargconvnet / x5:ac11571 | x5 / vargconvnet / python | done（源清点与职责映射） | done（统一分类；HGNetV2 保留评测/导出能力） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | googlenet / x5:ac11571 | x5 / googlenet / python | done（源清点与职责映射） | done（统一分类；HGNetV2 保留评测/导出能力） | done（双语 5 级 README） | passed（10 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B4 | hgnetv2 / x5:ac11571 | x5 / b0/b1/b2/b3/b4 / python | done（源清点与职责映射） | done（统一分类；HGNetV2 保留评测/导出能力） | done（双语 5 级 README） | passed（16 项；含每变体 predict/context 与源数值对照） | not-run（留待板端复验） | passed（本地主机独立评审；逐样例测试增强已由主任务复核） | no | [B4 记录](2026-09-22-b4-classification-review.md)、[独立评审](2026-09-22-b4-independent-host-review.md) |
| B5 | clip / x5:ac11571 | x5 / BPU image + CPU ONNX text / python | done（固定源与双编码器/BPE能力） | done（独立匹配任务；raw runner与cosine分离） | done（双语五级 README） | passed（15 项；真实BPE/源数值/CLI/API） | not-run（用户要求跳过板端环境） | passed（代码及十份README独立复核，3项P2关闭） | no | [B5 记录](2026-09-22-b5-vision-review.md) |
| B5 | siglip / s:380e1a2 | s100/s100p / 8 variants × 2 submodels / python | done（逐源清点；两板共用资产为源显式支持） | done（特征任务与 packed runner，保持 native dtype） | done（双语五级 README） | passed（19 项；源数值/CLI/API/链接） | not-run（用户要求跳过板端环境） | passed（独立完整复核；路径说明P2已确认关闭） | no | [B5 记录](2026-09-22-b5-vision-review.md)、[SigLIP 证据](evidence/2026-09-23-b5-siglip-host.json) |
| B5 | dinov2 / s:380e1a2 | s100/s100p/s600 / cls_feat、patch_feat / python | done（固定源与双输出量化契约） | done（复用H1反量化；独立特征任务） | done（双语五级 README） | passed（18 项；真实源数值/API/板测步骤fixture） | not-run（用户要求跳过板端环境） | passed（代码与十份README独立复核；2项P2关闭） | no | [B5 记录](2026-09-22-b5-vision-review.md) |
| B5 | vit / s:380e1a2 | s100 / int8、int16 / python | done（固定源与能力映射） | done（十分类复用共享核心） | done（双语五级 README） | passed（13 项；源 NV12/分数、README API） | not-run（用户要求跳过板端环境） | passed（独立本地主机复核；无阻断） | no | [B5 记录](2026-09-22-b5-vision-review.md) |
| B5 | 3dresnet / s:380e1a2 | s100 / r3d_18 / python（五维视频输入） | done（固定源/JSON标签/完整资源） | done（视频任务；共享Top-K；完整包导入） | done（双语五级 README） | passed（16 项；源视频/API/安全shell/精确tie负例） | not-run（用户要求跳过板端环境） | passed（完整独立主机复核；P1/P2已确认关闭） | no | [B5 记录](2026-09-22-b5-vision-review.md) |
| B6 | efficient_sam / x5:ac11571 + s:380e1a2 | x5/s100/s100p/s600 / python | done（固定源与逐文件哈希） | done（共享六阶段+精确制品绑定） | done（五级双语/完整源能力） | passed（主机总覆盖783；本批受影响范围复跑） | partial（a72f92b：X5 8GB priority0/7 与S100默认完整源对照passed；其他目标待测） | changes-required（B6-B1/B2已实板确认修复；完整目标矩阵待补） | no | [B6 记录](2026-09-23-b6-sam-review.md)、[独立评审](2026-09-23-b6-independent-host-review.md)、[板测复审](2026-09-24-b6-board-review.md) |
| B6 | mobile_sam / x5:ac11571 + s:380e1a2 | x5/s100/s100p/s600 / python | done（固定源与逐文件哈希） | done（共享六阶段+精确制品绑定） | done（五级双语/完整源能力） | passed（主机总覆盖783；本批受影响范围复跑） | partial（a72f92b：X5 8GB priority0/7 与S100默认完整源对照passed；其他目标待测） | changes-required（B6-B1/B2已实板确认修复；完整目标矩阵待补） | no | [B6 记录](2026-09-23-b6-sam-review.md)、[独立评审](2026-09-23-b6-independent-host-review.md)、[板测复审](2026-09-24-b6-board-review.md) |
| B7 | yolov5 / x5:ac11571 + s:380e1a2 | x5+s / python + cpp | done（固定源清点） | in-progress | in-progress | 分支独立通过（ac4046c Python/evaluator33；fbffddd native+Python37，非全批验收） | partial（X5九变体×8GB/4GB、S100/S600 x-672 Python 源对照 passed；S100P四类拒绝负例 passed；四板 native smoke passed，C++源数值未完） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B7 | fcos / x5:ac11571 | x5 / python（依赖 H1） | done（固定源清点） | in-progress | in-progress | 分支独立通过（73a6de1 38项，含metadata与实际乱序协议） | partial（73a6de1：X5 8GB/4GB efficientnetb0/b2/b3 完整源对照 passed，原顺序误拒绝已修复） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B7 | yoloworld / x5:ac11571 | x5 / python | done（固定源清点） | in-progress | in-progress | 分支独立通过（ac4046c 19项） | partial（ae0f185/73a6de1：X5 8GB/4GB dog 提示/图片完整源对照 passed；非全词汇精度） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B7 | lprnet / x5:ac11571 | x5 / python | done（固定源清点） | in-progress | in-progress | 分支独立通过（73a6de1 23项，含metadata与实际四维协议） | partial（73a6de1：X5 8GB/4GB lpr.bin 完整源对照 passed，四维绑定已修复） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B7 | modnet / x5:ac11571 | x5 / python（资产 manual） | done（固定源清点） | in-progress | in-progress | 分支独立通过（ac4046c 13项；无实板制品） | not-run（manual 模型制品尚未找到，不以模拟代替） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B7 | bytetrack / s:380e1a2 | s / python（依赖 yolov5） | done（固定源清点） | in-progress | in-progress | 分支独立通过（ac4046c 12项） | partial（ae0f185/4d45f9a：S100四帧合成及S100/S600真实视频前30帧源对照 passed；S100P源URL404未推理；非整段视频/MOT精度） | changes-required（真实 SDK 独立核对；分支整改中） | no | [B7 记录](2026-09-23-b7-detection-tracking-review.md)、[源清点](2026-09-23-b7-source-audit.md)、[实板独立核对](2026-09-24-b7-native-sdk-review.md) |
| B8 | unet / x5:ac11571 | x5 / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | unetmobilenet / s:380e1a2 | s / python + cpp（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | pp_liteseg / x5:ac11571 | x5 / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | yolo26_depth / x5:ac11571 + s:380e1a2 | x5+s / python + x5 cpp | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | depth_anything_v2 / s:380e1a2 | s / python | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | lanenet / s:380e1a2 | s / python + cpp（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | pointnet / s:380e1a2 | s / python（点云输入） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B8 | diffusiondrive / s:380e1a2 | s / python（依赖 H1） | pending | pending | pending | not-run | not-run | not-run | no | — |
| B9 | ultralytics_yolo 收编 yolo11/yolo11_pose/yolo11_seg/yolov13_imoonlab / s:380e1a2 | s / 新 family/variant / python + cpp | pending | pending | pending | not-run | not-run | not-run | no | [README 欠账已于 2026-09-26 提前清零](2026-09-26-yolo-readme-debt-closure.md)（基线与 workflow 旗标已删；本行系列收编仍 pending） |
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

### B1 R1a/R1b 修复与 B2 推进决定（2026-09-21）

- **R1a/R1b 已修复**（resnet 双语转换 README：校准命令改为 conversion cwd 内的 `python3 get_calibration_data.py` 并补记下载→校准→编译相对路径衔接；删除"变换链与 YAML 一致"错误声明，明确 mean 同/scale 异（脚本 0.017 vs YAML 逐通道）为保留的源配方差异、未经 OE 重建与数值对照确认；总览表"完整可重放"收紧）。三个转换文件及其 SHA 固定不变。resnet 52 OK、checker 0 violations。**修复记录为待独立确认**（见 [整改记录](2026-09-21-b1-remediation.md) 复审遗留整改节）；B1 各行 Closed=no、Review=changes-required 保持不变。
- **用户决定（2026-09-21，如实记录）**：S600 MobileNetV2 C++ 不补测，保持 not-run，不作为推进 B2 的阻断项（该板位本就在 B1 既定门槛之外）；R1a/R1b 文档修正完成后**授权直接开启 B2，无需等待 B1 下一次独立复审**。此为用户对批次顺序的显式授权，不改变 B1 的 Closed=no 与独立评审未通过的记录状态；B1 的独立复审仍须在后续完成 R1a/R1b 确认。

### 最新独立评审状态（716bdca）

B1 已关闭（pass / ready，限既定批次范围），历史未关闭说明由本段更新。B2 changes-required / not-ready，Closed=no；见 [独立报告](2026-09-21-b2-independent-review.md)。B2 正确板测计数为 28 次对照：26 次 ID 一致、2 次精确平局裁定；加 10 正向 CLI、3 拒绝负例。上表 Board 历史文本中跨实现“逐字节相同”不成立，待 B2-R3 修正；原始数据已保留。B3 仍 pending。

### B2 整改复审更新（0a6deaa）

436 tests 与 CI 通过，R2/R3 关闭；R1 默认选择修复通过，整改板测目前仅作者文字摘要，待 B2-R1-E 补齐 overlay 部署身份及原始输出。旧显式 asset-id 板测记录继续有效，不代表新增默认入口通过独立验证。Closed=no，B3 pending。见 [整改复审报告](2026-09-21-b2-independent-rereview.md)。

### B2 最终独立关闭（21c833a）

B2 pass / ready（既定范围）/ Closed=yes；此段更新前述历史未关闭状态。两板默认入口证据绑定通过，N1/N2 确认，126 项针对性独立测试和 CI 通过。CLI 分数一致仅到六位小数，不宣称 raw tensor 逐位相同。见 [关闭报告](2026-09-21-b2-independent-closure.md)。B3 保持 pending，等待用户安排。

### 外出期间板测执行策略（2026-09-22）

用户明确要求静默暂缓板测：恢复指示前不连接/探测局域网板卡、不周期重试或重复提醒。B3 Board 保持 not-run，主机侧工作和独立评审可继续；本次仅状态检查，Review=not-run、Closed=no 不变，不启动 B4。

### B3 独立评审（2026-09-22）

545 项主机测试与 CI 通过，但确认 B3-R1–R4，Review=changes-required，Closed=no。见 [独立报告](2026-09-22-b3-independent-review.md)。用户授权统筹远程电脑经 Git 对齐板测；环境尚在预检，Board 保持 not-run，Mac 不连接局域网板卡。

## 最新排期：主机侧连续推进（2026-09-22）

用户授权 Codex 接手后续开发与文档评审，板端复验/校准留待用户恢复环境。B3 原 R1–R4 整改已复核，Board=not-run、Closed=no；允许继续 B4 主机开发。三处新增文档清理为 Codex 作者修正。远程 API 与授权暂停，不执行板测。规则和队列见 [主机开发与板端交接](2026-09-22-host-development-and-board-handoff.md)。本节取代历史不进入 B4 的排期限制，不覆盖历史验收证据。

### 2026-09-23 用户暂停

本轮在 B7 整改过程中暂停，未关闭本批；已有子项测试不代表 B7 完整通过。B8 仅完成只读源清点，后续批次状态不升级。接手入口：[Claude Code handoff](2026-09-23-claude-code-handoff.md)，含最新工作区快照、未提交成果及中途修改。
