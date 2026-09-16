# 全仓源码迁移映射

日期：2026-09-16；源码基线：a282b36。本文是 [完整 Spec](2026-09-16-unified-model-zoo-spec.md) 的覆盖清单，不是已完成迁移或已验证平台矩阵。

**目录存在不等于模型在该平台受支持。** 表中 x3/x5/s 表示来源树；实际支持必须核对已发布资产、runtime 和实机信息。S 目录不能直接推断支持 S100、S100P、S600 全部三款。

已扫描源目录共 89 个（X5=37、S=37、X3=15）；另有根目录已合并 samples/vision/ultralytics_yolo。所有行初始状态均为“待审计/待基础验收”；旧 YOLO 转发已存在，但仍未满足本轮板测要求。

目标路径是组织规划。标为候选归并者若协议不等价，保留目标 Sample 内独立任务/版本模块；若部署流程实质不同，再审阅调整映射。不能为了满足表格去删除不同实现。

## 1. 逐源目录映射

| 来源树 | 当前路径 | 目标维护路径 | 处理要求 |
| --- | --- | --- | --- |
| x5 | `platforms/x5/samples/robotics/himloco` | `samples/robotics/himloco` | 保留离线观测到动作流程及现有 C++；不包含真机闭环验收 |
| x5 | `platforms/x5/samples/vision/clip` | `samples/vision/clip` | 保留 BPU/CPU 等实际组件和依赖，不与 SigLIP 自动合并 |
| x5 | `platforms/x5/samples/vision/convnext` | `samples/vision/convnext` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/edgenext` | `samples/vision/edgenext` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/efficient_sam` | `samples/vision/efficient_sam` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/efficientformer` | `samples/vision/efficientformer` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/efficientformerv2` | `samples/vision/efficientformerv2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/efficientnet` | `samples/vision/efficientnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/efficientvit` | `samples/vision/efficientvit` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/fasternet` | `samples/vision/fasternet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/fastvit` | `samples/vision/fastvit` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/fcos` | `samples/vision/fcos` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/googlenet` | `samples/vision/googlenet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/hgnetv2` | `samples/vision/hgnetv2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/lprnet` | `samples/vision/lprnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobile_sam` | `samples/vision/mobile_sam` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobilenetv1` | `samples/vision/mobilenetv1` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobilenetv2` | `samples/vision/mobilenetv2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobilenetv3` | `samples/vision/mobilenetv3` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobilenetv4` | `samples/vision/mobilenetv4` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/mobileone` | `samples/vision/mobileone` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/modnet` | `samples/vision/modnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/paddleocr` | `samples/vision/paddleocr` | 跨平台/版本流程核对；不能仅按 OCR 任务名称替换 |
| x5 | `platforms/x5/samples/vision/pp_liteseg` | `samples/vision/pp_liteseg` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/repghost` | `samples/vision/repghost` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/repvgg` | `samples/vision/repvgg` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/repvit` | `samples/vision/repvit` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/resnet` | `samples/vision/resnet` | 归并同系列规格，不改变权重和输入含义 |
| x5 | `platforms/x5/samples/vision/resnext` | `samples/vision/resnext` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/ultralytics_yolo` | `samples/vision/ultralytics_yolo` | 复用现有统一实现；补新规范与受影响板卡基础验收 |
| x5 | `platforms/x5/samples/vision/ultralytics_yolo26` | `samples/vision/ultralytics_yolo` | 旧入口已转发到统一实现；保留 LTRB 和平台差异，不重复迁移 |
| x5 | `platforms/x5/samples/vision/unet` | `samples/vision/unet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/vargconvnet` | `samples/vision/vargconvnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/yolo26_depth` | `samples/vision/yolo26_depth` | 独立深度流程，不并入 YOLO26 detect |
| x5 | `platforms/x5/samples/vision/yoloe` | `samples/vision/yoloe` | 审计提示方式和变体后归并平台实现 |
| x5 | `platforms/x5/samples/vision/yolov5` | `samples/vision/yolov5` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| x5 | `platforms/x5/samples/vision/yoloworld` | `samples/vision/yoloworld` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/llm/gemma4-e2b` | `samples/llm/gemma4-e2b` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/llm/minicpm5-2b` | `samples/llm/minicpm5-2b` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/speech/asr` | `samples/speech/asr` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/speech/kws` | `samples/speech/kws` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/speech/paraformer` | `samples/speech/paraformer` | 保留三 HBM + CPU CIF 完整流程；核实实际支持子平台 |
| s | `platforms/s/samples/vision/3dresnet` | `samples/vision/3dresnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/bytetrack` | `samples/vision/bytetrack` | 追踪流程与检测器依赖分开核对 |
| s | `platforms/s/samples/vision/depth_anything_v2` | `samples/vision/depth_anything_v2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/diffusiondrive` | `samples/vision/diffusiondrive` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/dinov2` | `samples/vision/dinov2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/efficient_sam` | `samples/vision/efficient_sam` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/efficientnet` | `samples/vision/efficientnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/lanenet` | `samples/vision/lanenet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/mobile_sam` | `samples/vision/mobile_sam` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/mobilenetv1` | `samples/vision/mobilenetv1` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/mobilenetv2` | `samples/vision/mobilenetv2` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/mobilenetv3` | `samples/vision/mobilenetv3` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/mobilenetv4` | `samples/vision/mobilenetv4` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/paddle_ocr` | `samples/vision/paddleocr` | 候选归并；识别/检测组件组合、字典、预处理分别核对 |
| s | `platforms/s/samples/vision/pointnet` | `samples/vision/pointnet` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/resnet152` | `samples/vision/resnet` | 归入 resnet，保留规格和输入配置 |
| s | `platforms/s/samples/vision/resnet18` | `samples/vision/resnet` | 归入 resnet，保留规格和输入配置 |
| s | `platforms/s/samples/vision/resnet50` | `samples/vision/resnet` | 归入 resnet，保留规格和输入配置 |
| s | `platforms/s/samples/vision/siglip` | `samples/vision/siglip` | 保留两个输出子模型语义；不冒充完整图文匹配 |
| s | `platforms/s/samples/vision/ultralytics_yolo` | `samples/vision/ultralytics_yolo` | 复用现有统一实现；补新规范与受影响板卡基础验收 |
| s | `platforms/s/samples/vision/ultralytics_yolo26` | `samples/vision/ultralytics_yolo` | 旧入口已转发到统一实现；保留 LTRB 和平台差异，不重复迁移 |
| s | `platforms/s/samples/vision/unetmobilenet` | `samples/vision/unetmobilenet` | 保持独立，不因名称含 unet 就替换为 X5 unet |
| s | `platforms/s/samples/vision/vit` | `samples/vision/vit` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vision/yolo11` | `samples/vision/ultralytics_yolo` | 候选归并；审计独立导出和输出协议，不等价部分保留独立模块 |
| s | `platforms/s/samples/vision/yolo11_pose` | `samples/vision/ultralytics_yolo` | 候选归并 pose；核对关键点解码及输出格式 |
| s | `platforms/s/samples/vision/yolo11_seg` | `samples/vision/ultralytics_yolo` | 候选归并 seg；核对掩码原型、坐标恢复与量化 |
| s | `platforms/s/samples/vision/yolo26_depth` | `samples/vision/yolo26_depth` | 独立深度流程，不并入 YOLO26 detect |
| s | `platforms/s/samples/vision/yoloe11_seg` | `samples/vision/yoloe` | 候选归并；明确与 X5 yoloe 的提示方式、输出及量化差异 |
| s | `platforms/s/samples/vision/yolov13_imoonlab` | `samples/vision/ultralytics_yolo` | 候选归并；必须核对其与既有 YOLOv13 的区别，不自动增加其他平台支持 |
| s | `platforms/s/samples/vision/yolov5` | `samples/vision/yolov5` | 同名跨平台实现归并前审计协议；单平台能力原样保留，不自动扩展支持 |
| s | `platforms/s/samples/vla/act` | `samples/vla/act` | gitlink；保持 326ea043be204de25223d95c7d918efe8672dc66，审计外部子模块后再处理入口 |
| s | `platforms/s/samples/vla/pi0` | `samples/vla/pi0` | gitlink；保持 a32de276bc1681a2b1531012de111eaa1c16acb6，不与 ACT 锁定版本混同 |
| x3 | `platforms/x3/demos/classification/GoogLeNet` | `samples/vision/googlenet` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/MobileNetV1` | `samples/vision/mobilenetv1` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/MobileNetV2` | `samples/vision/mobilenetv2` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/MobileNetV4` | `samples/vision/mobilenetv4` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/MobileOne` | `samples/vision/mobileone` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/RepGhost` | `samples/vision/repghost` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/RepVGG` | `samples/vision/repvgg` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/RepViT` | `samples/vision/repvit` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/classification/ResNet` | `samples/vision/resnet` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/detect/FCOS` | `samples/vision/fcos` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/detect/PaddleOCR` | `samples/vision/paddleocr` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配 |
| x3 | `platforms/x3/demos/detect/YOLOv10` | `samples/vision/ultralytics_yolo` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配；输出协议与目标任务逐项比对，X3 板测后才确认能力 |
| x3 | `platforms/x3/demos/detect/YOLOv5` | `samples/vision/yolov5` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配；输出协议与目标任务逐项比对，X3 板测后才确认能力 |
| x3 | `platforms/x3/demos/detect/YOLOv8` | `samples/vision/ultralytics_yolo` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配；输出协议与目标任务逐项比对，X3 板测后才确认能力 |
| x3 | `platforms/x3/demos/Instance_Segmentation/YOLOv8-Seg` | `samples/vision/ultralytics_yolo` | 核实 X3 发布资产；提取 Notebook 独有逻辑；运行时经决策门选择原生兼容层或过渡适配；输出协议与目标任务逐项比对，X3 板测后才确认能力 |

## 2. 目录外能力与资料

| 当前范围 | 目标或处理方式 | 必须保留/验证 |
| --- | --- | --- |
| platforms/{x5,s}/datasets | 根 datasets，同名数据逐内容比对 | 数据版本、子集、标签、脚本、来源与许可，不按目录名覆盖 |
| platforms/x3/demos 下类别 README 与量化教程 | 相应分类/Sample README 与 conversion 说明 | 独立 Markdown 不因只扫描子目录而漏掉 |
| platforms/{x5,s}/utils | 分配到 samples/_shared、Sample、datasets 或对应资料 | 函数消费者、默认行为、数值差异与测试 |
| platforms/registry.json | docs/release/platforms.json | 平台 ID、历史来源、现有声明；替换旧路径，保留 provenance |
| 三平台 models.yaml / benchmarks.yaml | docs/release/{x3,x5,s}/ | 稳定 ID、资产 URL/hash、Benchmark 数值与历史证据 |
| 平台 release schemas | docs/release/schemas/ | 相同才共用，必要差异通过 schema 版本表达 |
| tools/catalog-publisher | docs/catalog-publisher，整项目迁移 | package/lock、src/tests/scripts、sources、CI、构建产物接口 |
| tools/catalog-redirect | 文档站维护重定向源；旧站点发布仍需验证 | 旧 URL、query 参数、锚点行为，不直接删除旧部署入口 |
| .github/workflows | 原位置，更新路径与独立发版逻辑 | PR 检查、源码/Skills 发布隔离、权限和并发 |
| .gitmodules 及 ACT/PI0 gitlink | 更新为 samples/vla 下新路径 | 保留各自固定 SHA，不拉最新替代，不自动把子模块内容复制入主仓 |
| platforms/{x5,s}/tros | 已读说明归 docs/tros，实际代码若存在另列 | 已有 TROS 关联信息不因无完整实现而丢失 |
| platforms/x3/resource 等资源 | 被单 Sample 使用放其 test_data/；文档图片随文档；共享小数据归 datasets | 引用路径、许可、图片内容；独有资源不删除 |
| 平台 LICENSE、CHANGELOG、历史发布说明 | 根许可证与 docs/releases 中保留适用来源说明 | 不推定旧文件可重新授权，不重写历史 |
| archive/.staging 与未跟踪工作文件 | 清点引用和所有权，默认不触碰 | 不因目标树没有显示就删除用户文件 |
| 原七 Skills 交付包 | skills/ 与适用规范/CI 按内容合并 | 未获取前不宣称已导入；旧补丁不可整体覆盖当前仓库 |

## 3. Notebook 去向清单

本次三平台目录扫描共发现 20 个 Notebook，均在 X3；以下逐文件列出，均需提取独有内容后移除。执行时另扫 git ls-files 和已检出的子模块，数量不同应解释而非强行匹配。

| Notebook | 归属目标 | 验收 |
| --- | --- | --- |
| `platforms/x3/demos/classification/GoogLeNet/test_GoogLeNet.ipynb` | `samples/vision/googlenet` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/MobileNetV1/test_MobileNetV1.ipynb` | `samples/vision/mobilenetv1` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/MobileNetV2/test_MobileNetV2.ipynb` | `samples/vision/mobilenetv2` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/MobileNetV4/test_MobileNetV4.ipynb` | `samples/vision/mobilenetv4` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/MobileOne/test_MobileOne.ipynb` | `samples/vision/mobileone` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/RepGhost/test_RepGhost.ipynb` | `samples/vision/repghost` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/RepVGG/test_RepVGG.ipynb` | `samples/vision/repvgg` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/RepViT/test_RepViT.ipynb` | `samples/vision/repvit` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/classification/ResNet/test_ResNet18.ipynb` | `samples/vision/resnet` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/FCOS/jupyter_BSP_FCOS_Detect.ipynb` | `samples/vision/fcos` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/FCOS/jupyter_ModelZoo_FCOS_Detect.ipynb` | `samples/vision/fcos` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/PaddleOCR/test_paddle_ocr.ipynb` | `samples/vision/paddleocr` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv10/jupyter_BSP_YOLOv10_Detect.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv10/jupyter_ModelZoo_YOLOv10_Detect.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv5/jupyter_BSP_YOLOv5.ipynb` | `samples/vision/yolov5` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv5/jupyter_ModelZoo_YOLOv5.ipynb` | `samples/vision/yolov5` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv8/jupyter_BSP_YOLOv8_Detect.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/detect/YOLOv8/jupyter_ModelZoo_YOLOv8_Detect.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/Instance_Segmentation/YOLOv8-Seg/jupyter_BSP_YOLOv8_Instance_Segmentation.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |
| `platforms/x3/demos/Instance_Segmentation/YOLOv8-Seg/jupyter_ModelZoo_YOLOv8_Instance_Segmentation.ipynb` | `samples/vision/ultralytics_yolo` | 脚本/Markdown 内容去向可审查，干净进程可执行；尚未迁移 |

## 4. 执行时逐行附加的验收记录

同一目标 Sample 的多行来源合并为一次独立验收，但必须列出全部受影响平台。按需为完成项附：来源提交 → 实现提交 → 模型与输入 → 板卡环境 → 命令及日志/结果路径 → 结论与检查者 → 未覆盖范围。该记录是人可读项目文档，不是新的自动执行配置。

若某原目录只有历史文档、缺少可用资产或同名内容实属其他平台，应标记“来源待核实”，保留证据并提交范围判断，不能无声删除或新增支持。
