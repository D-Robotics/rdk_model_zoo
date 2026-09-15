# YOLO26、YOLOE、YOLOv5 后续合并方案（原始提案）

> 修订：用户已否决下文 YOLO26 独立 Sample 方案。YOLO26 改为并入 `samples/vision/ultralytics_yolo`，使用 `--family yolo26`；执行依据见 [统一入口计划](../plans/2026-09-15-yolo26-unified-entry.md)。下文结构仅为历史讨论，YOLOE / YOLOv5 尚未实施。

基线：本地 develop，d9ef273。本文是根据当前源码检查形成的设计与分批计划，不表示已修改实现或完成模型验证。

## 目标与边界

每个 Sample 一个维护目录、一个 Python 入口。保留 conversion / evaluator / model / runtime / test_data 框架。合并相同 Sample 的平台实现，不把三个模型系列强行塞进上一轮的 ultralytics_yolo。

- YOLO26：合并 X5/S 的 ultralytics_yolo26，覆盖 detect / cls / seg / pose / obb。
- YOLOE：将 X5 yoloe 与 S yoloe11_seg 收敛到 yoloe；保留已经修正的 X5 Prompt-Free 实现约束。
- 独立 YOLOv5：合并 X5/S yolov5，并将 X3 老实现纳入同一 Sample 入口，X3 后端单独保留。它与上一轮的 YOLOv5u 是不同协议。
- yolo26_depth 是独立深度估计 Sample，含独立标定、评测与 lite 产物；本轮 YOLO26 指 ultralytics_yolo26，不顺带处理 depth。
- 不新增未经证明的平台支持，不重写历史 tag，不更改模型服务器资产。先在 develop 本地实施。

## 已检查的差异

| Sample | 源码发现 | 初步难度与合并边界 |
| --- | --- | --- |
| YOLO26 | X5/S 都有五个运行类、五个评测脚本和五个 ONNX 导出脚本；输入分别为 packed NV12 与 Y/UV；检测头按 4 维 LTRB 解码，两端均调用 NMS，但聚合和输出排序不同 | 中等；分类/检测先行，分割/姿态/OBB 分别验收；不能直接套用 YOLOv8 的 DFL 解码 |
| YOLOE | 两端默认 4585 类、DFL=16、mask channels=32；X5 检查十个浮点输出并返回整图掩码；S 显式反量化并返回按框缩放的掩码列表 | 中高；共享前处理、框解码等经证明等价部分，掩码表示和量化处理须保留明确适配 |
| YOLOv5 | 两端为 anchor-based；X5 默认下载 v7.0 的 n/640，S 下载 x/672；X5 返回检测列表，S 返回三个数组；X5 调用 OpenCV NMSBoxes，S 调用工具函数 NMS；X3 使用 hobot_dnn.pyeasy_dnn | X5/S 中等，含 X3 中高；保留版本、尺寸、anchors、输出量化差异，X3 不套用 hbm_runtime |

YOLOE 的 S 下载脚本明确拒绝 S600，且固定指向 nashe 资产，不能据目录名推断 S100P/S600 已支持。当前 catalog 中 yoloe 仅登记 X5 的 3 个配置；需要核对 S 侧源码、README 和 Manifest 的发布状态，不能直接增加支持标记或挪用 X5 benchmark。

当前 catalog 中 yolov26 为 120 个配置，独立 yolov5 为 36 个配置（含 X3）。这些是审计基线，不是合并后必然增加的模型数量。

## 选择的结构

三种处理方式：只搬目录并保留两套代码最容易，但不能完成实现去重；建立一个覆盖全部 YOLO 的大基类会扩大改动；推荐按 Sample 合并任务实现，平台输入、量化和工具链保留小范围适配。

```text
samples/vision/
├── ultralytics_yolo/          # 已完成首批合并
├── ultralytics_yolo26/
├── yoloe/                    # 包含原 S yoloe11_seg
└── yolov5/                   # 独立 anchor-based YOLOv5，含 X3 后端

每个 Sample：
├── README.md / README_cn.md
├── conversion/               # 导出及按工具链区分的配置/流程
├── evaluator/                # 原有评测；未提供的能力明确标注
├── model/                    # 资产索引与统一下载入口
├── runtime/
│   ├── python/
│   │   ├── main.py           # 本 Sample 唯一用户入口
│   │   ├── run.sh
│   │   ├── 任务实现.py
│   │   └── backends/         # 仅在确有运行时差异时增加，例如 YOLOv5 X3
│   └── cpp/                 # 仅保留实际提供的 C++ 能力
├── test_data/
└── tests/
```

不让 YOLOE import 另一个 Sample 的内部文件。首批沿用已验证的小型平台/输入适配方式；跨 Sample 公共代码提取另设独立提交，只有接口和行为均一致时才提取到 utils/yolo，避免在本轮顺带改造成通用推理 SDK。

## 分批执行与独立验收

1. 固定清单和行为基线：保存各平台资产 URL、版本、输入尺寸/格式、输出组织、CLI 默认值、返回类型、评测入口与 catalog 数据；特别记录 YOLOE S 发布状态及 YOLOv5 X3 版本集合。
2. YOLO26 分类/检测：建立共用入口、平台输入适配、资产下载和旧路径转发；合并可证明等价的导出部分，保留独立编译流程。验收两端空输出、阈值边界、多类别重叠、返回顺序、输入尺寸以及 opset。
3. YOLO26 分割/姿态/OBB：分别对照 mask 裁剪/缩放、keypoint 解码/可见度、角度/旋转框与抑制方式。每任务建立合成张量对照和评测接口检查，不能仅以截图相似通过验收。
4. YOLOE：核对 Prompt-Free 资产、标签顺序和输出量化信息；保护 X5 既有严格校验，统一框解码及可共享部分。明确新 API 的掩码坐标空间，旧 API 用兼容适配保持原返回格式。测试整图与局部掩码映射、空输出、极端长宽比、图像边界及量化/浮点输出。
5. YOLOv5 X5/S：资产选择由平台、版本、尺寸共同决定，使用模型元数据校验尺寸；核查 anchors 排列、objectness×class 置信度、sigmoid/反量化执行次数、NMS 的坐标输入及类别处理。NMS 现有调用有待验证，确认缺陷后单独修复提交并记录基线差异。
6. YOLOv5 X3：同一 main.py 路由到独立旧运行时后端，保留历史 v2/v7 等已有资产选择。按 X3 旧实现建立对照，不为统一形式替换其板端 API；不承诺与 X5/S 共用同一底层运行类。
7. 总体验收与本地提交：每个 Sample 单独代码审查、单独测试结果和提交；更新中英文命令、旧路径说明、下载脚本及 catalog 源链接。旧 README/Manifest 证据继续可解析。无证据更正时目录统计和 benchmark 数值保持不变；若确认补登记 YOLOE S，单独记录数据变更及来源。

## 用户可见效果

以下为目标命令格式，尚未实现：

```bash
python samples/vision/ultralytics_yolo26/runtime/python/main.py --platform x5 --task detect --dry-run
python samples/vision/ultralytics_yolo26/runtime/python/main.py --platform s600 --task obb --dry-run
python samples/vision/yoloe/runtime/python/main.py --platform x5 --dry-run
python samples/vision/yolov5/runtime/python/main.py --platform s100 --dry-run
python samples/vision/yolov5/runtime/python/main.py --platform x3 --dry-run
```

相同 Sample 的通用修复修改一处；添加板卡时补平台配置、资产和必要后端，不复制整个 Sample。缺少对应编译产物时给出明确错误。旧调用入口继续工作，但迁移后的 Sample 要求完整仓库，不能只拷贝旧平台子树。

所有主机检查通过只表示代码合并可接受。各支持平台还需真实模型输入/输出检查、精度或数值对照、时延测试及工具链编译验收，之后才能标记为发布完成。
