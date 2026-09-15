# YOLO26 合并验收记录

实施分支：本地 `develop`。合并前基线：`d9ef273`。维护入口：`samples/vision/ultralytics_yolo`，版本参数 `--family yolo26`。

## 合并结果

| 部分 | 最终结构与行为 |
| --- | --- |
| Sample 入口 | 与 YOLOv8 等系列共用 `runtime/python/main.py`，支持 detect / cls / seg / pose / obb |
| 平台 | X5 / S100 / S100P / S600；复用现有平台选择与 NV12 输入适配 |
| 任务实现 | 分类直接复用 YoloCls；其余四任务保留 YOLO26 的四通道 LTRB 协议，与 DFL 解码分开 |
| 公共逻辑 | 模型加载、输入绑定、预处理、调度、绘制、下载、COCO / ImageNet 评测共用 |
| 导出 | 同一个 export_monkey_patch.py 入口；五个任务补丁各维护一份，存放在 conversion/yolo26 |
| 编译 | 同一个 mapper.py 入口，通过 family/platform 选择原有 X5/S 标定、编译流程 |
| 下载 | 100 个已有资产，每平台 25 个；分类 224×224，其余 640×640；URL 无变化 |
| 旧路径 | X5/S ultralytics_yolo26 的运行、下载、导出、编译、评测脚本转发到共用实现 |
| Python 兼容 | 旧 X5 姿态返回 list[dict]、S 姿态返回四元组；旧 X5 分割返回整图掩码，通过适配层保留 |
| 范围外 | 未改 YOLOE、独立 YOLOv5、yolo26_depth、X3 实现及平台全局 utils |

保留原有 conversion / evaluator / model / runtime / test_data 框架。平台编译过程仍有不同实现，但没有新增独立维护的 YOLO26 Sample。未来模型版本可以在协议核实后复用任务实现；本次不预先宣称未来版本与 YOLO26 兼容。

## 保留的差异

- X5 输入为单 packed NV12，S 输入为 Y/UV 两张量。
- 运行 CLI 的 NMS 默认 X5 0.70、S 0.45；旧 Python 类和评测 CLI 的默认值分别保留，不能混同。
- YOLO26 分类两端默认 resize=0；其他系列既有默认值不变。
- X5 OBB 角度使用 `(sigmoid - 0.25) * pi`，S 使用 `(sigmoid - 0.5) * pi`；X5 保留角度归一化、按类别的原手工旋转框 NMS 和坐标裁剪，S 保留 OpenCV NMSBoxesRotated 与原坐标变换。
- X5 导出默认 opset=11 / simplify=True，S 默认 opset=19 / simplify=False；显式覆盖通过测试。
- 旧 X5 ImageNet ordered 标签转为零基索引，S 使用文件名标签；旧评测路径保留数据集地址、阈值和结果文件默认值。
- YOLO26 姿态预测 JSON 保留 X5 的 1/2 可见性标记和 S 的固定 1；其他 YOLO 系列维持既有标记。
- OBB 评测脚本只导出预测 JSON，没有新增或宣称 DOTA AP。

## 有意修正，不能算作数值完全等价

1. 原 X5 部分任务按输出空间尺寸升序排列，却按 stride 8/16/32 解码；分割原型的旧分组判断也有问题。现在按尺寸与通道绑定输出角色，拒绝缺失、歧义、错误 DFL 通道和非浮点输出。
2. X5 分割原来硬编码 640，并直接把掩码缩放到原图。现在采用共用的真实输入几何和逆 letterbox 处理；旧 API 将裁剪掩码恢复为整图格式。此项是掩码坐标修正，未声称新旧 X5 掩码逐像素相等。
3. 导出失败不再仅打印后正常退出，而是返回错误；依赖显式安装，帮助命令不加载训练或板端依赖。

原 Benchmark / Manifest 保持历史发布证据，未作为此次修改后的重新实测结果。发布 catalog 与修改前逐对象比较完全一致，校验值仍为 `catalog-v1.0.0-38a804d954f9b6ce`：54 系列、584 配置、820 条 Benchmark。

## 主机验收

| 检查 | 结果与覆盖 |
| --- | --- |
| 共用 Sample 回归 | 30 项通过；随后新增姿态可见性检查，YOLO26 专项共 11 项再次通过 |
| YOLO26 输入契约 | 五任务 × 四平台，模拟板端元数据；输入尺寸、顺序和 NV12 绑定 |
| 模型资产 | 四平台 × 五任务 × 五规格，100 个 URL 全部匹配已发布 catalog |
| 新旧 CLI | 旧 X5/S 五任务 dry-run、评测和导出 help、编译 help、旧评测默认参数解析 |
| Shell | 实际 Bash 运行；旧通用 X5 全量 67、旧 YOLO26 全量 25、默认五任务、S march 位置参数 |
| 绘制 | 五任务通过模拟结果执行实际绘制/输出流程 |
| 导出 | 模拟 Ultralytics，验证五任务平台默认值、显式覆盖、失败异常；未执行真实导出 |
| YOLO26 原实现对照 | 26 组合成数据全部通过所比较字段；框/分数/类别/关键点/旋转框及 S 掩码；不包含 X5 掩码逐像素等价 |
| 原 YOLO 回归对照 | 20 组合成数据对照通过 |
| S 分类命名回归 | 2 项通过 |
| 目录发布 | 95 项测试通过，TypeScript 类型检查及 catalog 可重复构建通过 |
| 静态检查 | Python 编译检查、git diff --check |

可重复主机检查：

```bash
python -m unittest discover -s samples/vision/ultralytics_yolo/tests
python -m unittest discover -s platforms/s/tests -p test_yolo_cls_resolution.py
npm --prefix tools/catalog-publisher run check
```

本地审计快照和数值对照结果保存在忽略目录 audit/，原始实现可由基线提交恢复。

## 上板验收边界

没有真实板卡推理、完整数据集精度评测、真实 ONNX 导出或 OpenExplorer 编译结果。发布前仍需逐平台验证五任务，重点复验 X5 分割掩码/姿态输出顺序及 OBB 重叠边界框。当前验收结论为源码整合与主机回归通过，不是发布级精度验收。

只做本地提交，未推送；main 和历史平台 tag 不变，模型服务器未修改。
