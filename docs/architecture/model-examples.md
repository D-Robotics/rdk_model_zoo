# 可读模型范例架构（ResNet / Ultralytics YOLO）

日期：2026-10-01。状态：已实施（分支 `codex/readable-model-examples-20261001`）；
实施映射见 [迁移说明](../migration/2026-09-30-model-examples.md)，方案见
[设计文档](../superpowers/specs/2026-09-30-readable-model-examples-design.md)。

本文回答三个问题：示例代码各层放什么、调用链长什么样、自训练模型从哪里接入。
事实来源是两个 sample 的实际源码与 README；本文不新增第二套规范，
README 契约（`docs/sample-standards/readme-contract.md`）与推理契约
（`docs/sample-standards/inference-contract.md`）继续适用。

## 1. 调用链与分层

```text
main.py（薄入口：解析参数 → 构造模型 → predict → 展示）
  → 模型类（classify.py 的 ResNetClassifier / detect.py 的 YoloDetect）
      preprocess → infer → postprocess，predict 显式串联三步
  → 样例 runner（RuntimeModelRunner / ModelRunner：绑定与张量校验）
  → 薄 SDK 会话 samples/_shared/runtime.py:RuntimeSession
      （目标身份检查 → 懒导入 hbm_runtime → 构造模型实例）
  → hbm_runtime（板端 SDK，非本仓代码）
```

分层规则（评审关注点）：

- **main.py 只做必要的事**：参数解析、模型-free 模式委托（list/dry-run/download）、
  选择解析、模型构造、`predict` 调用与结果展示。冗长选项声明与展示函数放在
  示例本地 `cli.py`（YOLO 侧为 `yolo_cli.py`）。入口不含图像缩放、张量转换、
  SDK 加载、NMS、坐标还原等业务。
- **模型类文件可见完整主线**：一个文件内可见初始化、`preprocess`、`infer`、
  `postprocess`、`predict`。既有的 `pre_process` / `forward` / `post_process`
  名称是同一实现的薄别名，不维护第二份实现。复杂算法（NV12 打包、DFL/LTRB
  解码、NMS、量化变换）继续留在共享模块，不为了"文件完整"复制。
- **公共 Runtime 无模型算法**：`samples/_shared/runtime.py` 只承接 SDK 导入、
  模型实例创建与按目标身份检查；输入输出沿用 SDK 原生映射，不猜测模型名或
  输出语义。集成的 runner 用会话作为加载边界，随后在已加载 SDK 对象上执行
  自身已校验的调用（与会话 `run` 的直通行为一致），不声称所有推理都经过
  `session.run`。
- **不新增框架**：没有新的顶层包、插件系统或统一模型描述文件；自训练接入
  使用既有配置类型（`ModelSelection`/`YoloDetectConfig`）与标签文件。

## 2. 三条使用路径

两个范例各提供三条路径；细节以各 sample README 与其 runtime/conversion/model
子文档为准，此处为导航。

### 2.1 运行官方模型

```bash
# ResNet（x5；s100/s600 替换 target 与引用）
bash samples/vision/resnet/model/download.sh x5
python3 samples/vision/resnet/runtime/python/main.py --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin

# YOLO 检测（x5）
python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolo11 --task detect
```

`--list-models` / `--dry-run` 在主机即可运行，不加载板端 SDK。

### 2.2 接入自训练模型

- **ResNet**：本地编译制品用 `model_binding.custom_selection(model_path,
  target, input_height=, input_width=, class_count=, …)` 声明契约，无需官方
  Manifest 注册；绑定仍校验实际张量名/形状/类型与声明的类别数。标签可选：
  不给标签时结果保留类别 ID；序列长度必须等于类别数，否则明确报错。示例见
  [runtime/python/README.md 的 Custom models 节](../../samples/vision/resnet/runtime/python/README.md#custom-model)。
- **YOLO**：`.pt → ONNX → 目标编译` 两阶段在 conversion 完成（X5 opset 11 /
  simplify 1，S opset 19 / simplify 0；输入约定与校准文件见
  [conversion/README.md](../../samples/vision/ultralytics_yolo/conversion/README.md)），
  运行时 `--model-path` 指向产物并配 `--family` 选择解码协议；自定义类别数用
  `--classes-num`（detect/seg/obb）与 `--label-file`。支持的协议边界：DFL
  （yolov5u/v8/v9/11/12/v13）、直接 LTRB（yolo26）、S 系列 NMS-free
  （yolov10）；不声称任意 Ultralytics 权重可直接运行。

### 2.3 修改业务调用

直接导入模型类，`predict` 接受本地图片路径或 BGR `uint8` 数组，返回既有
结构化结果（`ClassificationResult` / `DetectionResult`）。逐调用几何放在
prepared 输入上，连续不同尺寸图片互不污染；预测本身不打印、不绘图、不写
文件，展示由调用方完成。最小示例见各 runtime README 的 integration-example
章节。

## 3. 对应表：网络 → 权重 → 编译 → 产物 → Runtime

### 3.1 ResNet

| 项 | 事实 | 证据位置 |
| --- | --- | --- |
| 网络 | TorchVision ResNet18/50/152，ImageNet-1k | conversion/README（source-model） |
| 权重 | `IMAGENET1K_V1` 官方预训练（export 脚本 `--weights`）；resnet152 有已发布 ONNX | `export_resnet18_onnx.py`、conversion/README |
| 输入/输出协议 | 224×224 NV12（x5 packed / S split Y+UV），默认 letterbox；输出 F32 分数向量，rank 规则 squeeze → `(1000,)` | runtime/python/README（stage-io） |
| 编译 | x5 `hb_mapper`→`.bin`（bayes-e）；s100/s600 `hb_compile`→`.hbm`（nash-e/nash-p）；resnet18 校准配方属 OE `13_resnet18`（declared gap），resnet152 保留源分支 OE 配方 | conversion/README |
| 产物 | 7 个已发布制品（model/README artifacts 表） | model/README |
| Runtime | `main.py` → `classify.ResNetClassifier` → `RuntimeModelRunner`（共享会话加载） | runtime/python |

未证实项：S100P 无 ResNet 行；ResNet50 无本地导出配方（OE 示例为权威）；
板端复跑本轮 not-run。

### 3.2 Ultralytics YOLO

| 项 | 事实 | 证据位置 |
| --- | --- | --- |
| 网络 | yolo26 / yolov5u / yolov8 / yolov9 / yolov10 / yolo11 / yolo12 / yolov13（见家族表） | model/README、runtime/python/README |
| 权重 | 用户本地 Ultralytics `.pt`（不随仓分发；裸模型名不作为本地权重证据） | conversion/README（source-model） |
| 输入/输出协议 | detect/seg/pose/obb 640×640 NV12（cls 224）；DFL 家族输出 cls+64bin box（+mask/kpt），yolo26 输出 cls+4 通道 LTRB（+mask/kpt/角度）；yolov10 S 系 NMS-free | conversion/CONVERSION_CONTRACT.md、runtime/python/README |
| 编译 | x5 `hb_mapper`→`.bin`，校准 raw float32 `.rgbchw`；s100/s100p/s600 `hb_compile`→`.hbm`，校准 float32 `.npy`（/255） | conversion/README（toolchain-targets 表） |
| 产物 | 家族×任务×尺度清单见 model/README artifacts 表 | model/README |
| Runtime | `main.py` → 任务类（detect.py 等，按 `yolo_dispatch` 协议分派）→ `ModelRunner`（共享会话加载） | runtime/python |

未证实项：并非每个家族×任务×尺度组合都有板端验证；OBB 需航拍图；
本轮板端推理 not-run。

## 4. 边界与不承诺

- 薄 SDK 会话不是自研推理引擎；真实执行依赖板端安装的 `hbm_runtime`。
- 本轮未执行：真实 ONNX 导出、工具链量化编译（OE/Mapper/HMCT）、板端推理。
  这些证据在各 sample 文档中单列 not-run，不以主机测试替代。
- C++ 运行时维持既有能力与构建方式，未纳入本轮可读性重构。
- 历史平台目录（`platforms/`）的移除与依赖迁移见迁移说明，不属于本文范围。
