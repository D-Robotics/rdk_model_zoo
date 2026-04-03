# RDK X5 Demo Refactor Status

## 文档目的

本文档用于记录 `rdk_x5` 分支下各 Demo 的迁移重构状态、当前已确认符合规范的范围，以及本次迁移过程中沉淀出的工程经验，作为后续批量迁移其它 Demo 的参考基线。

---

## 当前已确认符合规范的 Demo

截至本次重构与板端验证，已明确完成并通过验收的 Demo 如下：

### 1. ultralytics_yolo26

路径：

`samples/vision/ultralytics_yolo26`

当前状态：

- 已迁入 `rdk_x5` 分支
- 已统一为 `RDK X5 + .bin + hbm_runtime`
- 已按 Sample 目录规范整理为：
  - `conversion/`
  - `evaluator/`
  - `model/`
  - `runtime/`
  - `test_data/`
- 已提供：
  - `runtime/python/main.py`
  - `runtime/python/run.sh`
  - 各 task 对应独立模型封装文件
- 已完成中英文 README 同步
- 已完成 `ConvNeXt/runtime/python` 风格的注释对齐

符合规范的 task 范围：

- `detect`
- `seg`
- `pose`
- `obb`
- `cls`

板端实测通过的模型范围：

- `detect`: `yolo26n/s/m/l/x_detect_bayese_640x640_nv12.bin`
- `seg`: `yolo26n/s/m/l/x_seg_bayese_640x640_nv12.bin`
- `pose`: `yolo26n/s/m/l/x_pose_bayese_640x640_nv12.bin`
- `obb`: `yolo26n/s/m/l/x_obb_bayese_640x640_nv12.bin`
- `cls`: `yolo26n/s/m/l/x_cls_bayese_224x224_nv12.bin`

板端验证结论：

- `run.sh` 可直接运行
- `fulldownload.sh` 可下载全量模型
- 全部 25 个模型在开发板 `192.168.127.10` 上已完成实跑验证

---

## 当前记录范围说明

本次文档只记录“已经实际检查并验证过”的 Demo。

目前已达到该标准的只有：

- `samples/vision/ultralytics_yolo26`

其它 Demo 是否符合规范，当前未在本轮中做同等深度的结构检查与板端实测，因此暂不在本文档中标记为“已符合规范”。

---

## 本次迁移重构的关键经验

### 1. 不要把 `rdk_s` 当作代码基线，只能把它当作架构参考

`rdk_s` 与 `rdk_x5` 对应不同板卡，运行时能力、模型产物和部署条件并不完全一致。

因此迁移策略应当是：

- 内容迁移参考原始 `main`
- 目标运行环境以 `rdk_x5` 为准
- 架构拆分和代码组织可以参考 `rdk_s`
- 但不能直接照搬 `rdk_s` 代码

### 2. X5 的事实基线是 `.bin`，不是 `.hbm`

本次板端验证已确认：

- `RDK X5` 当前这套 Demo 的推理模型产物是 `.bin`
- `hbm_runtime` 在 X5 上可以直接加载 `.bin`

因此后续 `rdk_x5` Demo 迁移时应统一遵循：

- 模型文件默认按 `.bin` 处理
- 文档、脚本、默认参数都应围绕 `.bin`
- 不要再把 X5 Sample 默认写成 `.hbm`

### 3. Sample 中不要引入“猜模型结构”的动态逻辑

如果模型输出顺序和 layer 结构是固定的，就应按固定协议显式解析，而不是运行时通过：

- tensor shape
- channel 数
- output name 模式

去推断输出语义。

正确做法是：

- `detect`: 按 `[Cls, Box] * 3` 固定索引解析
- `pose`: 按 `[Cls, Box, Kpts] * 3` 固定索引解析
- `obb`: 按 `[Cls, Box, Angle] * 3` 固定索引解析
- `seg`: 按固定 `base_idx` 顺序解析

这类固定协议写法更符合 Sample 的可读性和可维护性要求。

### 4. 入口脚本要薄，模型逻辑要下沉

`main.py` 的职责应限制在：

- 参数解析
- 默认路径组织
- 加载图片和标签
- 创建模型对象
- 调用 `predict`
- 结果保存或日志输出

不应在 `main.py` 中堆叠大量模型细节、推理协议判断和复杂后处理逻辑。

### 5. 优先复用 `utils/py_utils`

对于多个 Demo 会共用的能力，应优先沉入公共工具层，例如：

- `NMS`
- `scale_coords_back`
- `scale_keypoints_to_original_image`
- `filter_classification`
- `decode_ltrb_boxes`

原则是：

- 通用能力放 `utils/py_utils`
- task 私有逻辑保留在 sample 内

这样后续新 Demo 迁移时可以减少重复代码。

### 6. 板端验证必须覆盖脚本入口，不只验证 Python 文件

只跑 `python main.py` 不够，必须额外验证：

- `run.sh` 是否可执行
- 下载脚本是否可执行
- 默认模型和默认图片路径是否有效

因为验收时用户更关注的是“整个 sample 能不能直接用”，而不是某个 Python 文件单独能不能跑。

### 7. Windows 开发环境要额外注意 shell 脚本行尾

本次在板端实际遇到：

- `run.sh`
- `download_model.sh`
- `fulldownload.sh`

上传到 Linux 后因为 `CRLF` 触发 `^M` 问题，导致脚本不可执行。

因此后续凡是新增或修改 `.sh` 文件，都需要额外确认：

- 行尾为 `LF`
- 板端可直接执行

### 8. 规范验收必须同时覆盖“结构、文档、板端实跑”

一个 Demo 是否真正符合规范，不能只看代码目录或文档外观，需要至少同时满足：

- 目录结构符合仓库规范
- `main.py` 参数风格符合规范
- 注释风格符合统一标准
- README 中英文同步
- 板端默认运行链路可用
- 全量模型或至少目标模型集已验证

缺任一项，都只能算“部分完成”。

---

## 对后续 Demo 迁移的建议流程

建议后续每个 Demo 按以下顺序推进：

1. 先确定 `rdk_x5` 上的真实模型格式与运行时约束。
2. 再对齐 Sample 目录结构。
3. 再收敛 `main.py` 与模型封装。
4. 再同步 README 与注释。
5. 最后做板端 `run.sh + 全量模型` 验证。

这样能避免前期文档和代码改完后，最后才发现板卡约束不一致，导致返工。

---

## 后续维护建议

建议后续继续在本文档中按相同格式追加其它 Demo，例如：

- Demo 名称
- 所在路径
- 已符合规范的 task 范围
- 已验证模型范围
- 遗留问题
- 板端验证结论

这样可以逐步形成 `rdk_x5` 分支的迁移重构台账。
