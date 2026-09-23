# B7 源审计：YOLOv5、FCOS、YOLOWorld、LPRNet、MODNet、ByteTrack

日期：2026-09-23。审计依据为固定源 `ac115717197920355fc390bb04299b20e6436864`（X5）与 `380e1a2bf42041af54be6f34935e50197cfadff9`（S）。范围由迁移图 B7 确定：`yolov5`（X5+S，Python+C++）、`fcos`（X5，Python）、`yoloworld`（X5，Python）、`lprnet`（X5，Python）、`modnet`（X5，Python）、`bytetrack`（S，Python，依赖 YOLOv5）。逐文件 hash 与清单快照见 [source inventory](evidence/2026-09-23-b7-source-inventory.json)。本报告只做主机静态审计；未下载模型、联网、运行 SDK、编译板端 C++ 或板测。

## 资产和交付面

| sample | 固定资产事实 | 源文件能力 | 迁移时必须保留 |
| --- | --- | --- | --- |
| yolov5 | X5 manifest 只有 `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`；源文档另列 tag v2/v7 的 n/s/m/l/x 共 9 个。S manifest 有 `s100/`、`s600/` 的 `yolov5x_672x672_nv12.hbm`，未列 s100p。 | X5 Python+C++；S Python+C++；三尺度 anchor detector。 | 具体文件、march、输入布局、输出量化和 resize 绑定到 artifact；不能用文件名猜协议或用 X5 `.bin` 替代 S `.hbm`。 |
| fcos | X5 manifest 三个 `.bin`：512/B0、768/B2、896/B3，hash 均未知。 | Python；5 个 classification、5 个 box、5 个 center-ness 输出。 | 输入尺寸和输出 shape 按实际 metadata 绑定，保留五层 shape 匹配、dequant、FCOS grid/stride 和 NMS。 |
| yoloworld | manifest 仅有 `yolo_world.bin`；运行必需的 `test_data/offline_vocabulary_embeddings.json` 不在 manifest。 | Python；图像加离线文本 embedding 双输入。 | vocabulary 文件是独立必需资产，必须校验维度和 prompt 身份；不能把它当普通分类 label 文件。 |
| lprnet | manifest 仅有 `lpr.bin`，hash 未记录。`test_data/test_input.dat` 是随源提供的预打包输入，不是 manifest 模型。 | Python；无图像预处理，读取 `float32` 二进制。 | 输入文件路径和 `1x3x24x94` reshape；68 字符表、CTC 去重、blank `-`（索引 67）逐字保留。 |
| modnet | manifest 标记 `manual`，`modnet_512x512_rgb.bin` 无 URL/hash。 | Python；RGB matting，可选背景合成。 | 手工模型 identity、几何 context、zero padding、`(pixel-127.5)/127.5`、uint8 matte 与可选 composite。 |
| bytetrack | manifest 有 s100/s100p/s600 三个 YOLOv5x HBM；与 S yolov5 使用同名制品。视频 `track_test.mp4` 不在源 test_data，run.sh 会联网下载。 | Python；YOLOv5x detector + 3rdparty BYTETracker。 | tracker 状态、COCO person class 0 过滤、低分框关联、阈值和依赖包；视频输入必须显式准备。 |

所有发布模型的 manifest hash 当前为 unknown/null；源下载脚本直接 `wget`/`curl`，没有发布 hash 验证。统一 model 层应以 manifest 资产对象为准，外部路径要求精确 asset ID；下载由显式 model 命令完成，runtime runner 不得自动下载或安装依赖。

## Python 运行契约

### YOLOv5 和 ByteTrack

X5 [`yolov5_det.py`](../../../platforms/x5/samples/vision/yolov5/runtime/python/yolov5_det.py) 读取逻辑 NCHW `(1,3,H,W)` metadata，默认 `resize_type=0` 直接 resize，BGR 转 packed NV12 后只喂一个 runtime input；三路输出按 metadata 顺序对应 stride 8/16/32，raw float 输出直接做 sigmoid、anchor decode、score threshold 和 OpenCV NMS（约 87–275 行）。其默认值为 classes 80、score 0.25、NMS 0.45、标准 9 anchors。X5 源 Python 没有 output dequant；这一点必须与 S 分开。

S [`yolov5.py`](../../../platforms/s/samples/vision/yolov5/runtime/python/yolov5.py) 读取 NHWC 平面 shape，默认 `resize_type=1` letterbox，返回两个 split NV12 输入 Y/UV（约 245–286 行）；输出先通过 `post_utils.dequantize_outputs`，再 reshape/decode/filter/NMS/scale（约 301–350 行）。`predict` 允许覆盖 resize、score、NMS，但 `pre_process` 会把覆盖值写回 `cfg.resize_type`，且 `post_process` 用 `score_thres or default`/`nms_thres or default`，显式传 0 会被吞掉；统一实现应改为 per-call context 和 `is None` 判断，避免 A/B/A 串扰或无法表达零阈值。S 的模型 shape、输出量化参数和 output order 必须从实际 metadata 绑定。

X5 与 S 不能抹平为一个 NV12 入口：X5 packed 单输入和 S split Y/UV 是不同 physical contract；X5 默认 stretch、S 默认 letterbox 也是可见结果差异。两侧共用的只应是 YOLO 解码的明确数学层、资产选择和结果类型，平台 tensor IO 与 runtime adapter 保持独立。

ByteTrack [`bytetrack.py`](../../../platforms/s/samples/vision/bytetrack/runtime/python/bytetrack.py) 包装 S YOLOv5 detector，并在 `predict` 中只保留 `cls_ids == 0` 的 person，再调用有状态 `BYTETracker.update`；`tracker` 实现位于 `3rdparty/tracker/`，需要连同 `basetrack.py`、`byte_tracker.py`、`kalman_filter.py`、`matching.py` 迁移。`track_thresh=0.3`、`track_buffer=60`、`match_thresh=0.8`、`frame_rate=30`、`mot20=False` 是源默认。源 `ByteTrack.post_process`（`bytetrack.py:161-175`）把整张 image 作为一个宽度参数传给需要 `(ori_img_w, ori_img_h)` 的 detector post，直接调用会缺少高度参数；而 `predict` 绕过该方法调用 detector.predict。统一 API 应修复并为显式三阶段调用加回归测试，同时显式区分无状态 detector 与有状态 stream tracker；不能把 `predict` 当纯函数或每帧重建 tracker。

### FCOS

[`fcos_det.py`](../../../platforms/x5/samples/vision/fcos/runtime/python/fcos_det.py) 默认 512、`conf_thres=0.5`、`iou_thres=0.6`、直接 resize、strides `[8,16,32,64,128]`。构造时按 `(1,H/stride,W/stride,C)` shape 把输出分成 cls/box/center 三族，不依赖固定 tensor name；post 先 dequant，再计算 `sqrt(sigmoid(cls_max)*sigmoid(center))`，box 乘 stride，按 class NMS 后缩放回原图。统一 binding 必须保存三族及其 shape/quant descriptor；H1 的 raw/dequant transform 是 FCOS 的硬依赖，不能套 YOLOv5 raw-f32 策略。

### YOLOWorld

[`yoloworld_det.py`](../../../platforms/x5/samples/vision/yoloworld/runtime/python/yoloworld_det.py) 的输入是 FP32 image `(1,3,640,640)` 与 text `(1,32,512,1)`。图像按最长边缩放、左上角 zero pad、BGR→RGB、无额外 mean/std；prompt embedding 从 JSON 取出并用最后一个 prompt 填满 32 slots。输出是 `classes_score (1,8400,32)` 和 `bboxes (1,8400,4)`；按 text slot 取 argmax、score threshold 0.05、按 slot 做 NMS 0.45，再以 `_scale` 还原坐标。

这里有两个迁移边界：空 prompt 列表会在填充 slots 时访问 `class_ids[-1]`，CLI 的逗号解析也可产生空列表，统一入口必须显式拒绝；此外 `_scale` 与 `_selected_class_ids` 存在对象可变状态（约 134–151、227–231 行），应放入 per-call context，避免交错调用把一次 prompt/image 的结果用于下一次 post。

### LPRNet

[`lprnet.py`](../../../platforms/x5/samples/vision/lprnet/runtime/python/lprnet.py) 不接受图片；`pre_process()` 读取用户给定 `.dat`，按 runtime metadata reshape，源协议是 float32 `1x3x24x94`。输出 squeeze 成 `68x18`，`decode_plate` 对 argmax time sequence 做 CTC-style consecutive dedup 和 blank 去除，字符表顺序是 31 个省份字符、数字、字母和 `-`。统一任务需要将 binary input 作为明确的 input artifact，不能为它增加未经源证明的 resize/normalization。

### MODNet

[`modnet.py`](../../../platforms/x5/samples/vision/modnet/runtime/python/modnet.py) 做 BGR→RGB、`(pixel-127.5)/127.5`、按长边 `ref_size=512` 的 `INTER_AREA` resize，四周 zero pad 后转 NCHW；post 将 `[0,1]` matte 转 uint8，去 padding 后线性 resize 回原图。`_pad_x/_pad_y/_new_*` 和原图尺寸保存在对象字段（约 98–103、184–197 行），所以统一实现必须返回冻结 per-call geometry context，再由 post 消费；否则连续不同长宽图会串 context。背景合成是 CLI/API 的附加能力，不应混进 forward。

## C++ 与特殊 runtime API

只有 YOLOv5 有 C++ 源，不能把它作为所有 B7 Python sample 的 C++ 支持证明。

- X5 `runtime/cpp/main.cc` 使用 `hbDNNInfer`、`hbSysAllocCachedMem`/`hbSysFreeMem`、packed `.bin` 和单 packed NV12；模型、图片、阈值、anchors、`PREPROCESS_TYPE` 都是编译期宏。当前宏默认 letterbox，而 X5 Python 默认 direct resize；CMake 项目名仍为 `rdk_yolov8_detect`，结果文件固定 `cpp_result.jpg`，并要求三路输出 `quantiType == NONE`。迁移时应保留 X5-only C++，改为显式 target/asset 配置，并将 C++ 与 Python 的默认 resize 差异写入 API。
- S `runtime/cpp/src/yolov5.cpp` 使用 `hbDNNInferV2`、`hbUCPSubmitTask`、`hbUCPMallocCached`/`hbUCPFree` 和 split NV12；CMake 链接 `dnn hbucp gflags fmt`，并在 configure 时读取 `/sys/class/boardinfo/soc_name`。S C++ 默认 `hw_list={(84,84),(42,42),(21,21)}`，即 672 输入，post 统一走 `dequantizeTensorS32`。`infer()` 在提交前把 `param_to_use->priority` 强制写成 0（`src/yolov5.cpp:256-269`），会丢弃调用方 priority，需在迁移审查中单独决定是否保留源行为或修复。
- S C++ main 的 default model 是 `/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm`，而 manifest 的下载相对路径是 `s100/yolov5x_672x672_nv12.hbm`；旧 run.sh 还会读 boardinfo、安装包和自动下载。统一 CLI 应在 identity gate 后解析明确 target，不把 board path 或 s100 fallback 当通用默认。

## Conversion、evaluator 和文档事实

| sample | 源 conversion 材料 | 可复用事实 | 当前缺口/风险 |
| --- | --- | --- | --- |
| yolov5 X5 | 两个 Bayes-e YAML：NCHW 与 NV12；含 input scale `1/255`、默认 calibration、O3。 | 640、三头输出、bayes-e、输出名 small/medium/big 的历史配方。 | 没有源内 export/calibration 脚本；S conversion README 仅“待补充”；X5 文档列 9 模型但 manifest 仅发布 1 个。 |
| fcos | 只有双语 README 和 3 张 hb_perf 截图。 | 三个输入尺寸和 15 输出协议。 | 没有 ONNX/export/PTQ YAML/calibration；`your_fcos_config.yaml` 不能视为可复现配方。 |
| yoloworld | 双语 README，仅记录 tensor protocol。 | 640、32×512 text embedding、8400 rows。 | 无 YAML、export、calibration、checkpoint；需要保留 offline vocabulary JSON。 |
| lprnet | 双语 README，仅 OE 命令模板。 | `1x3x24x94` float32 输入、`1x68x18` 输出。 | 无源模型 export、YAML、calibration、checkpoint，不能声称完整转换可重放。 |
| modnet | README 声称存在 `onnx_export/`、`ptq_yamls/`，但固定源 inventory 中这些目录/文件不存在。 | 官方 MODNet 链接、512 RGB F32 protocol。 | README 与源树矛盾；manifest manual 且无 URL，需手工 artifact 记录。 |
| bytetrack | 无神经网络 conversion；README 指向上游 YOLO/OE。 | detector 输入/输出需转换为 `(x1,y1,x2,y2,score,class_id)`。 | 不应复制 YOLO conversion 当作 tracker conversion；视频输入仍缺失。 |

各 sample 的 `evaluator/` 都是 README，未发现可运行 evaluator、数据集 harness 或基线结果生成器。现有性能表、截图和 `test_data` 图片只能作为历史参考，不能在统一迁移中宣称 evaluator 已实现或数值通过。源 run.sh 普遍隐含安装依赖和下载模型；ByteTrack 还隐含下载 `track_test.mp4`，这与统一 runtime 的 SDK-free `help/list/dry-run` 和显式资产准备边界冲突。

## 推荐统一边界和主机测试

1. **资产层**：用 manifest 的完整 asset facts（sample、target、march、format、filename、URL/hash）解析模型；X5/S 的 YOLOv5 与 ByteTrack 的同名 HBM 要求完整 pair identity。MODNet 只允许带人工确认 identity 的 external path；YOLOWorld vocabulary 与 LPR `.dat` 作为独立 input assets。
2. **runtime 层**：runner 懒加载 SDK，`--help`、`--list-models`、`--dry-run` 不读板卡、不下载；X5 packed NV12、S split NV12、YOLOWorld RGB+text、MODNet RGB、LPR prepacked float32 分别绑定。输出 shape/dtype/quant descriptor 取实际 metadata，未知时拒绝。
3. **task 层**：保持 `pre_process → forward → post_process` 纯阶段。YOLOv5/FCOS 解码、anchor/grid、FCOS sqrt confidence、NMS、LPR CTC 和 MODNet matte 必须做 source 数值 fixture；YOLOWorld prompt context、MODNet geometry context 每调用独立。ByteTrack 的 tracker state 是有意的 stream state，应由任务对象显式持有并测试 A/B/A 帧序。
4. **共享层**：第一消费者内保留 sample-local 协议；S YOLOv5 与 ByteTrack 稳定后可共享 detector core，但不要共享 X5/S tensor IO 或 C++ infer adapter。共享 `NV12` 只提供 packed/split 显式原语，不自动猜布局。
5. **最小主机验收**：每个 asset/target 的 metadata binding 与拒绝未知 shape；X5/S YOLOv5 输入和 dequant/raw 分支；FCOS 三族输出重排与 5 strides；YOLOWorld JSON prompt、空 prompt、32-slot mapping、A/B/A；LPR 字符表/blank/tie；MODNet 不同长宽 A/B/A；ByteTrack person filter、tracker persistence、缺视频显式错误；全部 CLI 的 help/list/dry-run 不加载 SDK。C++ 只做静态/板端 recipe 记录，未编译或未上板必须保持 not-run。

## 当前结论

B7 的真正迁移难点不是目录搬运，而是 YOLOv5 的 X5/S 物理输入与输出量化分叉、FCOS 的 metadata shape 族、YOLOWorld 的第二输入与 prompt 身份、LPR 的预打包二进制和字符解码、MODNet 的几何 context、以及 ByteTrack 的有状态 tracker 和缺失视频资产。当前 source inventory 已固定 136 个源文件；模型二进制均不在源树，发布 hash 以外均未知。没有代码迁移、转换重建、evaluator 执行或板端证据，因此 B7 仍应标记为 audit complete / refactor pending / host not-run / board not-run。
