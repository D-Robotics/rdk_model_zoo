# YOLOv5 评估器

<a id="dataset"></a>
## 数据集

源提供 X5 的 `test_data/bus.jpg`、S 的 `test_data/kite.jpg` 以及 `coco_classes.names`，没有带标签的 mAP benchmark harness。评估器在相同图片、target、制品和阈值下比较完整的源/统一运行，是一致性证据工具，不是 mAP 评估器。

<a id="environment"></a>
## 环境

直接在已识别的目标板上运行，需要 Python、NumPy、OpenCV 和目标 `hbm_runtime`，并导入固定源 runtime；它不是一台在外部驱动板卡的主机。主机单测注入 fake runtime，不能证明硬件或板端结果。评估器不会下载模型或图片。

<a id="command"></a>
## 评估命令

在仓库根目录、准备好精确模型且板卡身份已识别后，选择一个全新的空证据目录：

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

工具运行源路径和统一路径，把完整 native 输入/输出/结果数组保存为 `.npy`，在 `comparison.json` 记录 metadata、代码/模型/图片 hash、阈值和板卡身份；所有比较通过才返回 `0`。输出目录必须不存在。本轮没有板端运行。

### 原生 C++ 源/统一比较（板端）

上面的 Python 命令驱动 Python runtime，不能作为 C++ 交付的最终证据——两者的前处理可能不同。`evaluator/native/` 下的原生比较运行**固定 C++ 源本身**（X5 `main.cc` @
`ac115717197920355fc390bb04299b20e6436864`，S `src/yolov5.cpp`/`src/main.cpp` @
`380e1a2bf42041af54be6f34935e50197cfadff9`，SHA 固定、失配即拒绝），以插桩副本
方式只加入**只读观测点**；原有预处理/推理/解码/NMS 不动，绝不把统一 decoder
当作 legacy。两侧在同一板卡、同一模型、同一图片上各自独立完成推理。

在目标板上（仓库已检出、模型已下载；X5 默认
`yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin` + `test_data/bus.jpg`，S 默认
x-672 资产 + `test_data/kite.jpg`；两侧阈值一致）：

```bash
# 0) 固定快照必须先取到本地。板上浅克隆不含这些提交；先 fetch（否则
#    instrument.py 会带同样提示 fail closed）：
git fetch --depth=1 origin ac115717197920355fc390bb04299b20e6436864   # x5
git fetch --depth=1 origin 380e1a2bf42041af54be6f34935e50197cfadff9   # s100/s600

# 1) 生成插桩固定源构建（SHA/anchor 失配即拒绝；审计见
#    instrumentation-audit.json——S 的构建闭包（yolov5.hpp + utils/c_utils）
#    一并 SHA 固定并复制进 work dir，构建不引用任何可变工作树文件）。
#    X5 重绑定示例：
python3 samples/vision/yolov5/evaluator/native/instrument.py \
  --target x5 --repo-root . --work-dir /tmp/yolov5-fixed-src \
  --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image-path samples/vision/yolov5/test_data/bus.jpg
#    （S100/S600 在此都传 --target s100，选择共同的 S 源代码；
#    统一入口构建和比较时必须选择实际板卡目标。）

# 2) 编译插桩固定源，经外部 runner 运行。runner 记录真实进程证据（含被
#    gflags 删除参数在内的完整 argv、分离的 stdout/stderr、真实退出码、
#    起止 UTC、cwd、板卡身份、binary/model/image 运行前后 hash、插桩 audit
#    校验）。C++ 观测头只记录进程外不可得的张量/阶段数据，不再 tee
#    stdout/stderr、不再冒充进程退出码。捕获目录必须为空；任何崩溃都会
#    留下 in-progress 标记，比较阶段据此拒绝：
cmake -S /tmp/yolov5-fixed-src -B /tmp/yolov5-fixed-src/build && \
  cmake --build /tmp/yolov5-fixed-src/build -j2
python3 samples/vision/yolov5/evaluator/native/run_capture.py \
  --binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --capture-dir /tmp/yolov5-source-capture \
  --model <模型文件> --image <图片文件> \
  --audit /tmp/yolov5-fixed-src/instrumentation-audit.json \
  -- [--model_path <m> --test_img <i> --label_file <l>]   # S 的参数原样透传

# 3) 统一二进制也经同一 runner 运行（--role unified：统一侧同样获得完整
#    进程证据——真实 argv/退出码与自己的 binary/model/image hash；该角色无
#    插桩 audit）。构建时 -DYOLOV5_TARGET=s100/s100p/s600 须与板卡一致。
python3 samples/vision/yolov5/evaluator/native/run_capture.py --role unified \
  --binary <build>/yolov5_cpp --capture-dir /tmp/yolov5-unified-run \
  --model samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
  --image samples/vision/yolov5/test_data/bus.jpg \
  -- --target x5 --test-img samples/vision/yolov5/test_data/bus.jpg \
     --model-path samples/vision/yolov5/model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin \
     --dump-dir /tmp/yolov5-unified-dump

# 4) 比较两次同板运行（--target 必须与统一 build_target 完全一致——s100
#    比较拒绝 s100p/s600 构建；s600 构建用 --target s600 比较；两侧都需要
#    各自的 runner 记录）：
python3 samples/vision/yolov5/evaluator/native/compare_native.py \
  --target x5 --repo-root . \
  --source-capture /tmp/yolov5-source-capture --unified-dump /tmp/yolov5-unified-dump \
  --source-binary /tmp/yolov5-fixed-src/build/yolov5_fixed_capture \
  --unified-binary <build>/yolov5_cpp \
  --unified-run-record /tmp/yolov5-unified-run/run-record.json \
  --output /tmp/yolov5-native-comparison
```

比较阶段在任何数值比较之前校验**两侧**的 runner 记录（真实退出码、argv、
cwd、UTC、binary/model/image 运行前后 hash 一致；统一侧记录与 manifest
binary hash 绑定）与观测头捕获时 hash、被背书二进制、统一 manifest 自带
hash 与逐 payload digest 的一致性。板卡身份采用仓库**精确别名注册表**
（docs/release/platforms.json，与 samples/_shared/platforms.py 同一契约）：
S100P 与 s100 是不同 target，S100Whatever 等未知字符串不构成身份，X5 板经
socinfo 名（X5U/X5H/X5M）解析。插桩 audit 校验失败时 run_capture.py **根本
不运行**源二进制——先中止并持久化失败记录；audit 校验按完整协议执行
（schema、target、固定 commit、精确固定源集合且每个 anchor 恰匹配一次、
按 target 完整的闭包、观测头与 CMake hash）。结构要求
在**两侧**强制——目标对应的输入数、恰好三个形状唯一的输出头与完整双射、
dtype 与量化类型/axis/scale 长度一致（绝不被浮点转换掩盖）、值有限、统一
manifest 逐 payload 必填 bytes+SHA。布局解码额外拒绝通道 stride 重叠
（stride[3] < itemsize）或未按 itemsize 对齐、以及像素 stride 小于
channels*stride[3] 的布局，同时仍解码真实受支持形态（含按对齐 stride 的通道
padding）。阈值与 scale 描述符按 **float32 位值**
比较（原生语义）：0.45f 序列化为 0.44999998807907104 与 manifest 字符串
"0.450000" 是同一个值；位值不同即失败。最终原图坐标（detections_original）
**双侧必备**——缺失或单侧缺失直接阻断验收，而不是以免责声明放行。布局
解码只接受两种已证实形态（均匀行距 strided、精确尺寸 compact），其余显式
拒绝，绝不猜着解码。

比较按两侧记录的物理布局（stride/dtype）还原逻辑数组，因此 padded 布局正确
比较，而未初始化 padding 字节保留在证据（`originals/`）中、绝不声称逐字节
相等。固定判据、运行中不得调整：inputs 精确相等；raw 输出
`allclose(atol=1e-5, rtol=0)`；scale/zero-point 描述符精确相等；boxes
`atol=1e-4`、scores `atol=1e-5`、class id 精确相等，比较前做已声明的排序
归一（按 class_id、score、x1..y2）。检测在**模型输入空间**比较，并**强制**比较最终原图坐标（双侧
`detections_original`——捕获端与本轮起的统一 dump 都会产出）；最终坐标缺失
或单侧缺失直接判整个比较失败，而不是以免责声明放行。任何材料缺失、运行
非零返回码、模型/图片 hash 或阈值不一致都以非零返回码失败并保留已收集
证据；native 失败绝不能以空数组通过。**板端状态：本轮 not-run；上述步骤由协调者在真实板卡执行。**
主机测试覆盖插桩生成（含固定闭包与浅克隆准备提示）、插桩源的 stub 编译
（只证明注入代码可编译，不代表真实 SDK 构建通过）、观测头精度/标记/拒绝
行为、对**真实 v2 板端 manifest schema** 的比较、stride 还原与全部失败模式。

<a id="metrics"></a>
## 指标

输入要求精确一致；raw 输出检查 shape/dtype，使用 `rtol=0, atol=1e-5`；结果 boxes 使用 `atol=1e-4`，scores `1e-5`，class IDs 精确相等。X5/S 必须各自按源协议比较，host fake fixture 不等于板测。历史性能列于下方，不是本轮测量。

<a id="outputs"></a>
## 输出

每个 run 目录含 `legacy_*`、`unified_*` `.npy` 数组和 `comparison.json`。模型预加载失败或比较不一致会保留 error/failed 记录并返回非零，不会把 mismatch 改成 pass。数组保留全部捕获的输入/输出 tensor 和解码结果字段。

<a id="reference-results"></a>
## 参考结果

下表完整保留源 X5 历史数据，本轮没有复测：

| 模型 | 尺寸 | 参数量 | BPU 吞吐 | Python 后处理 |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

S100/S600 以及当前源/统一板端比较均为 `not-run`。

<a id="boundaries"></a>
## 边界

评估器不下载模型、不构建转换产物，也不会把主机测试写成板端兼容。X5 刻意保留源 OpenCV XYXY-to-NMSBoxes quirk，S 使用按类 XYXY NMS；不能跨 target 要求结果相等。

源端 runner 将运行前实际校验的审计字节归档为 `instrumentation-audit.json`，并在运行记录保存 `audit_file`、`audit_sha256`。副本在子进程结束后写入，以满足观测头要求输出目录初始为空的约束。请保留完整源捕获目录和统一入口进程记录目录：审计缺失或内容变化、统一入口 stdout/stderr 缺失均会拒绝比较；成功输出的 `originals/` 包含审计与双方日志。旧捕获若没有审计绑定，不能通过该检查；仅有校验通过布尔值不能代替被校验的文档。
