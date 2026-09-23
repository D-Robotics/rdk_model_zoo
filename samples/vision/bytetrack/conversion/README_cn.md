# ByteTrack 转换

<a id="source-model"></a>
## 源模型

ByteTrack 本身是后处理，唯一神经制品是上游 YOLOv5x HBM，manifest 有三个 target-specific 行。固定 S 源没有独立 tracker 导出或 checkpoint。

<a id="toolchain-targets"></a>
## 工具链与目标

S100 对应 `s100/yolov5x_672x672_nv12.hbm` Nash-e，S100P 对应自己的 `s100p/...`，S600 对应自己的 `s600/...`。转换需在 x86 Linux 主机使用 RDK S OpenExplorer；源资源为 [OE overview](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) 和 [toolchain manual](https://toolchain.d-robotics.cc/)。sample 没有 S YAML 或 exporter。

<a id="export"></a>
## 导出

ByteTrack 没有导出步骤，因为 tracker 没有神经图。重建 detector 需按上游 YOLOv5 源流程，得到 672x672、split Y/UV 输入和三个输出 head 的 detector。固定源没有锁定 checkpoint/exporter，本轮没有导出。

<a id="calibration"></a>
## 校准

校准属于上游 detector 转换。本目录没有校准目录或生成器；`test_data` 静态图不是代表性校准集。

<a id="compile"></a>
## 编译

只有在准备好外部 ONNX/checkpoint、target-specific YAML 和校准集后，才能使用该环境提供的 OE 命令。固定 S 源没有可核对的完整命令/config，因此不写通用 `hb_mapper` 假装可复现。产物必须绑定三个 target-relative HBM asset ID 和 S split-NV12 metadata。

<a id="validation"></a>
## 转换后验证

先用 S YOLOv5 runtime 检查 detector metadata，再对准备好的视频运行 tracker。用 `evaluator/compare.py` 比较完整 detector tensor 和 track ID。本轮未运行导出、编译、板测或视频验证。

<a id="artifacts"></a>
## 产物

三个外部输出就是 `model/README_cn.md` 中的 manifest HBM 行；ByteTrack 没有独立编译 tracker 制品。`TRACKER_SOURCE_MAP.json` 记录 tracker 源文件 hash 以及唯一相对 import 改动。

<a id="known-gaps"></a>
## 缺失项

- 没有 S 导出脚本、checkpoint pin、YAML、校准生成器或编译日志。
- 源视频缺失，客户文档只保留显式 archive URL。
- 转换和板端验证为 `not-run`；发布 SHA-256 均未知。
