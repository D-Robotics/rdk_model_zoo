# YOLOv5 转换

<a id="source-model"></a>
## 源模型

固定 X5 源说明 Ultralytics YOLOv5 `v2.0` 与 `v7.0` 分支及匹配权重，来源为 [v2.0](https://github.com/ultralytics/yolov5/tree/v2.0) 和 [v7.0](https://github.com/ultralytics/yolov5/tree/v7.0)。源没有锁定 commit，也没有完整 exporter 仓库；下面的分支/权重配对和检测头修改是源说明，本轮没有执行转换。

<a id="toolchain-targets"></a>
## 工具链与目标

两个 YAML 是 X5 Bayes-e 640x640 配置，保留 `O3`、latency 模式、default calibration 和 `scale_value: 0.003921568627451`。固定源没有 S100/S600 Nash-e 或 S100P Nash-m YAML；S conversion README 也明确是未完成内容。不能把 X5 YAML 套到 S HBM。

<a id="export"></a>
## 导出

v2.0 源要求修改 `models/yolo.py` 使 head 输出 NHWC，输出名为 `small/medium/big`，ONNX opset 11，可选简化；对应 v2.0 checkout 和 `yolov5s_tag2.0.pt`。v7.0 同样修改 NHWC head，使用 ONNX-only、opset 11，并配 `yolov5n.pt`/tag-v7。仓库没有 exporter 或 checkpoint；以下只是源流程，本轮未执行：

```bash
# cwd：外部 clone，不是本 sample 目录
python3 -m pip install -r requirements.txt
# 按上文修改 models/yolo.py，准备匹配的 v2.0 或 v7.0 checkpoint
python3 export.py --weights yolov5n.pt --img-size 640 640 --opset 11 --include onnx
# 输出：带 small/medium/big head 的外部 ONNX
```

<a id="calibration"></a>
## 校准

逐字节 YAML 指向 `./calibration_data_rgb_f32_coco_640`，并声明 `cal_data_type: float32`、`calibration_type: default`。源 sample 没有校准数据生成器；该目录和 COCO tensor 是外部前置，不能把随附推理图当校准集。

<a id="compile"></a>
## 编译

在 `samples/vision/yolov5/conversion` 中，把匹配的外部 ONNX 和校准目录放到所选 YAML 指定路径后：

```bash
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx/yolov5n_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config ./yolov5_detect_bayese_640x640_nv12.yaml
```

预期产物是 YAML 前缀下的 `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`。命令需要 OE，本轮未运行。NCHW YAML 作为源参考保留；其 runtime 输入声明是 NCHW，而发布的 X5 runtime 路径是 NV12，必须有意选择配置。

<a id="validation"></a>
## 转换后验证

在 OE 主机/板端环境检查模型和吞吐：

```bash
hb_perf ./yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file ./yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```

成功条件是三个 `(1,H/stride,W/stride,255)` head、640 输入 metadata 与所选目标路径的输出行为一致。本轮没有转换或板测。

<a id="artifacts"></a>
## 产物

`yolov5_detect_bayese_640x640_nchw.yaml` 和 `yolov5_detect_bayese_640x640_nv12.yaml` 从 `platforms/x5/samples/vision/yolov5/conversion/` 逐字节复制，是源配置参考，不代表本地构建成功。发布 runtime 制品见 `model/README_cn.md`。

<a id="known-gaps"></a>
## 缺失项

- 没有锁定上游 commit、checkpoint、exporter、校准生成器或 S 转换配方。
- X5 Python 与 S 的物理 tensor 协议、NMS 和反量化不同，不能用一段共享转换说明替代 target-specific metadata。
- 编译、导出、校准和板测均为 `not-run`；manifest 发布 SHA-256 未知。
