# YOLO 检测模型转换

[English](README.md)

本目录把 Ultralytics 浮点模型导出为 ONNX，再编译为共用 Python Sample
使用的目标 BPU 制品。导出和编译在主机执行，不能在板端运行。此次合并的
代表范围是 YOLOv8/YOLO11 的 DFL 检测和 YOLO26 的直接 LTRB 检测；分割、
姿态、分类和 OBB 的原有任务脚本仍由旧选择器保留，本指南不新增这些任务。

## 准备两个主机环境

第一步使用 Ultralytics 训练/导出环境，需提前准备 `ultralytics`、PyTorch
以及该权重所需的 ONNX 导出依赖。请先确认本地权重路径再调用脚本。脚本
直接调用 `ultralytics.YOLO`；Ultralytics 对已知的裸模型名可能自动下载，
因此裸名称不能证明使用了本地权重。本仓库不把这种隐式下载记录为来源或
验证；需要可复现时应传入已经存在的绝对 `.pt` 路径。可以在本目录运行，
也可以给权重和输出传绝对路径。

第二步使用对应的 D-Robotics OpenExplore 工具链环境：

| 目标 | 工具检查 | 制品 | 标定文件 | 编译配置中的输入约定 |
| --- | --- | --- | --- | --- |
| X5 | `hb_mapper --version` | `.bin` | 原始 float32 `.rgbchw` | `bayes-e`，运行时 `nv12` |
| S100 | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-e`，运行时 `nv12` |
| S100P | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-m`，运行时 `nv12` |
| S600 | `hb_compile --help` | `.hbm` | 除以 255 的 float32 `.npy` | `nash-p`，运行时 `nv12` |

Mapper 会检查工具链是否已加载、ONNX 是否只有一个静态
`tensor(float)` 四维输入，以及标定目录中是否有 JPG/JPEG/PNG。缺依赖时
只报错，不会静默安装。工具链环境和板端运行时镜像是两套环境，不要把
`hb_mapper` 或 `hb_compile` 安装步骤放入板端部署流程。

X5 可在 x86 Linux 主机使用本仓库此前记录的 OE 1.2.8 CPU 镜像。若发布
版本不同，以 D-Robotics 提供的镜像和 tag 为准；镜像来源为
[X5 工具链包](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz)：

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

S 系列请从 [RDK S OE 工具链文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)
或[工具链下载页](https://toolchain.d-robotics.cc/)选择与 S100/S100P 或 S600
匹配的镜像，再加载和挂载：

```bash
docker load -i ai_toolchain_ubuntu_22_s100_<release>.tar
docker images
docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace --workdir /workspace \
  <loaded-s100-or-s600-image>:<tag> /bin/bash
```

可参考 [S100/S100P 工具链页](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=4.0.5&p=RDK+S100)
和 [S600 工具链页](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=5.1.0&p=RDK+S600)。
`-v /workspace` 会把权重、ONNX、标定图片和输出目录暴露给容器；进入容器
后在 `/workspace` 运行下方 Mapper 命令。

## 导出 ONNX

通用 Ultralytics 检测器通过 `--platform` 选择已核定的 opset 默认值：X5
使用 opset 11，S 系列使用 opset 19。`--opset` 与历史拼写 `--optse` 等价，
显式传入的值优先。

```bash
cd samples/vision/ultralytics_yolo/conversion

# YOLOv8/YOLO11 风格 DFL 检测器，导出给 X5。
python export_monkey_patch.py \
  --platform x5 --pt /models/yolo11n.pt --require-local --opset 11

# 同一模型系列，导出给 S600。
python export_monkey_patch.py \
  --platform s600 --pt /models/yolo11n.pt --require-local --opset 19
```

通用导出器通过 `ultralytics.YOLO` 加载权重，在模型头应用 Sample 的 BPU
monkey patch，然后调用 `model.export(format='onnx', simplify=False,
opset=...)`。Ultralytics 通常把 ONNX 写在权重旁边；启动 Mapper 前请确认
实际路径，Mapper 不会猜测或替换相似文件。

YOLO26 检测器使用单独的图导出器，因为它输出三个 NHWC 分类张量和三个
四通道直接 LTRB 张量。默认值为 X5 `opset=11, simplify=1`，S 系列
`opset=19, simplify=0`。

```bash
python export_monkey_patch.py \
  --family yolo26 --task detect --platform x5 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx

python export_monkey_patch.py \
  --family yolo26 --task detect --platform s600 \
  --pt /models/yolo26n.pt --require-local \
  --output /models/yolo26n_det_bpu.onnx
```

也可以直接运行 `yolo26/export_yolo26_detect_bpu.py`，参数为 `--weights`、
`--output`、`--imgsz`、`--platform`、`--opset`、`--simplify` 和可选的
`--require-local`。导出失败会
返回错误，不会在文档中把一个未生成的路径当成制品。

## 准备标定并编译

在标定目录放入 20–50 张有代表性的输入图片。共用流程会把 BGR 转为 RGB，
缩放到静态 ONNX 宽高，HWC 转 NCHW，并写入 float32。X5 写入未除 255 的
RGB 张量 `*.rgbchw`；S 写入除以 255 后的 NumPy `*.npy`。这是编译器标定
格式；运行时的 NV12 打包仍由目标板输入绑定负责。

从仓库根目录运行统一入口（历史平台入口会补上相同的 platform）：

```bash
# X5 -> hb_mapper makertbin、bayes-e、.bin
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform x5 \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100 -> hb_compile、nash-e、.hbm
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s100 --march nash-e \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled

# S100P/S600 分别使用 nash-m/nash-p。
python samples/vision/ultralytics_yolo/conversion/mapper.py \
  --platform s600 --march nash-p \
  --onnx /models/yolo11n.onnx \
  --cal-images /datasets/calibration \
  --output-dir /models/compiled
```

Mapper 成功后，把对应目标的制品复制到板端，用完整路径运行共用运行时。
例如上面的 S600 命令生成 `/models/compiled/yolo11n_nashp_640x640_nv12.hbm`：

```bash
# 在 S600 板卡上从仓库根目录运行
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/compiled/yolo11n_nashp_640x640_nv12.hbm \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolo11n-s600.jpg
```

X5 使用 `--platform x5` 和 `.bin` 文件名；其他 Nash 目标使用对应的
`--platform s100`/`s100p` 与 `.hbm` 文件名。编译成功不能跳过运行时的
输入/输出绑定，板端仍会在推理前检查制品元数据及 DFL 或直接 LTRB 契约。

需要权重必须已经存在时传 `--require-local`；省略该参数则保留 Ultralytics
解析已知裸模型名的历史能力。

通用 YOLO 和 YOLO26 mapper 现在调用同一个 `conversion/workflow.py`。其中
`mapper_x5.py`、`yolo26/mapper_x5.py` 选择 X5 配置；`mapper_s.py`、
`yolo26/mapper_s.py` 选择 Nash 配置并保留 `--march`。共用流程仍明确保留
以下目标差异：

* X5 执行 `hb_mapper makertbin --config config.yaml --model-type onnx`，
  使用 `bayes-e` 和原始 RGB/NCHW 标定；保留 Softmax int8 优化，
  `--quantized int16` 时追加 `set_all_nodes_int16`。
* S 执行 `hb_compile --config config.yaml`，使用选择的 Nash march、归一化
  NumPy 标定，并加入 `input_no_padding`、`output_no_padding`；
  `--quantized int16` 时加入 S 的 `quant_config` 模型/输出类型设置。

配置写在 `--ws` 指定目录下本次创建的唯一子目录中（`--ws` 是工作区父
目录）。成功后默认删除该子目录，`--save-cache` 会保留它。编译失败会保留
子目录中的配置、标定文件和日志以便排查。已有同名制品必须显式传
`--overwrite` 才会替换；不会删除用户提供的工作区父目录、输入模型、标定
图片目录或输出目录。

## 输出名称和路径

当 ONNX 文件名为 `yolo11n`、输入为 640×640 时，默认输出目录是 ONNX 所在
目录，名称如下：

```text
X5:    yolo11n_bayese_640x640_nv12.bin
S100:  yolo11n_nashe_640x640_nv12.hbm
S100P: yolo11n_nashm_640x640_nv12.hbm
S600:  yolo11n_nashp_640x640_nv12.hbm
```

`--output-dir` 只改变最终制品和编译日志位置。X5 成功时可能留下
`hb_mapper_makertbin.log`，S 成功时可能留下 `hb_compile.log`。使用
`--save-cache` 时，在输出的唯一工作区子目录检查 `config.yaml`、标定目录
和 `bpu_model_output/`。

## 代码流程和兼容符号

维护入口保持为：

```text
export_monkey_patch.py
  ├─ 通用检测器 -> 共用补丁 + Ultralytics 导出
  └─ YOLO26 detect -> yolo26/export_yolo26_detect_bpu.py

mapper.py -> mapper_x5.py / mapper_s.py
          -> --family yolo26 时的 yolo26/mapper_x5.py / mapper_s.py
          -> workflow.py
             inspect_onnx -> calibration_images/select_calibration_images
             -> prepare_calibration -> render_config -> 编译 -> 移动制品/日志
```

`platforms/x5/...` 和 `platforms/s/...` 下的历史路径仍是转发入口，保留旧
类名和命令默认值，不再含有第二套标定或编译实现。Mapper dispatcher 继续
接受既有 `--family`，所以旧分割、姿态、分类命令不会被改送到检测协议。

YOLOv8/YOLO11 检测使用三层 DFL 输出契约；YOLO26 检测使用直接 LTRB，两者
不能互换。导出或编译成功不能证明自定义图满足运行时协议；运行时会在板端
推理前检查输入输出形状和类型。有限的运行时检测协议见
[`DETECTION_CONTRACT.md`](../DETECTION_CONTRACT.md)，转换旧新符号对应及
目标适配边界见 [`CONVERSION_CONTRACT.md`](CONVERSION_CONTRACT.md)。

## 故障排查

* **找不到 `hb_mapper` 或 `hb_compile`：** 加载对应 OpenExplore 环境，先
  执行表中的工具检查命令。板端镜像不是编译器环境。
* **ONNX 是动态尺寸、非 float32 或多输入：** 重新导出静态 batch-one、
  NCHW 图。输入契约不明确时 Mapper 会在写标定前停止。
* **没有标定图片：** 直接在 `--cal-images` 中放入 JPG/JPEG/PNG；其他后缀
  会忽略。`--cal-sample false` 保留全部图片，或调整 `--cal-sample-num`。
* **`--platform` 与 `--march` 冲突：** X5 不要传 Nash march；S100/S100P/S600
  分别配 `nash-e`/`nash-m`/`nash-p`。
* **已有制品：** 选择新的 `--output-dir`，或确认替换目的后显式传
  `--overwrite`。
* **编译失败：** 使用 `--save-cache` 重跑，检查唯一工作区的 `config.yaml`、
  标定文件和编译器日志。默认失败工作区也会保留。

本地合并包含规划、配置、入口和运行时契约的主机测试；本地没有执行真实
ONNX 导出或 OpenExplore 编译。板卡验证记录引用发布证据中的既有运行时制品，
不能被解释为本地这次转换已经执行。
