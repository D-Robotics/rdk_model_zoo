[English](README.md) | 简体中文

# Faster R-CNN model conversion

<a id="source-model"></a>
## 源模型

[TorchVision v0.21.0](https://github.com/pytorch/vision/tree/v0.21.0/torchvision/models/detection) 的 `fasterrcnn_mobilenet_v3_large_320_fpn`，权重为 `COCO_V1`。构造模型时使用 COCO 权重与 `weights_backbone=None`。`export_split.py` 将[官方权重](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth)下载到 Torch Hub 缓存，SHA-256 为 `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`。

<a id="directory"></a>
## 目录结构

```text
export_split.py              # 导出 PartA 与两种 PartB 布局
prepare_calibration.py       # 图像及 ROI 特征校准数据
recorded_calibration_urls.txt # COCO 图像地址与哈希
make_configs.py              # 八份板卡/阶段编译配置
compile_models.sh            # X5 或 S 系列编译
verify_pair.py               # 文件哈希比对
```

<a id="toolchain-targets"></a>
## 工具链

| Stage | Image / version |
| --- | --- |
| Export and calibration | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`; torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 1.23.0, ONNX 1.15.0, ONNX Runtime 1.19.0, Pillow 9.3.0 |
| X5 compile | `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`; hb_mapper 1.24.3, HBDK 3.49.15 |
| S compile | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`; HBDK4 4.7.5 |

通过 [X5 指南](https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/environment_config)与 [S 指南](https://developer.d-robotics.cc/rdk_s_doc/en/Advanced_development/toolchain_development/algorithm_toolchain/overview)获取镜像，在 x86_64 Linux 上运行 Docker。将仓库挂载到 `/work`，下面的命令从宿主机仓库根目录执行。

<a id="export"></a>
## 导出 ONNX

在 CPU 容器中执行导出和校准：
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash -lc '
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu &&
    python3 -m pip install numpy==1.23.0 onnx==1.15.0 onnxruntime==1.19.0 Pillow==9.3.0 &&
    python3 export_split.py &&
    python3 prepare_calibration.py --images ../test_data/bus.jpg ../../yolov5/test_data/kite.jpg --coco-urls recorded_calibration_urls.txt --hflip repo
  '
```
导出产生 `onnx/part_a_f0only.op11.onnx`、`part_a_f0only.op16.onnx`、`part_b_batch150.op11.onnx`、`part_b_packed.op16.onnx` 及 `export_report.json`。部署画布为 320×320，等比例缩放后在右侧、底部补零。PartA 输出 RPN 分数、回归量及 feat0，PartB 接收补齐到 150 个的 ROI 特征。

<a id="calibration"></a>
## 校准数据

命令使用仓库中的 `bus.jpg`、`kite.jpg` 及其水平翻转图，再加地址表中的 18 张 COCO 图片，共 22 个样本。`prepare_calibration.py` 校验地址表中的图片哈希，并生成 `calib/manifest.json`。

| 目录 | 数据 |
| --- | --- |
| `calib/parta_x5` | 无文件头的 float32 RGB CHW，值域 [0,255] |
| `calib/parta_s` | 已归一化的 float32 NumPy 图片数组 |
| `calib/partb` | Float32 ROI 特征，150×256×7×7 |
| `calib/partb_packed_s` | Float32 ROI 特征布局，1×256×1050×7 |

使用自己的图片时，在导出容器中执行 `python3 prepare_calibration.py --images /work/calibration_images --hflip all`。

<a id="compile"></a>
## 编译板端模型

X5：
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 bash -lc \
  'python3 make_configs.py && bash compile_models.sh x5'
```
S100、S100P、S600：
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash -lc \
  'python3 make_configs.py && bash compile_models.sh s'
```
`make_configs.py` 生成 `build/<target>/<part>/config.yaml`。PartA 使用 NV12 输入、BT.601 视频值域，RGB 均值为 `123.675 116.28 103.53`，缩放系数为 `0.0171247538317 0.0175070028011 0.0174291938998`。X5 使用 O3/latency/单核，S 系列使用 O2/latency/单核。PartB 接收 float32 特征，在模型图内量化；X5 的输入为 `[150,256,7,7]`，S 系列为 `[1,256,1050,7]`，输入、输出不补齐。

<a id="validation"></a>
## 查看并运行产物

在仓库根目录执行：
```bash
python3 samples/vision/fasterrcnn/conversion/verify_pair.py x5
```
命令输出 SHA-256，并与 `model/frcnn_mbv3_320.json` 比对：返回 0 表示两个记录哈希匹配，1 表示文件缺失，2 表示哈希不同。自行编译的模型在运行时同时指定两个文件路径。将文件复制到板卡后执行：
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```

<a id="artifacts"></a>
## 产物

| Target | march | PartA | PartB |
| --- | --- | --- | --- |
| x5 | bayes-e | `build/x5/parta/frcnn_mbv3_320_parta.bin` | `build/x5/partb/frcnn_mbv3_320_partb.bin` |
| s100 | nash-e | `build/s100/parta/frcnn_mbv3_320_parta.hbm` | `build/s100/partb/frcnn_mbv3_320_partb.hbm` |
| s100p | nash-m | `build/s100p/parta/frcnn_mbv3_320_parta.hbm` | `build/s100p/partb/frcnn_mbv3_320_partb.hbm` |
| s600 | nash-p | `build/s600/parta/frcnn_mbv3_320_parta.hbm` | `build/s600/partb/frcnn_mbv3_320_partb.hbm` |
路径相对于 `conversion/`，部署到板卡时将两个文件放到 `model/<target>/`。

<a id="known-gaps"></a>
## 自备校准图片

将 JPEG/PNG/BMP 图片放入容器挂载目录，通过 `--images` 指定，`--hflip` 可选择 `repo`、`all` 或 `none`。先生成四份校准数据目录，再执行 `make_configs.py` 和所选系列的编译命令。

