[English](README.md) | 简体中文

# 固定权重版本的 MobileNet 主机流程

本工具支持固定版本权重下载、batch=1 FP32 ONNX 导出、分平台校准数据生成、OE 配置生成和全量分类评测。
四代 sample 分别提供 `conversion/export.py`、`evaluator/evaluate.py` 入口。
在仓库根目录执行命令；模型、数据集和运行结果保存到仓库外，每次使用新的输出目录。

## 模型和环境

`checkpoints.json` 固定六个 checkpoint 的 revision、文件 SHA256、模型卡和许可证：
`v1-100`、`v2-100`、`v3-large-100`、`v4-small`、`v4-medium-224`、`v4-medium-256`。
Medium-224 对应全部四个平台，Medium-256 对应 S100/S100P/S600；其他规格的候选矩阵包含四个平台。
矩阵记录目标范围，板端支持以实际转换和验收为准。

使用 Python 3.10 和 `requirements.lock.txt`。先按
[PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v280)
安装 PyTorch 2.8.0 CUDA 12.8 wheel，再安装此锁文件；同一环境可以进行 CPU ONNX 评测。
每个 campaign 保存完整 `python -m pip freeze`，本机环境同时记录继承的基础环境和 wheel 安装报告。

```bash
python -m pip install -r utils/tools/mobilenet/requirements.lock.txt
```

## 权重与预处理

权重清单记录上游默认配置和显式部署合同。V1 的 mean/std 为 0.5，其他模型采用各自记录的 ImageNet 归一化。
使用 PIL bicubic 短边缩放后中心裁剪；尺寸与 `crop_pct` 显式指定，上游 `test_input_size` 不自动覆盖部署尺寸。

新 ONNX 输入为归一化 RGB float32 NCHW `[1,3,H,W]`，输出为 `[1,1000]` logits。
类别顺序采用 timm `ImageNetInfo(ilsvrc2012)` 的 0–999 索引。
历史 manifest 制品可能采用其他来源、预处理、形状和概率输出，需要按各自合同使用。

```bash
python utils/tools/mobilenet/workflow.py fetch --model v4-small \
  --output /absolute/workbench/weights/v4-small

python samples/vision/mobilenetv4/conversion/export.py \
  --model v4-small --source-dir /absolute/workbench/weights/v4-small \
  --images samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
           samples/vision/mobilenetv4/test_data/zebra_cls.jpg \
  --opset 11 --simplify --output /absolute/workbench/run/export
```

导出器验证权重哈希，离线严格加载 safetensors，检查 ONNX，并对比真实图片的 logits 数值和 Top-5 排序。
`export.json` 保存图哈希、输入形状、环境、源码哈希、参数量和数值结果。
Conv/Linear MACs 仅统计这两种算子，不能直接作为模型总 GFLOPs。

## 数据清单与冻结

评测 manifest 包含 `dataset`、`count`、`class_count: 1000` 和有序 `images` 数组；
每项为 `{path, sha256, label_id}`。标签采用数值类别 ID，不使用目录名字典序编号。
校准记录允许以 `name` 替代 `path`，无需标签。所有图片逐项验证哈希。

`freeze.py` 面向 ImageNetV2 10,000 图和 COCO 200 图 campaign：验证全部输入、每类十张评测图，
并检查校准/评测哈希无交集。输出目录保存两份 manifest、权重清单、完整环境锁、工具链镜像身份、
类别映射、campaign 和 SHA256SUMS。

```bash
python utils/tools/mobilenet/freeze.py \
  --source-root /absolute/workbench/inputs/source_models/mobilenet \
  --evaluation-manifest /absolute/workbench/imagenetv2/manifest.json \
  --evaluation-root /absolute/workbench/imagenetv2/images \
  --calibration-manifest /absolute/workbench/calibration/manifest.json \
  --calibration-root /absolute/workbench/calibration \
  --environment-lock /absolute/workbench/environment.lock.txt \
  --toolchains /absolute/workbench/toolchains.json \
  --output /absolute/workbench/campaign
```

源权重目录为 `<repo-name>/<revision>/{config.json,README.md,model.safetensors}`。
`toolchains.json` 记录实际 Docker tag 和不可变 image ID。
campaign 的 Top-1/Top-5 各下降最多一个百分点是本次候选门槛，并非仓库统一标准；
修改冻结输入应新建 campaign。COCO 校准集是否适用，由后续独立全量分类评测验证。

## 校准和 OE 配置

校准与评测共用相同 RGB 裁剪。X5 OE v1.2.8 使用 [0,255] 原始 RGB float32 NCHW，
OE 加载器负责 RGB/NV12 变换；S OE v3.7.0 使用归一化后的原 ONNX 输入 float32 NCHW。
X5 文件为无文件头的 little-endian float32 `.rgb`，S 文件为 `.npy`；X5 加载器不支持 NumPy 文件头。
两边 YAML 均声明 RGB 训练输入、NV12 运行输入，均值为 mean×255，缩放为 1/(255×std)。
不要对 X5 校准数据重复归一化。

```bash
python utils/tools/mobilenet/workflow.py calibrate --model v4-small \
  --platform x5 --manifest /absolute/workbench/calibration/manifest.json \
  --images-root /absolute/workbench/calibration --expected-images 200 \
  --evaluation-manifest /absolute/workbench/imagenetv2/manifest.json \
  --output /absolute/workbench/run/x5/calibration

python utils/tools/mobilenet/workflow.py config --model v4-small \
  --platform x5 --export-dir /absolute/workbench/run/export \
  --calibration-dir /absolute/workbench/run/x5/calibration \
  --output /absolute/workbench/run/x5/config
```

S 试点将两条命令的平台改为 `s100` 并使用独立输出目录。配置回执包含 march 和编译命令。
Docker 中保持输入/输出绝对路径与宿主机一致。在对应镜像内只检查配置与 200 份数据加载：

```bash
python3 /absolute/repository/utils/tools/mobilenet/check_toolchain.py \
  --platform x5 --config /absolute/workbench/run/x5/config/config.yaml
```

校准 batch 使用工具链默认值，其实际数值需要从编译日志核实。本工具不会应用 YOLO 图改写。

## 全量 FP32 ONNX 评测

```bash
python samples/vision/mobilenetv4/evaluator/evaluate.py \
  --model v4-small --export-dir /absolute/workbench/run/export \
  --manifest /absolute/workbench/imagenetv2/manifest.json \
  --images-root /absolute/workbench/imagenetv2/images --expected-images 10000 \
  --provider CPUExecutionProvider --threads 4 \
  --output /absolute/workbench/run/evaluation
```

评测器验证完整清单和 ONNX 身份，一次加载模型，逐图保存预测，输出 [0,1] 范围的 Top-1/Top-5。
回执记录真实 ORT provider；请求的 CUDA provider 不可用时失败。
当前基线是 RGB FP32，不含 NV12 往返。后续板端评测应调用 `prepare_rgb`，将连续 BGR 裁剪图
传入相同尺寸、`resize_type=0` 的 sample 分类器，明确记录 NV12 与量化带来的差异。
这套新 checkpoint 合同不采用历史 CLI 的默认 letterbox。

## 主机检查

```bash
python -m unittest utils.py_utils.tests.test_classification_host -v
python -m unittest discover -s utils/tools/mobilenet/tests -v
```

检查覆盖 timm 预处理一致性、归一化域、标签、样本数、数据交集、输出排序和制品身份。
Runtime/C++ 性能和正式发布状态由后续板端与发布流程验证。
