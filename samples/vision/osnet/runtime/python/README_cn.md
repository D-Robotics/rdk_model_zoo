[English](README.md) | 简体中文

# OSNet Python 特征提取

<a id="overview"></a>
## 用途

从一个已解码 RGB 人体块提取 512 维特征，CLI 制品选择与模型 API 分开。

<a id="directory"></a>
## 目录

main.py 构造模型并调用 predict；cli.py 管理参数、输入和输出；matching.py 完整实现三阶段；run.sh 转发参数。

<a id="environment"></a>
## 环境

使用提供匹配 hbm_runtime、libdnn 1.24.5、HBRT 3.15.55.0 的 X5 镜像，这是制品记录中的环境；原实验未记录系统镜像版本。Python ≥3.10、NumPy、Pillow ≥9.1 为必需依赖，转换另需 PyYAML。回读包、BIN 和固定输入预留约 40 MB。使用已有环境，脚本不安装依赖。 hbm_runtime 仅由匹配板端镜像提供。依赖记录在 ../../requirements-host.txt；如自行准备隔离用户环境，可显式执行：

```bash
python3 -m pip install -r samples/vision/osnet/requirements-host.txt
```


S 系列板端已验证 UCP/DNN 3.13.6、HBRT 4.7.5；使用板卡自带 hbm_runtime、NumPy 和 Pillow（S100 的 Pillow 9.0.1 亦可运行，主机导出依赖见 requirements-host.txt）。S100/S100P 系统为 4.0.5-Beta，S600 为 5.1.0。

<a id="usage"></a>
## 使用

在 X5 的仓库根目录运行，先准备匹配 BIN 和 --fixtures。零参数使用固定输入首图；也可指定自有图像。正常退出 0，保存 (512,) float32 特征。

```bash
python3 samples/vision/osnet/runtime/python/main.py
python3 samples/vision/osnet/runtime/python/main.py --target x5 \
  --variant spatial-rgb-int16 --test-img /absolute/path/person_crop.jpg --output person.npy
python3 samples/vision/osnet/runtime/python/main.py --dry-run
```

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| --target | str | x5 | x5 / s100 / s100p / s600 / auto |
| --variant | str | spatial-rgb-late8 | Registered variant / 已登记版本 |
| --model-path | Path | null | Registered BIN/HBM filename / 已登记模型文件名 |
| --test-img | Path | null | Defaults to first prepared fixture crop / 默认固定输入首图 |
| --output | Path | feature.npy | NumPy feature file / NumPy 特征 |
| --raw | bool | false | Disable L2 / 不做 L2 |
| --dry-run | bool | false | Show selection without SDK / 无 SDK 预览 |
| --list-models | bool | false | List identities without SDK / 无 SDK 列表 |

<a id="results"></a>
## 结果

API 返回独立拥有内存的 NumPy float32 (512,) 数组。默认除以 max(norm,1e-12)，--raw 保留原值；不做 softmax，不返回类别或框。输出文件名没有 .npy 时由 np.save 添加。

<a id="integration-example"></a>
## 集成示例

前提：已准备默认 BIN 和 images.dat。在 X5 仓库根目录运行以下完整示例：

```python
from pathlib import Path
from samples.vision.osnet.runtime.python.cli import read_image
from samples.vision.osnet.runtime.python.matching import OSNet, OSNetConfig

image = read_image()  # first crop from prepared test_data/fixture/images.dat
model = OSNet(OSNetConfig(
    Path("samples/vision/osnet/model/spatial-rgb-late8.bin"), target="x5"))
feature = model.predict(image)
prepared = model.preprocess(image)
raw = model.infer(prepared)
same_feature = model.postprocess(raw)
print(feature.shape, same_feature.shape)
```

<a id="stage-io"></a>
## 三阶段输入输出

Input 为 Pillow 图像或 H×W×3 uint8 RGB（OpenCV BGR 须先转换）。preprocess 用 Pillow 双线性缩放到 128×256，减 128 后排列为 int8 CHW 平面字节，再 reshape 为元数据 [1,256,128,3]，返回模型名/输入名嵌套映射。整个人体块的特征不需坐标还原，Context 为空。infer 调用 SDK 并返回嵌套特征映射。postprocess 要求 512 个有限 float32 值，复制为独立数组并可选 L2 归一化。元数据虽是 NHWC，此有符号平面 RGB 合同仅适用于这些编译产物。

<a id="troubleshooting"></a>
## 故障排查

缺模型/固定输入：执行 model/download.py 并加 --fixtures。摘要不符：选择匹配 --variant 的制品。未知板卡/目标不符：只在实际检测到的对应 X5/S 系列板卡执行。形状或类型不符：检查 RGB 制品及 SDK。库接口允许自定义路径，但要求元数据和实际有符号平面合同一致；CLI 制品身份仍固定。模型阶段不下载、不安装、不打印、不落盘。

S 系列 preprocess 输出 F32 NCHW [1,3,256,128]：RGB 双线性缩放后 /255，再按 mean=[.485,.456,.406]、std=[.229,.224,.225] 归一化。X5 仍使用上述 signed planar 合同。命令见 [S 系列流程](../../evaluator/S_SERIES_cn.md)。
