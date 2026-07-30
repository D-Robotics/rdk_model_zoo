[English](./README.md) | [简体中文](./README_cn.md)

# Python 推理

## 环境要求

- RDK X5 板端环境
- BSP 自带且与当前 `libdnn` 匹配的 `hbm_runtime`
- Python 3、NumPy 和 OpenCV

不要安装 PyPI 上的同名 `hbm_runtime` 包。

## 运行方式

使用默认模型、图片和输出目录：

```bash
bash run.sh
```

显式指定全部路径：

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

默认模型为 `yolo26n_depth_bayese_768x768_nv12.bin`，默认输入为 `test_data/bus.jpg`，默认输出目录为 `test_data/python_result`。

## 输出文件

- `log_depth.npy`：原始 calibrated log-depth。
- `depth_native.npy`：恢复到原图尺寸的相对深度。
- `depth.png`：深度伪彩色图。
- `overlay.png`：原图与深度可视化叠加图。
- `report.json`：模型、输入、几何信息、输出尺寸和延迟元数据。

## 代码接口

`yolo26_depth.py` 提供可复用的 `Yolo26Depth` 类。模型专用 letterbox 还原逻辑保留在样例内，NV12 转换复用 `utils/py_utils/preprocess.py`。

按照[源码文档说明](../../../../../docs/source_reference/README.md)生成接口文档。
