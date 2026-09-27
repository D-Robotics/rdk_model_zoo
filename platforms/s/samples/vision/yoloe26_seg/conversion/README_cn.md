[English](./README.md) | 简体中文

# YOLOE-26 PF 模型转换

本目录提供 YOLOE-26 无提示词实例分割模型 n/s/m/l/x 五种规格的导出和编译流程，
目标平台仅为 RDK S100（`nash-e`）和 S100P（`nash-m`），不支持 S600。

以下命令均从本示例根目录 `samples/vision/yoloe26_seg/` 执行。

## 目录内容

```text
conversion/
├── mapper.py
├── onnx_export/
│   └── export_yoloe26_seg_pf.py
├── README.md
├── README_cn.md
└── requirements.txt
```

导出器接收自行准备的本地 `yoloe-26<SIZE>-seg-pf.pt` 检查点。
本示例不自动下载这些权重。每个规格会导出静态 ONNX、按检查点类别顺序排列的
`.names` 词表及 JSON 元数据。元数据记录 ONNX 哈希和固定的十个输出：

```text
[cls_8, box_8, mc_8,
 cls_16, box_16, mc_16,
 cls_32, box_32, mc_32, proto]
```

输入为一个 `1x3x640x640` RGB 张量。板端前处理将相同 letterbox 规则处理后的
BGR 图片转换为 NV12。letterbox 填充值为 `114`，模型输入缩放系数为 `1/255`。

## 导出 ONNX

在用于导出的 Python 环境中安装依赖：

```bash
python3 -m pip install -r conversion/requirements.txt
```

每次处理一个规格，重试时使用新的输出目录：

```bash
python3 conversion/onnx_export/export_yoloe26_seg_pf.py \
  --weights /path/to/yoloe-26n-seg-pf.pt \
  --size n \
  --output-dir /path/to/exports/n \
  --test-image /path/to/test_data/office_desk.jpg \
  --threads 2
```

对 s、m、l、x 重复执行。导出目录包含
`<size>/yoloe_26<size>_seg_pf.{onnx,json,names}` 后，即可将五模型目录交给
`mapper.py` 批量处理。

## 准备配置与编译 HBM

`mapper.py` 校验所有 ONNX 和元数据，按确定性采样规则准备默认 100 张图片的
校准集，为每个模型生成可审阅的 YAML。只有显式添加 `--compile` 才会运行编译器，
且不允许复用已经存在的输出目录。

先省略 `--compile`，仅准备校准数据和配置用于检查：

```bash
python3 conversion/mapper.py \
  --onnx /path/to/exports \
  --metadata /path/to/exports \
  --cal-images /path/to/cal_images \
  --march nash-e \
  --output-dir /path/to/build/yoloe26_s100_preview \
  --sample-count 100
```

正式编译时使用新的输出目录并添加 `--compile`，该命令会连续执行准备和编译，
不会在生成配置后暂停等待确认：

```bash
python3 conversion/mapper.py \
  --onnx /path/to/exports \
  --metadata /path/to/exports \
  --cal-images /path/to/cal_images \
  --march nash-e \
  --output-dir /path/to/build/yoloe26_s100_compiled \
  --sample-count 100 \
  --compile
```

S100P 请改为 `--march nash-m`，并使用另一个新的输出目录。
模型及匹配的元数据、标签保存在 `<output-dir>/<size>/`；
`manifest.json` 和 `compile_results.json` 汇总本次结果。HBM 命名如下：

```text
yoloe_26<SIZE>_seg_pf_nashe_640x640_nv12.hbm  # nash-e / S100
yoloe_26<SIZE>_seg_pf_nashm_640x640_nv12.hbm  # nash-m / S100P
```

配置采用兼容 OE 3.7.0 的 KL INT8 PTQ，设置
`remove_node_type: Quantize;Dequantize`、`input_no_padding: true`、
`output_no_padding: false`、`jobs: 4` 和优化级别 `O2`。
输出可能带 padding，读取 HBM 输出时必须使用有效形状和 Runtime 提供的 stride。

## Docker 环境

使用已验证的 OE 3.7.0 CPU 镜像执行 S100/S100P 编译。
导出文件和校准数据应放在仓库挂载目录中，或者额外挂载到容器可访问的位置：

```bash
REPO_DIR=/path/to/rdk_model_zoo
docker run --rm -it --shm-size=2g \
  -v "$REPO_DIR":/workspace \
  -w /workspace/samples/vision/yoloe26_seg \
  --entrypoint /bin/bash \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

容器中执行相同的 `mapper.py` 命令时，请使用容器可访问的 `/workspace` 路径。
镜像名称虽包含 S600，本示例仍只允许 `nash-e` 和 `nash-m`。
镜像应已包含工具链，转换脚本不会自动安装软件或修改板卡配置。

## 验证说明

导出器会逐个模型检查与上游静态 PF 的等价性，并执行 ONNX Runtime 输出对拍。
编译按模型独立执行，保存 `compile.log` 和逐模型结果。
本地编译成功仅代表 `compiled_not_board_validated`，不代表已通过板端精度或性能验收。
部署时请同时复制 HBM 及同目录中匹配的 JSON 和 `.names` 文件。
