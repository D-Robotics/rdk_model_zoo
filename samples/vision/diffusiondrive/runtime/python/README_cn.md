[English](./README.md) | 简体中文

# DiffusionDrive Python 推理

本示例使用 `hbm_runtime` 完成一次四输入 DiffusionDrive 推理，保存解码后的 NPZ 张量和 PNG 可视化结果。

## 环境依赖

S600 系统需提供 Python 3、`hbm_runtime`、NumPy 和 OpenCV：

```bash
python3 -c "import hbm_runtime, numpy, cv2"
```

## 目录结构

```text
.
|-- diffusiondrive.py           # 配置类、模型类、后处理和可视化
|-- main.py                     # argparse 推理入口
|-- run.sh                      # 一键检查模型并执行推理
|-- run_all_cases.sh            # 运行全部 test_data/case_* 样例
|-- README.md                   # 英文说明
`-- README_cn.md                # 中文说明
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | S600 HBM 路径 | `../../model/s600/diffusiondrive_r34_256x1024_s600.hbm` |
| `--input-npz` | 四路 float32 输入 | `../../test_data/reference_inputs.npz` |
| `--output-npz` | 解码张量输出 | `./diffusiondrive_outputs.npz` |
| `--img-save-path`、`--output-image` | 可视化输出，两个参数名都支持 | `./diffusiondrive_result.png` |
| `--agent-score-thres` | 目标 sigmoid 阈值 | `0.5` |
| `--priority` | 调度优先级 | `0` |
| `--bpu-cores` | BPU 核编号 | `0` |

## 快速运行

使用默认路径：

```bash
bash run.sh
```

指定输入输出：

```bash
python3 main.py \
  --model-path ../../model/s600/diffusiondrive_r34_256x1024_s600.hbm \
  --input-npz ../../test_data/reference_inputs.npz \
  --output-npz ../../test_data/my_outputs.npz \
  --img-save-path ../../test_data/my_result.png
```

输入 NPZ 必须包含 `camera`、`lidar`、`status`、`noise`。代码从 HBM metadata 动态读取量化 scale，不硬编码模型参数。

运行全部已打包场景：

```bash
bash run_all_cases.sh
```

结果默认写入 `runtime/python/results/case_*/`；也可以把其他输出目录作为脚本的第一个参数。

## 代码文档

二次集成接口为 `DiffusionDriveConfig` 及 `DiffusionDrive.pre_process`、`forward`、`post_process`、`predict`。源码文档生成方式见仓库 [source documentation guide](../../../../../docs/source_reference/README.md)。

## 注意事项

- 生产程序应只加载一次模型并复用实例，进程启动时间不是 BPU 推理耗时。
- `noise` 是显式输入；需要复现结果时必须固定。
- 本示例只可视化模型张量。完整 NAVSIM 地图、标注、GIF 和 PDM Score 需在有 NAVSIM 数据的 x86 主机运行。
