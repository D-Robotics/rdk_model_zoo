[English](./README.md) | 简体中文

# MobileNetV4 Python 运行示例

本示例使用 `hbm_runtime` 运行 MobileNetV4 图像分类模型，并打印 ImageNet
Top-K 结果。支持 S100 / S600 small 和 medium HBM 模型，运行时根据 SOC 自动识别。

## 目录结构

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- mobilenetv4.py
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-variant` | 模型版本：`small` 或 `medium` | `small` |
| `--model-path` | HBM 模型路径；为空时按 `--model-variant` 选择 sample 内模型 | `small`: `../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm` |
| `--test-img` | 输入图片路径 | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet 标签文件 | `../../test_data/imagenet_classes.names` |
| `--top-k` | 打印的分类结果数量 | `5` |
| `--priority` | 运行优先级，0 最低 | `0` |
| `--bpu-cores` | BPU 核心编号列表 | `0` |

## 快速运行

Small 模型：

```bash
bash run.sh
```

Medium 模型：

```bash
bash run.sh medium
```

脚本会通过 `../../model/download_model.sh` 下载模型，并使用 sample 内
`../../model/<soc>/` 目录（`<soc>` ∈ {`s100`, `s600`}）。

## 直接运行

把 `<soc>` 替换为 `s100` 或 `s600`：

```bash
python3 main.py \
  --model-variant small \
  --model-path ../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

```bash
python3 main.py \
  --model-variant medium \
  --model-path ../../model/<soc>/mobilenetv4_medium_256x256_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

## Runtime 接口

`mobilenetv4.py` 提供：

- `MobileNetV4Config`
- `MobileNetV4.set_scheduling_params(...)`
- `MobileNetV4.pre_process(...)`
- `MobileNetV4.forward(...)`
- `MobileNetV4.post_process(...)`
- `MobileNetV4.predict(...)`
- `MobileNetV4.__call__(...)`

wrapper 会将 resize 后的 BGR 图片转换为 NV12 Y 和 UV 两个平面，并按固定
两个输入 tensor 传入 `HB_HBMRuntime`。

`zebra_cls.jpg` 的预期结果：

```text
Top-5 Classification Results:
  [0] zebra: ...
```

源码注释规范可参考 `../../../../../docs/source_reference/README.md`。
