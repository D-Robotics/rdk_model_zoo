[English](./README.md) | 简体中文

# ResNet18 Python 运行示例

本示例使用 `hbm_runtime` 运行 ResNet18 图像分类模型，并打印 ImageNet Top-K
结果。

## 目录结构

```text
.
|-- README.md
|-- README_cn.md
|-- main.py
|-- resnet18.py
`-- run.sh
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | HBM 模型路径 | `../../model/s100/resnet18_224x224_nv12.hbm` |
| `--test-img` | 输入图片路径 | `../../test_data/zebra_cls.jpg` |
| `--label-file` | ImageNet 标签文件 | `../../../../../datasets/imagenet/imagenet_classes.names` |
| `--top-k` | 打印的分类结果数量 | `5` |
| `--priority` | 运行优先级，0 最低 | `0` |
| `--bpu-cores` | BPU 核心编号列表 | `0` |

## 快速运行

```bash
bash run.sh
```

脚本会通过 `../../model/download_model.sh` 下载模型，默认使用 sample 内
`../../model/s100/` 目录。RDK S600 用户请先执行
`bash ../../model/download_model.sh s600`，并将 `--model-path` 改为
`../../model/s600/resnet18_224x224_nv12.hbm`。

## 直接运行

```bash
python3 main.py \
  --model-path ../../model/s100/resnet18_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Runtime 接口

`resnet18.py` 提供：

- `Resnet18Config`
- `Resnet18.set_scheduling_params(...)`
- `Resnet18.pre_process(...)`
- `Resnet18.forward(...)`
- `Resnet18.post_process(...)`
- `Resnet18.predict(...)`
- `Resnet18.__call__(...)`

wrapper 会将 resize 后的 BGR 图片转换为 NV12 Y 和 UV 两个平面，并按固定两个
输入 tensor 传入 `HB_HBMRuntime`。

`zebra_cls.jpg` 的预期结果：

```text
Top-5 Classification Results:
  [0] zebra: ...
```

源码注释规范可参考 `../../../../../docs/source_reference/README.md`。
