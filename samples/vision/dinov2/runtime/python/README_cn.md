[English](./README.md) | 简体中文

# DINOv2 Python Runtime

DINOv2 ViT-S/14 int16 `.hbm` 模型的板端推理演示，基于 `hbm_runtime`。

## 输入协议

- 输入名：`input`
- 输入格式：float32 NCHW RGB，由板端 CPU 完成全部预处理（方形 resize 到
  224、`/255`、ImageNet mean/std 归一化）。
- `.hbm` 直接接收最终 float 张量；图内不包含任何图像预处理。

## 输出

- `cls_feat`：`(1, 384)` 全局图像 embedding。
- `patch_feat`：`(1, 256, 384)` 稠密 patch 级特征。

## 使用方法

```bash
bash run.sh                 # 默认：cls_feat 输出，模型自动检测
bash run.sh patch_feat      # 查看 dense 输出
```

等效直接调用：

```bash
python3 main.py \
  --model-path ../../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
  --test-img ../../test_data/dog.jpg \
  --second-img ../../test_data/bus.jpg \
  --output cls_feat
```

未指定 `--model-path` 时，模型 march 由板端 SoC 自动检测。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | 自动（按 SoC 检测） | `.hbm` 模型路径。 |
| `--test-img` | `../../test_data/dog.jpg` | 第一张测试图。 |
| `--second-img` | `../../test_data/bus.jpg` | 相似度演示的第二张图。 |
| `--image-size` | 224 | 方形输入分辨率。 |
| `--output` | `cls_feat` | 查看的输出：`cls_feat` 或 `patch_feat`。 |
| `--priority` | 0 | 运行时优先级（0-255）。 |
| `--bpu-cores` | 0 | BPU 核编号。 |

## 许可

见 [../../../../LICENSE](../../../../LICENSE)。
