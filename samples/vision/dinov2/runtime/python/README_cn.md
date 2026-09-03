[English](./README.md) | 简体中文

# DINOv2 Python Runtime

DINOv2 ViT-S/14 int16 `.hbm` 模型的板端推理演示，基于 `hbm_runtime`。

## 环境与文件

支持 RDK S100（`nash-e`）、RDK S100P（`nash-m`）和 RDK S600（`nash-p`）。
请在包含 `hbm_runtime` 的 RDK 系统镜像上运行，并安装 Python 3、NumPy 和
OpenCV。

```text
runtime/python/
├── README.md       # English guide
├── README_cn.md    # 中文说明
├── dinov2.py       # 预处理与 HBM wrapper
├── main.py         # 命令行演示与 stdout summary
└── run.sh          # 模型下载与演示启动脚本
```

## 输入协议

- 输入名：`input`
- 输入格式：固定形状 `1x3x224x224` 的 contiguous float32 NCHW RGB 张量；
  模型的输入协议固定为 224。
- 板端 CPU 使用 OpenCV 将 BGR 转 RGB，按比例 bicubic 将短边 resize 到
  256，中心 crop 为 224 x 224，执行 `/255` 和 ImageNet mean/std 归一化，
  再转换为 contiguous float32 NCHW。
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

## Stdout 输出

demo 会先打印模型 metadata，随后打印所选 embedding 的 summary：输出名、
shape、dtype、mean、std、min、max 和 L2 norm。当 `--second-img` 文件存在时，
还会打印两张图 embedding 的 cosine similarity。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | 自动（按 SoC 检测） | `.hbm` 模型路径。 |
| `--test-img` | `../../test_data/dog.jpg` | 第一张测试图。 |
| `--second-img` | `../../test_data/bus.jpg` | 相似度演示的第二张图。 |
| `--output` | `cls_feat` | 查看的输出：`cls_feat` 或 `patch_feat`。 |
| `--priority` | 0 | 运行时优先级（0-255）。 |
| `--bpu-cores` | 0 | BPU 核编号。 |

## 许可

见 [../../../../../LICENSE](../../../../../LICENSE)。
