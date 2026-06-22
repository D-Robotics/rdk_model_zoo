[English](./README.md) | 简体中文

# SigLIP 模型评估说明

本目录记录 SigLIP 视觉编码器在 RDK S100/S100P 上的性能和精度参考数据。每个 HBM 模型由 `last_hidden_state` 和 `pooler_output` 两个子模型 pack 而成，两个子模型共享权重。

## 性能数据

### pooler output

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,1,768)` | `0.2 B / 0.09 B` | 26.8 ms | 18.8 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,1,768)` | `0.2 B / 0.09 B` | 46.7 ms | 32.3 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1,768)` | `0.2 B / 0.09 B` | 81.7 ms | 55.8 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 68.8 ms | 47.2 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,1,1024)` | `0.7 B / 0.32 B` | 132.5 ms | 91.4 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 89.8 ms | 62.2 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,1,1152)` | `0.9 B / 0.43 B` | 255.7 ms | 175.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,1,1152)` | `1.0 B / 0.43 B` | 89.6 ms | 61.9 ms |

### last hidden state

| Model Name | Input Size | Embedding Size | Params total / vision | RDK S100 | RDK S100P |
|---|---|---|---|---|---|
| siglip-base-patch16-224 | `(1,3,224,224)` | `(1,196,768)` | `0.2 B / 0.09 B` | 26.0 ms | 18.3 ms |
| siglip-base-patch16-384 | `(1,3,384,384)` | `(1,576,768)` | `0.2 B / 0.09 B` | 45.9 ms | 31.7 ms |
| siglip-base-patch16-512 | `(1,3,512,512)` | `(1,1024,768)` | `0.2 B / 0.09 B` | 80.8 ms | 55.3 ms |
| siglip-large-patch16-256 | `(1,3,256,256)` | `(1,256,1024)` | `0.7 B / 0.32 B` | 67.6 ms | 46.5 ms |
| siglip-large-patch16-384 | `(1,3,384,384)` | `(1,576,1024)` | `0.7 B / 0.32 B` | 131.3 ms | 90.5 ms |
| siglip-so400m-patch14-224 | `(1,3,224,224)` | `(1,256,1152)` | `0.9 B / 0.43 B` | 88.6 ms | 61.4 ms |
| siglip-so400m-patch14-384 | `(1,3,384,384)` | `(1,729,1152)` | `0.9 B / 0.43 B` | 254.2 ms | 174.5 ms |
| siglip-so400m-patch16-256-i18n | `(1,3,256,256)` | `(1,256,1152)` | `1.0 B / 0.43 B` | 88.3 ms | 61.1 ms |

## 性能测试方法

```bash
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file <*.hbm>
hrt_model_exec perf --thread_num 1 --model_name pooler_output --model_file <*.hbm>
```

测试板卡状态：

- S100P: CPU `6 x A78AE @ 2.0GHz`，BPU `1 x Nash-M @ 1.5GHz`
- S100: CPU `6 x A78AE @ 1.5GHz`，BPU `1 x Nash-E @ 1.0GHz`

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```

## 精度数据

### pooler output 零样本分类

| Model Name | PyTorch TOP1 / TOP5 | BPU TOP1 / TOP5 |
|---|---|---|
| siglip-base-patch16-224 | 0.7123 / 0.9143 | 0.7118 / 0.9144 |
| siglip-base-patch16-384 | 0.7411 / 0.9318 | 0.7418 / 0.9319 |
| siglip-base-patch16-512 | 0.7490 / 0.9343 | 0.7482 / 0.9340 |
| siglip-large-patch16-256 | 0.7490 / 0.9238 | 0.7490 / 0.9242 |
| siglip-large-patch16-384 | 0.7584 / 0.9252 | 0.7595 / 0.9256 |
| siglip-so400m-patch14-224 | 0.7659 / 0.9361 | 0.7651 / 0.9357 |
| siglip-so400m-patch14-384 | 0.7872 / 0.9433 | 0.7893 / 0.9447 |
| siglip-so400m-patch16-256-i18n | 0.7678 / 0.9395 | 0.7668 / 0.9397 |

### last hidden state 语义一致性

| Model Name | Cosine Similarity mean (min ~ max), 1% low | MSE mean (min ~ max), 1% low |
|---|---|---|
| siglip-base-patch16-224 | 0.991 (0.951 ~ 0.997), 0.980 | 0.087 (0.024 ~ 0.471), 0.039 |
| siglip-base-patch16-384 | 0.989 (0.960 ~ 0.997), 0.977 | 0.113 (0.029 ~ 0.409), 0.050 |
| siglip-base-patch16-512 | 0.987 (0.956 ~ 0.995), 0.974 | 0.142 (0.045 ~ 0.507), 0.067 |
| siglip-large-patch16-256 | 0.990 (0.933 ~ 0.997), 0.974 | 0.069 (0.018 ~ 0.497), 0.024 |
| siglip-large-patch16-384 | 0.985 (0.900 ~ 0.995), 0.965 | 0.111 (0.034 ~ 0.775), 0.048 |
| siglip-so400m-patch14-224 | 0.984 (0.850 ~ 0.995), 0.961 | 0.104 (0.028 ~ 1.038), 0.041 |
| siglip-so400m-patch14-384 | 0.980 (0.859 ~ 0.993), 0.957 | 0.140 (0.040 ~ 1.093), 0.059 |
| siglip-so400m-patch16-256-i18n | 0.984 (0.878 ~ 0.996), 0.959 | 0.082 (0.018 ~ 0.570), 0.030 |

## 精度测试方法

1. `last_hidden_state` 语义一致性验证使用 COCO2014 val 验证集 5,000 张图片，指标为余弦相似度和均方误差。
2. `pooler_output` 零样本分类精度验证使用 ImageNet-1k val 验证集 50,000 张图片，指标为 TOP1 和 TOP5 正确率。
3. 两类精度验证的图像前处理均为 RGB `(127,127,127)` letterbox，浮点模型和 BPU 模型保持一致。

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
