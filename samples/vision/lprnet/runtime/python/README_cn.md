# LPRNet 识别 Python 示例

本示例展示如何在 RDK X5 上使用 `hbm_runtime` 运行 LPRNet 车牌识别模型。

## 环境依赖

- `RDK OS >= 3.5.0`
- 板端镜像已预装 `hbm_runtime`

## 目录结构

```text
.
├── main.py
├── lprnet.py
├── run.sh
├── README.md
└── README_cn.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-path` | LPRNet `.bin` 模型路径 | `../../model/lpr.bin` |
| `--priority` | Runtime 调度优先级 | `5` |
| `--bpu-cores` | BPU 核心索引 | `0` |
| `--test-bin` | 打包好的 `float32` 输入张量路径 | `../../test_data/test.bin` |

## 快速运行

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

- 使用默认参数运行：
  ```bash
  python3 main.py
  ```

- 显式指定参数运行：
  ```bash
  python3 main.py \
    --model-path ../../model/lpr.bin \
    --test-bin ../../test_data/test.bin
  ```

## 接口说明

- `LPRNetConfig`：封装模型路径和二进制输入路径。
- `LPRNet`：实现 `set_scheduling_params`、`pre_process`、`forward`、`post_process`、`predict`。
