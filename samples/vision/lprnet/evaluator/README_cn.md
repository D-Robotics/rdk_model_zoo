# LPRNet 模型评测

本目录用于记录 LPRNet 在 RDK X5 上的 benchmark 与验证说明。

## 评测前提

- `RDK OS >= 3.5.0`
- 板端 Python 3 环境

## 数据准备

- 本示例使用仓库内自带的二进制输入张量：
  - `../test_data/test.bin`
- 对应的可视化参考图为：
  - `../test_data/example.jpg`

## 使用方法

运行 Python 示例：

```bash
cd ../runtime/python
bash run.sh
```

## Benchmark 结果

| 模型 | 测试帧数 | FPS | 平均延迟 | BPU 占用率 | ION 内存 |
| --- | --- | --- | --- | --- | --- |
| `lpr.bin` | `100` | `266 FPS` | `3.75 ms` | `9%` | `1.11 MB` |

## 验证结果说明

该示例的运行验证应确认：

- 模型可在 RDK X5 上正常加载
- 可以从 `test.bin` 读取输入张量
- runtime 能输出解码后的车牌字符串
