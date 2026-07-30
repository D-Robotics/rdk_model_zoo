[English](./README.md) | [简体中文](./README_cn.md)

# 模型评估

本目录提供可复用的数值对比和 SUN RGB-D 评测工具。生成的数据集、预测结果、报告和可视化必须输出到样例目录之外。

## RDK X5 性能数据

模型使用 OpenExplorer v1.2.8 / Mapper 1.24.3 编译，输入尺寸为 768×768，max percentile 为 `0.9999`，采用 O3 latency 优化和 int16 尾卷积输出。HRT 数据仅统计模型执行时间。

| 平台 | 模型 | 输入尺寸 | 单线程延迟 | 单线程 FPS | 双线程总 FPS |
| --- | --- | --- | ---: | ---: | ---: |
| RDK X5 | YOLO26n Depth | 768×768 | 23.194 ms | 43.085 | 45.682 |
| RDK X5 | YOLO26s Depth | 768×768 | 36.168 ms | 27.637 | 28.615 |
| RDK X5 | YOLO26m Depth | 768×768 | 60.783 ms | 16.449 | 16.751 |
| RDK X5 | YOLO26l Depth | 768×768 | 75.336 ms | 13.272 | 13.470 |
| RDK X5 | YOLO26x Depth | 768×768 | 161.022 ms | 6.210 | 6.253 |

当前样例尚未发布板端精度数据；在具备已验证预测结果和真值后，可使用下方工具生成精度报告。

## 准备 SUN RGB-D 输入

生成确定性的 deployment-letterbox 和 Ultralytics-validator 输入：

```bash
python3 prepare_sunrgbd.py \
  --source-root SUNRGBD_ROOT \
  --source-manifest SOURCE_MANIFEST.json \
  --output OUTPUT_DIR \
  --size 768
```

输出内容包括 RGB CHW uint8 张量、深度数组以及记录预处理几何信息的 manifest。

## 单图数值对比

对比 X5 还原深度和官方浮点结果：

```bash
python3 eval_numeric.py \
  --image ../test_data/bus.jpg \
  --official OFFICIAL_DEPTH.npy \
  --x5 X5_DEPTH.npy \
  --output REPORT_DIR
```

命令会输出数值指标、JSON 报告和对比可视化。

## SUN RGB-D 评测

基于准备好的 SUN RGB-D manifest 评测模型输出：

```bash
python3 eval_sunrgbd.py --help
```

评测脚本支持 deployment-letterbox 和 Ultralytics-validator 两种预处理协议，并输出常用单目深度指标。可通过命令帮助选择输出文件、对齐方式和预处理协议。

## 注意事项

- 评测输入和生成结果不会保存在仓库中。
- 相对深度与米制真值比较前需要执行尺度或尺度偏移对齐。
- 输入准备和评测必须使用相同的预处理协议。
