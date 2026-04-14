简体中文 | [English](./README.md)

# YOLOv5 模型评测

该目录用于记录 YOLOv5 在 RDK X5 上的 benchmark 数据、运行验证结果和性能说明。

## 支持模型

当前 X5 benchmark 覆盖以下模型：

- `YOLOv5s_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5m_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5l_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5x_tag_v2.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5n_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5s_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5m_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5l_tag_v7.0_detect_640x640_bayese_nv12.bin`
- `YOLOv5x_tag_v7.0_detect_640x640_bayese_nv12.bin`

## 测试环境

- 设备：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`
- 输入尺寸：`640x640`
- 输入格式：`NV12`

## 验证方式

迁移后的 Python sample 通过以下方式验证：

```bash
cd ../runtime/python
bash run.sh
python3 main.py
python3 main.py --model-path ../../model/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

当前发布的 9 个 X5 YOLOv5 模型均可通过这条 Python 运行链路，并能正常保存结果图。

## Benchmark 结果

### RDK X5 性能数据

| 模型 | 分辨率 | 参数量 | BPU 吞吐 | Python 后处理 |
| :--- | :--- | ---: | :--- | :--- |
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

## 性能说明

- `BPU 吞吐` 表示对应模型在 RDK X5 上的参考吞吐能力。
- `Python 后处理` 表示 Python 后处理路径的参考 CPU 侧耗时。
- `参数量` 表示原始 FP32 模型的参数规模。
- 当前迁移版 sample 保留了原始 YOLOv5 的 anchor-based 解码协议。

## 参考材料

- 运行说明：`../runtime/python/README_cn.md`
- 模型下载：`../model/README_cn.md`
- 转换说明：`../conversion/README_cn.md`
- benchmark 参考资源：`../test_data/`
