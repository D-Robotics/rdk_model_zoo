简体中文 | [English](./README.md)

# YOLOE 模型评测

该目录用于记录 YOLOE 在 RDK X5 上的 benchmark 数据、运行验证结果和性能说明。

## 支持模型

当前 X5 benchmark 覆盖以下模型：

- `yoloe_11s_seg_pf_bayese_640x640_nv12.bin`

## 测试环境

- 设备：`RDK X5`
- 运行后端：`hbm_runtime`
- 模型格式：`.bin`
- 输入尺寸：`640x640`
- 输入格式：`NV12`

## 验证方式

Python 示例通过以下方式验证：

```bash
cd ../runtime/python
bash run.sh
python3 main.py
```

## Benchmark 结果

### RDK X5 性能数据

| 模型 | 分辨率 | 线程数 | 延迟 | FPS |
| :--- | :--- | ---: | :--- | :--- |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 142.9 ms | 7.0 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 2 | 149.5 ms | 13.3 FPS |
| YOLOE-11s-Seg-PF | 640x640 | 3 | 167.4 ms | 17.8 FPS |

## 性能说明

- `延迟` 为 RDK X5 上每帧端到端总耗时。
- `FPS` 为对应线程数下的整体吞吐量。
- 该模型支持 4585 类开放词汇实例分割。

## 参考材料

- 运行说明：`../runtime/python/README_cn.md`
- 模型下载：`../model/README_cn.md`
- 转换说明：`../conversion/README_cn.md`
- benchmark 参考资源：`../test_data/`
