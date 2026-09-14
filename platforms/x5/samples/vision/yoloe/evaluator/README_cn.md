简体中文 | [English](./README.md)

# YOLOE 模型评测

该目录用于记录 YOLOE 在 RDK X5 上的 benchmark 数据、运行验证结果和性能说明。

## 支持模型

当前 X5 benchmark 覆盖以下模型：

- `yoloe_11s_seg_pf_bayese_640x640_nv12.bin`
- `yoloe_11m_seg_pf_bayese_640x640_nv12.bin`
- `yoloe_11l_seg_pf_bayese_640x640_nv12.bin`

## 测试环境

- 设备：`RDK X5 V1.0`
- 系统：`3.4.1-rp1.0.2`
- 运行后端：C++ `libdnn 1.24.5` / `HBRT 3.15.55`
- 模型格式：`.bin`
- 输入尺寸：`640x640`
- 输入格式：`NV12`
- BPU：`core_id=1`（BPU core 0），`1000 MHz`

## 验证方式

使用固定 NV12 输入，通过 C++ libdnn 接口计时。每轮预热 10 帧、测试 200 帧，共 3 轮。

## Benchmark 结果

### RDK X5 性能数据

| 模型 | 分辨率 | 线程数 | 平均延迟（ms） | P50（ms） | P95（ms） | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 146.16 | 144.72 | 152.73 | 6.84 |
| YOLOE-11m-Seg-PF | 640x640 | 1 | 177.14 | 176.17 | 182.54 | 5.65 |
| YOLOE-11l-Seg-PF | 640x640 | 1 | 189.97 | 187.99 | 196.30 | 5.26 |

## 性能说明

- 延迟、P50、P95 统计 Runtime 推理耗时，含模型内 CPU/BPU 执行，不含图像前处理和检测/掩码后处理。
- `FPS` 为总计时帧数除以总耗时。
- 11s 双线程测试出现 ION 分配失败，表中仅列单线程结果。

## 参考材料

- 运行说明：`../runtime/python/README_cn.md`
- 模型下载：`../model/README_cn.md`
- 转换说明：`../conversion/README_cn.md`
- benchmark 参考资源：`../test_data/`
