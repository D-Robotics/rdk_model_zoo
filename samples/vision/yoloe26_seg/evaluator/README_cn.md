[English](./README.md) | 简体中文

# 模型评测

## Runtime 性能

测试设备为 S100 V1P0，系统 RDK OS 4.0.5-Beta，UCP 3.13.6 / HBRT 4.7.5。
HBM 使用 OE 3.7.0、HBDK 4.7.5、INT8 KL，batch=1、640×640 NV12、4585 类。
测试于 2026-09-08 使用 `hrt_model_exec perf` 完成：200 帧，开启预热，
thread_num=1，core_id=0。

这里只展示**模型 Runtime** 延迟和吞吐量，不包含应用前处理、输出反量化、
后处理、可视化或 I/O，也不代表 Python/C++ 端到端 FPS。

| 板型 | 模型 | Runtime 延迟（ms） | Runtime FPS |
|---|---|---:|---:|
| S100 | YOLOE-26n Seg PF | 4.943 | 200.74 |
| S100 | YOLOE-26s Seg PF | 9.944 | 100.08 |
| S100 | YOLOE-26m Seg PF | 11.765 | 84.55 |
| S100 | YOLOE-26l Seg PF | 13.417 | 74.18 |
| S100 | YOLOE-26x Seg PF | 22.013 | 45.31 |
| S100P | YOLOE-26n/s/m/l/x Seg PF | 未测量 | 未测量 |

```bash
hrt_model_exec perf --model_file /path/model.hbm \
  --thread_num 1 --core_id 0 --frame_count 200 --enable_warmup true \
  --profile_path ./profile
```

## Runtime 契约回归

在 `samples/vision/yoloe26_seg/` 示例根目录准备 Python 3、NumPy、OpenCV
和 SciPy，然后执行：

```bash
python3 evaluator/test_runtime.py
```

测试覆盖框内局部 mask 的公共可视化、裁剪及退化 ROI 的结果对齐、分阶段
前处理/推理接口，以及调度参数更新。在无法导入 `hbm_runtime` 的开发主机上，
测试只为缺失的硬件绑定提供替代；在开发板上会导入已安装的真实绑定。
仓库中的前处理、可视化代码和 NumPy/OpenCV/SciPy 依赖仍使用真实实现。
这是接口契约回归，不是 HBM 推理、数据集 mAP 评测或板端性能 benchmark。

在安装了 C++ Runtime 依赖的 S100/S100P 上，从同一示例根目录执行板端
契约检查（请按板型替换 HBM 路径）：

```bash
g++ -std=c++17 -O2 evaluator/cpp/test_cpp_contract.cpp \
  runtime/cpp/src/yoloe26seg.cpp -Iruntime/cpp/inc -I../../../utils/c_utils/inc \
  -I/usr/hobot/include -L/usr/hobot/lib -ldnn -lhbucp \
  $(pkg-config --cflags --libs opencv4) -o /tmp/yoloe26_cpp_contract
/tmp/yoloe26_cpp_contract \
  model/nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm test_data/office_desk.jpg
```

该检查已在 S100P 的 n 模型上通过，覆盖未初始化时预测、初始化失败后重试、
重复初始化、框内二值 mask，以及错误输出张量形状的拒绝处理。

## 精度验证状态

此前五个 S100 模型的 Python/C++ 实测返回了相同的检测数量和类别，
框与分数差异小于 1e-7，mask 面积一致。面积相等不代表 mask 逐像素完全相同。

2026-09-17 在 S100P 上完成了 n 模型的 Python/C++ 回归，
环境为 UCP 3.13.6 / HBRT 4.7.5。已发布的 nash-m n 模型 HBM（SHA256
`58922c669bcccec7a00c11719db6c3b9c4eaa4e0388b58bd120fda914fe9d048`）
处理内置 `office_desk.jpg` 时，零参数 `main.py` 和通过 CMake 构建的 C++ 程序
均成功输出 14 个检测。两种语言各自与基线提交
`855a2b176fd215e4e62e73372c85da33f4f076bc` 比较，以及 Python/C++ 相互比较，
检测框、分数、类别和每个框内 mask 的像素均完全一致。比较前，基线整图 mask
按相同的裁剪后整数截断边界转换为框内 mask。四项 Python Runtime 契约测试
在开发主机和该 S100P 上都通过。
这项证据仅覆盖 n 模型和内置图片，不代表 s/m/l/x、数据集 mAP 或 S100P Runtime
性能已验证。

校准使用了 100 张具有代表性的 COCO train2017 图片。数据集框与 mask 的 mAP
尚未通过验收，INT8 基线与浮点推理存在数值差异。生产使用前应在独立的带标注
评测集上验证，并显式建立 PF 词表到数据集类别的映射：PF 类别 ID 不是 COCO ID。
本说明不提供未经测量的 mAP 或 S100P Runtime 数值。
