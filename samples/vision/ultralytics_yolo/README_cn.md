# Ultralytics YOLO：X5 / S 共用 Sample

[English](README.md)

同一个 Sample、同一个 Python 入口，通过 `--platform x5|s100|s100p|s600` 选择目标。本轮从 X5/S 合并 YOLOv8 的检测、分割、姿态、分类，同时保留本 Sample 中 YOLOv5u、YOLOv9、YOLOv10、YOLO11、YOLO12 及 X5 YOLOv13 的已有模型组合。YOLO26 已接入同一个入口，覆盖 detect / cls / seg / pose / obb。YOLOE、独立 YOLOv5 和 yolo26_depth 暂不改动。

```text
ultralytics_yolo/
├── conversion/           # 共用导出补丁；mapper.py 选择 X5/S 工具链
│   ├── export_monkey_patch.py
│   ├── mapper.py
│   ├── mapper_x5.py      # hb_mapper，原始 float32 rgbchw 标定
│   └── mapper_s.py       # hb_compile，归一化 npy 标定
├── evaluator/            # 共用 COCO / ImageNet 评测入口
├── model/                # 统一下载；S 模型仍按 nash-e/m/p 存放
├── runtime/
│   ├── python/           # main.py + 任务/输出协议分派 + S YOLOv10 后处理
│   └── cpp/              # 原 X5 C++ 参考实现，不扩展 S 支持
├── test_data/            # 示例图片及标签
└── tests/                # 主机回归检查，不要求 BPU
```

从仓库根目录运行：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolov8 --task detect --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s600 --family yolov8 --task cls --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s100 --list-models
# 在对应板卡上去掉 --dry-run 执行推理；默认模型不存在时会下载。
```

| 差异 | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| 编译产物 | .bin / bayese | .hbm / nashe、nashm、nashp |
| Python 输入 | 单个 packed NV12 | 两个 NHWC Y、UV 张量 |
| 运行 CLI NMS 默认值 | 0.70 | 0.45 |
| 运行 CLI 分类缩放 | YOLO26 直接缩放（0）；其他系列 letterbox（1） | 直接缩放，resize-type=0 |
| 分类配置类默认缩放 | 0，保留原 API | 0，保留原 API |
| 已发布分类文件名 | YOLO26 为 224x224；其他系列为 640x640 | 224x224 |
| YOLOv10 运行路径 | DFL + NMS，保留原实现 | DFL 解码后不做 NMS |
| C++ | X5 参考实现 | 本 Sample 未提供 |

分类文件名是发布资产的名称，不能替代二进制输入形状检查。运行时从模型元数据读取尺寸，校验 batch=1、正偶数方形尺寸及输入协议；不明确的 flat 输入需要显式 `--input-shape HxW`，与元数据冲突会报错。DFL 检测类暂限三个特征层、reg=16；YOLO26 使用 stride 8/16/32 的四通道 LTRB；姿态暂限 17 点。这里只验证了主机逻辑，尚未执行真实板卡推理或编译工具链回归。

Python 主机检查需要 NumPy、OpenCV、SciPy；实际推理需要板卡系统提供的 `hbm_runtime`，脚本不会静默安装依赖。`--help`、`--dry-run`、下载列表不加载板卡运行时。显式 `--model-path` 不会自动下载文件；可识别的文件名会选择模型家族，自定义名称请同时指定 `--family`。

旧 `platforms/x5/`、`platforms/s/` 下本 Sample 的 Python、导出、转换、评测和下载入口转发到这里，不能再单独拷贝平台子树执行。旧 README 和 Manifest 保留为历史 Benchmark 证据；后续实现修改在本目录进行。其他 Sample 继续使用原平台目录。本轮不改历史 tag、线上下载地址或仪表盘快照。

- [Python 参数与接口](runtime/python/README_cn.md)
- [下载](model/README_cn.md)
- [转换](conversion/README_cn.md)
- [评测](evaluator/README_cn.md)
- [C++](runtime/cpp/README_cn.md)

回归检查：`python -m unittest discover -s samples/vision/ultralytics_yolo/tests`；资产清单测试需要先运行 `npm --prefix tools/catalog-publisher run build`。

## YOLO26：使用同一个 Sample 入口

通过 `--family yolo26` 选择版本，通过 `--task detect|cls|seg|pose|obb` 选择任务，平台仍使用 `--platform x5|s100|s100p|s600`。每个平台已有 25 个资产（五任务 × n/s/m/l/x），共 100 个；所有分类模型均为 224×224，其余任务为 640×640。这是既有支持的合并，不是新增 100 个模型。

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s600 --family yolo26 --task detect --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolo26 --task cls --dry-run
```

`yolo_dispatch.py` 按模型系列选择任务输出协议；YOLO26 是四通道 LTRB，YOLOv8 是 DFL。平台输入、分类实现、绘制、下载和评测基础设施共用。五个导出补丁放在 `conversion/yolo26/`，X5/S 编译流程保留各自实现。后续新版本在确认张量协议兼容后可以复用，不需要复制整个 Sample。

旧 `platforms/{x5,s}/samples/vision/ultralytics_yolo26` 路径转发到这里，旧 Python 适配层保留姿态、分割返回格式。X5 OBB 的角度偏置、按类别 NMS 和裁剪行为仍与 S 分开保留。输出按尺寸与通道绑定，不再依赖编译器枚举顺序；X5 分割根据真实输入尺寸撤销 letterbox，修复原来硬编码 640 的掩码缩放。这些修正仍须上板复验精度。原 Benchmark 表作为历史证据保留，不代表新代码已重新实测。
