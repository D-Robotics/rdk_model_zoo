# ResNet18 评估

评估有两个目的：确认目标板卡可以按预期张量契约执行选择的制品，以及在
明确数据集和工具链后测量精度或延迟。主机测试只覆盖契约逻辑，不模拟
`hbm_runtime`，也不替代板卡验收。

## 主机检查

在仓库根目录运行完整 ResNet 测试：

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

测试覆盖 Manifest 选择、严格运行时元数据绑定、缩放几何、packed/split
NV12 布局、注入执行、安全标签、兼容 wrapper 返回形状和确定性的 score 解码。
导出 smoke test 与板端编译和精度验证是不同的检查。

## 板端功能检查

从匹配的 Manifest 行准备制品，在对应板卡运行 canonical Python 命令。例如
在 X5 上：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

在 S100 或 S600 上替换为对应的 `s:resnet18:<target>/...` 引用、制品路径和
`platforms/s/datasets/imagenet/` 标签。保存板卡身份、模型引用、运行时元数据、
raw F32 score 张量、Top-K 输出、图片路径、缩放方式和完整命令。板卡连接、制品
或运行时缺失时状态记为 `not-run`；主机测试通过不会改变该状态。

旧 Python 入口可用于直接兼容性比较：

```bash
python3 platforms/x5/samples/vision/resnet/runtime/python/main.py --help
python3 platforms/s/samples/vision/resnet18/runtime/python/main.py --help
```

比较时要使用相同图片、模型字节、标签、缩放方式和 Top-K。先比较类别 ID
和 raw score，再比较打印的标签格式。兼容类返回旧的元组/列表形状，但预处理
和解码使用统一实现。

## S 系列原生检查

合并后的 C++ 源码只在 S100 或 S600 上编译运行：

```bash
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动器检查模型、图片和标签文件，执行 CMake 并运行二进制，不安装 `gflags`、
OpenCV 或 Horizon DNN，也不下载模型。S600 的 `MODEL_PATH` 和 `BUILD_DIR` 用法
见上级 README。若需独立审计兼容构建，可对
`platforms/s/samples/vision/resnet18/runtime/cpp` 执行 CMake；该目录现在添加
canonical CMake 目标并保留旧输出位置。

原生输出应使用相同 S 制品、`zebra_cls.jpg`、标签文件和 `--top_k 5` 与旧 S18
二进制比较。保存完整配置/编译命令和 Top-K 行。C++ 源码是审计过的 S18 实现
合并副本，并使用共享的 `platforms/s/utils/c_utils` 源文件；审计源码没有 X5
对应的 C++ 基线。

## 精度和性能

测量 ImageNet validation 时，使用 OE 转换参考中相同的数据预处理：224x224
输入和目标板卡的 NV12 契约。记录图片列表、标签映射、制品引用、板卡身份以及
score 是否使用 legacy softmax 解码。已发布的 X5 旧评估数据如下：

| 制品/评估 | Top-1 | 延迟 | FPS | 来源 |
| --- | ---: | ---: | ---: | --- |
| ResNet18 float 参考 | 71.5% | 2.95 ms | 449+ | X5 旧 evaluator README |
| ResNet18 quantized 参考 | 70.5% | 2.95 ms | 449+ | X5 旧 evaluator README |

这些是历史发布值，不是每次检出或重建制品的新结果。S18 README 只提供定性
冒烟检查：`zebra_cls.jpg` 应对斑马类别产生有限且非零的分数，没有发布完整
ImageNet 精度。转换时使用的 OE 工具（包括 `hb_perf` 和 `hrt_model_exec`）
应保存完整日志，再报告延迟或模型级输出。

## 结果解释

统一契约只接受一个 F32、1000 类输出张量。X5 元数据使用 `prob`，形状
`[1,1000,1,1]`；S100/S600 使用 `output`，形状 `[1,1000]`。两个 Python
源码 wrapper 都调用 softmax。由于现有转换源不能证明 X5 归一化发生位置，
报告时同时保留 raw output 和解码结果，并将策略标记为
`unverified_score_vector` 上的 `legacy_softmax`。

不要根据缺失的板测、同名但不同字节的制品或 ResNet50/152 旧文件发布新的
精度/性能结论。阻塞时记录 `not-run` 及原因，并将历史参考与新测量分开保存。
