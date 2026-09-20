# ResNet18 C++ 运行时（S 系列）

合并后的 S 系列 ResNet18 原生运行时：审计过的 S18 `hbDNNInferV2` 流程、
图像预处理、NV12 张量构建与 Top-K 输出代码以源码形式保留在 canonical
sample 中。旧 S18 C++ 目录只是薄薄的 CMake 兼容配置入口，不维护第二份
实现。

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | 状态 |
| --- | --- |
| S100 | supported-verified（2026-09-17 构建并运行；Top-5 与源基线一致） |
| S600 | supported-not-run（同一源码与 SoC 检测；板卡连接不可用） |
| X5 | not-supported（审计基线中不存在 X5 的 C++ 源码） |

CMake 读取 `/sys/class/boardinfo/soc_name` 并定义原源码使用的 SoC 宏；
身份文件不可读视为错误，不做回退。

<a id="dependencies"></a>
## 依赖

板端镜像需要：CMake 与 C++17 编译器；OpenCV 开发头文件/库；`gflags` 与
`fmt` 开发库；`/usr/hobot/include` 下的 Horizon DNN 头文件与
`/usr/hobot/lib` 下的库（`hbDNN`、`hbucp`）。工具实现来自 canonical CMake
目标引用的既有 `platforms/s/utils/c_utils` 文件。启动脚本不安装系统包、
不修改 SDK、不下载模型。

<a id="build"></a>
## 构建

启动脚本会自动构建；手动构建（cwd：仓库根目录）：

```bash
# 成功判据：build 目录中生成 resnet18 二进制
cmake -S samples/vision/resnet/runtime/cpp \
  -B samples/vision/resnet/runtime/cpp/build
cmake --build samples/vision/resnet/runtime/cpp/build --parallel
```

兼容路径选择的是同一个 canonical 目标：

```bash
cmake -S platforms/s/samples/vision/resnet18/runtime/cpp \
  -B /tmp/resnet18-legacy-build
cmake --build /tmp/resnet18-legacy-build --parallel
```

<a id="run"></a>
## 运行

前置条件：`bash samples/vision/resnet/model/download.sh s100`（或 S600
制品）。可在任意工作目录执行：

```bash
# 输入：model/s100 制品、随仓 zebra_cls.jpg、S 系列 ImageNet 标签
# 输出：stdout 上的 Top-K 行 — 成功判据：退出码 0
bash samples/vision/resnet/runtime/cpp/run.sh
```

S600 需显式选择制品；同一检出目录服务两块板时使用独立 build 目录：

```bash
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

启动脚本先检查模型、图片与标签文件，再调用 CMake、构建 `resnet18`，
并传入绝对路径使命令在任意目录可用。

<a id="parameters"></a>
## 参数

`resnet18` 二进制的原生 gflags（启动脚本会用 sample 的绝对路径覆盖前三
项）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | 按 SoC：`/opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm`（S100）或 `/opt/hobot/model/s600/basic/resnet18_224x224_nv12.hbm`（S600） | HBM 模型路径 |
| `--test_img` | `../../../test_data/zebra_cls.jpg`（相对历史 build 布局） | BGR 测试图 |
| `--label_file` | 仓库内 S 系列 ImageNet 标签路径 | 逐行一个标签 |
| `--top_k` | `5` | 打印的类别数量 |

通过启动脚本覆盖的示例：

```bash
bash samples/vision/resnet/runtime/cpp/run.sh \
  --model_path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
  --test_img /tmp/zebra_cls.jpg \
  --label_file /tmp/imagenet_classes.names \
  --top_k 5
```

<a id="interface-lifecycle"></a>
## 接口与生命周期

`main.cpp` 创建 `Resnet18` 模型对象，加载 HBM 并提取张量元数据，通过模型
预处理（NV12 Y/UV 张量构建）转换 BGR 图像，对 S 系列输入张量调用
`hbDNNInferV2`，用 Top-K 后处理解码 F32 输出，按标签文件打印配置数量的
类别，并在作用域退出时释放 DNN 资源。重活发生在构造之后而非构造函数中；
工具实现是既有的 `platforms/s/utils/c_utils` 源码。没有后台线程，进程执行
一次同步推理。

<a id="results-interpretation"></a>
## 结果解释

二进制按标签文件逐行打印 Top-K 类别，每行含类别 ID、分数与标签；成功
退出码为 0。每次原生评估都应记录板卡身份、制品引用、完整构建/运行命令与
Top-K 输出。S600 连接不可用或制品缺失记为 `not-run`，不能用 S100 结果
替代。
