[English](README.md) | [简体中文](README_cn.md)

# LaneNet C++ 推理

本入口保留 S 分支的 DNN/UCP 实现，并分离图像前处理、推理、结果解析、可视化和文件读写。输出为嵌入特征图与车道二值标签，**不执行**实例聚类或曲线拟合。

<a id="supported-boards"></a>
## 支持的板卡

| 目标 | 已发布契约 | 当前验证状态 |
| --- | --- | --- |
| S100 | `s:lanenet:s100/lanenet256x512.hbm` | 仅主机契约及 SDK 故障注入测试；板端推理 not-run |
| S100P / S600 / X5 | 本 sample 无对应资产 | 显式拒绝，不回退至 S100 |

启动器与原生资源管理器均在调用 SDK 前检查板卡身份，启动器还会在构建前检查。`--dry-run` 和 `--list-models` 不需要板卡或 SDK。`auto` 仅解析到唯一已发布的 S100 契约，不代表当前主机满足运行条件。

<a id="dependencies"></a>
## 依赖

请准备匹配的 S100 DNN/UCP 开发环境、CMake >=3.16、C++17 编译器、OpenCV 开发库（`core`、`imgproc`、`imgcodecs`）及线程库。启动器使用 Python 3 和仓库共享清单支持，不自动安装依赖或下载模型；原生可执行文件推理时不依赖 Python。

CMake 查找 `hobot/dnn/hb_dnn.h`、`hobot/hb_ucp.h`、`dnn` 和 `hbucp`。非标准安装可在手动配置 CMake 时设置 `DNN_INCLUDE_DIR`、`UCP_INCLUDE_DIR`、`DNN_LIBRARY`、`UCP_LIBRARY` 和 `OpenCV_DIR`。头文件与库必须来自同一 SDK 环境。[测试目录](../../tests)中的伪头文件仅用于故障注入，不能替代 SDK，也不是 ABI 兼容性证据。

<a id="build"></a>
## 构建

以下命令均从仓库根目录执行。先查看解析结果，不构建、不加载 SDK：

```bash
python3 samples/vision/lanenet/runtime/cpp/launcher.py --target s100 --dry-run --build
```

按[模型说明](../../model/README_cn.md)显式准备模型。在依赖齐备的 S100 上，以下命令检查身份、模型和输出路径，再构建至 `runtime/cpp/build/s100` 并执行：

```bash
bash samples/vision/lanenet/runtime/cpp/run.sh --target s100 --build --output outputs/lanenet_cpp_first
```

如需指定 SDK 路径，可手动对本目录执行 CMake 配置并传入上述查找变量，然后构建 `lanenet` 目标。构建本身不执行推理、不证明板端行为。本次迁移尚未在真实 SDK/OpenCV 开发环境中构建完整可执行文件。

<a id="run"></a>
## 运行

构建成功后，省略 `--build` 复用 sample 的二进制：

```bash
bash samples/vision/lanenet/runtime/cpp/run.sh --target s100 --output outputs/lanenet_cpp_next
```

显式指定二进制和模型时，将文件绑定到已发布契约。清单目前没有发布方校验和，本地摘要只能标识运行所用字节，不能认证其来源。

```bash
python3 samples/vision/lanenet/runtime/cpp/launcher.py --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/models/lanenet256x512.hbm --binary /data/bin/lanenet --test-img samples/vision/lanenet/test_data/lane.jpg --output outputs/lanenet_cpp_external
```

每次使用新输出目录。成功返回 0 的前提是生成原生 `report.json`；入口错误返回 2，原生非零返回码则保留。如果在创建结果目录前失败，仅向终端输出日志，不会保存启动报告；排障时请保留终端输出。

<a id="parameters"></a>
## 参数

以下为 Python 启动器（也是 `run.sh` 的入口）的默认值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--target` | `s100` | `auto` 或 S100 契约；其余可选目标显式拒绝 |
| `--asset-id` | `null` | 自动选择已发布资产；指定外部模型时必填 |
| `--model-path` | `null` | 解析至 sample 的 `model/s100/lanenet256x512.hbm` |
| `--test-img` | `samples/vision/lanenet/test_data/lane.jpg` | OpenCV 读取的 BGR 图像 |
| `--output` | `outputs/lanenet_cpp` | 新结果目录 |
| `--instance-save-path` | `null` | 可选的额外嵌入特征显示图 |
| `--binary-save-path` | `null` | 可选的额外二值显示图 |
| `--binary` | `null` | 解析至 `runtime/cpp/build/s100/lanenet` |
| `--build` | `false` | 显式构建，与 `--binary` 互斥 |
| `--list-models` | `false` | 只列出清单身份，不执行 |
| `--dry-run` | `false` | 只打印解析后的命令，不执行 |

额外图片路径必须不存在、互不相同，且不能覆盖结果文件或启动日志。启动器先解析相对路径，再从仓库根目录执行原生二进制；Shell 包装入口会先切换到仓库根目录。

直接运行二进制时，`--model-path` 和 `--test-img` 必填；`--target`、`--output` 分别默认为 `s100`、`outputs/lanenet_cpp`，额外图片路径默认空。兼容原有下划线参数（`--model_path`、`--test_img`、`--instance_save_path`、`--binary_save_path`）以及 `--key=value`。二进制自身没有清单选择、自动构建、下载或摘要记录功能；需要运行溯源请使用启动器。UCP 保留源实现的默认优先级与 ANY 核选择，不增加仅 Python 入口支持的调度参数。

<a id="interface-lifecycle"></a>
## 接口与生命周期

[LaneNetTask](inc/lanenet.hpp)仅提供 `pre_process`、`forward`、`post_process`、`predict`。调用者注入原始推理回调；[ModelRunner](inc/model_runner.hpp)持有模型、缓冲区和推理任务。捕获资源管理器的回调，其生命周期不得长于资源管理器本身。

| 阶段 | 输入 | 输出 / 契约 |
| --- | --- | --- |
| `pre_process` | 非空 `CV_8UC3` BGR 图像 | 独立连续 float32 NCHW `[1,3,256,512]`；BGR→RGB、INTER_AREA 缩放、/255、ImageNet 归一化 |
| `forward` | 预处理后的 float 向量 | 独立持有字节的原始张量，保留实际形状、类型、字节步长及分配大小 |
| `post_process` | 原始张量向量 | float32 CHW `[3,256,512]` 嵌入特征和 uint8 `[256,512]` 标签；标签必须为 0 或 1 |
| `predict` | BGR 图像 | 组合上述三阶段，不写文件或绘图 |

原生输出必须有唯一的 float32 `[1,3,256,512]` 嵌入张量，以及唯一的 int64 `[1,1,256,512]` 或 `[1,256,512]` 二值张量。按唯一形状和类型绑定角色，不假设输出索引；有歧义则报错。其他实际观察到的数值输出按索引原样保留，不虚构名称或语义；原生实现不查询输出名称。源文档称有三个输出，但没有说明第三个输出身份。[Python 入口](../python/README_cn.md)则按其 SDK 暴露的两个必需名称绑定。

张量复制检查所有字节步长，包括宽度填充，并拒绝重叠或越界布局。资源管理器检查 SDK 返回值，对部分初始化、分配、提交和等待失败使用作用域清理。析构清理为尽力执行；主机夹具不能证明厂商释放接口行为。返回张量独立持有数据，推理任务释放后仍然有效。

<a id="results-interpretation"></a>
## 结果解释

| 文件 | 含义 |
| --- | --- |
| `raw_output_N.npy` | 全部实际输出，去除内存填充并保留精确类型及形状；int64 不经浮点转换 |
| `embedding.npy` | 原始 float32 `[3,256,512]` 嵌入特征，未裁剪、未聚类 |
| `binary.npy` | 已验证的 0/1 uint8 `[256,512]` 标签 |
| `instance_pred.png` | 裁剪后的嵌入通道显示，颜色不代表车道 ID |
| `binary_pred.png` | 二值标签乘 255 |
| `report.json` | 实际张量元数据、输出角色索引、模型名称及处理边界 |
| `launch-report.json` | 启动命令、UTC 时间段、返回码及模型/输入/二进制/报告的实际摘要 |
| `native.stdout.log`、`native.stderr.log` | 结果目录存在时保存的完整原生输出流 |

输出图像均保持 256×512 模型网格。不恢复原始分辨率，不执行聚类、跟踪、曲线拟合、精度测量或延迟测量。嵌入显示先裁剪至 [0,1]，乘 255，再按最近值、半数取偶舍入。保留原生显示约定，同时明确替换原 Python 的溢出回绕和截断行为；比较原始嵌入时应与显示图分开。

主机测试编译张量/NPY 辅助模块，并用可控伪 SDK 调用测试真实资源管理器，包括部分分配和任务失败；还验证了超过 float64 精确整数范围的 int64 值。完整原生 OpenCV/SDK 构建及板端推理仍为 **not-run**，上述测试不能证明可部署。依赖缺失时检查 CMake 查找结果；身份拒绝时检查物理板卡；元数据拒绝时保留真实元数据，不通过改名或强制目标绕过检查。
