# YOLOE PF 模型转换

<a id="source-model"></a>
## 源模型

本目录从**已经导出到本地**的 YOLOE PF ONNX 生成校准数据和可核查的 OE 配置，不自动下载权重。11 系列使用 DFL16，26 系列直接输出 LTRB；两者都有固定顺序的 4585 类词表、三个步长（8/16/32）、32 个掩码系数和原型输出。文本提示、视觉提示模型不适用。

三条路线（X5 E11、S E11、S E26）的配方均在下文提供。统一准备入口不请求 `remove_node_type`/`remove_node_name`，以此保留浮点输出节点。已发布 S 制品输出为量化数据，与本地重新生成的浮点模型分开保存。

<a id="toolchain-targets"></a>
## 工具链与目标

| `--target` | `--variant` | March | 编译器 | 校准数据 |
| --- | --- | --- | --- | --- |
| x5 | 11s、11m、11l | bayes-e | `hb_mapper makertbin` | RGB float32 原始二进制，0..255 |
| s100 | 11s | nash-e | `hb_compile` | NPY RGB float32，0..1 |
| s100 | 26n、26s、26m、26l、26x | nash-e | `hb_compile` | NPY RGB float32，0..1 |
| s100p | 26n、26s、26m、26l、26x | nash-m | `hb_compile` | NPY RGB float32，0..1 |

ONNX 输入固定为 `[1,3,640,640]` RGB float32，运行时输入为 NV12。拒绝 S600、X5 E26、S100P E11，不自动回退到其他平台。准备入口支持全部 14 种目标/变体组合。X5 配方提供 11s YAML，并说明 11l 需增加第二个 attention 配置；应用到真实模型仍需实际编译验证。各路线的工具链版本：S E26 配方使用经过验证的 OE 3.7.0 CPU 容器；S E11 量化路线要求 D-Robotics OpenExplorer >= 3.0.31、Ultralytics >= 8.3.0；浮点导出路线的依赖固定在 `requirements-export.txt` 中。

通用 OE 资源：[OE 环境文档](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview) 与[工具链下载](https://toolchain.d-robotics.cc/)。

主机准备使用 Python 3.10+，建议独立环境：

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-host.txt
python3 samples/vision/yoloe/conversion/prepare.py --help
```

这些依赖仅用于 ONNX 检查、图片处理和 YAML，不安装 OE、PyTorch、Ultralytics 或 ONNX Runtime。编译时进入对应 X5/S OE 环境，按需补充准备依赖；不要直接覆盖工具链环境自身的编译依赖。

<a id="export"></a>
## 导出

统一 [export.py](export.py) 加载已经存在的本地 PF 权重，核对头类型、尺寸和有序词表，在新目录保存导出结果。

权重获取：克隆上游 YOLOE 仓库（<https://github.com/um-assn/yoloe.git>），执行 `pip install -r requirements.txt && pip install ultralytics`，再从 Ultralytics assets release 下载 PF 权重，例如 `wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt`（其他 E11 尺寸把 `11s` 换成 `11m`/`11l`）。S E11 量化路线另可使用 Model Zoo 导出脚本 <https://github.com/D-Robotics/rdk_model_zoo/blob/main/demos/Seg/YOLOE-11-Seg-Prompt-Free/YOLOE-11-Seg-Prompt-Free_YUV420SP/cauchy_yoloe11segPF_export.py>，以等价模块替换方式完成导出、无需重训。

先安装独立的导出依赖，其中 Ultralytics 固定为 E26 配方使用的版本：

```bash
# cwd: repository root; use a separate CPU export environment
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-export.txt
python3 samples/vision/yoloe/conversion/export.py --help
python3 samples/vision/yoloe/conversion/export.py \
  --weights /work/checkpoints/yoloe-11s-seg-pf.pt --variant 11s \
  --output-dir /work/export11s --test-image samples/vision/yoloe/test_data/office_desk.jpg
python3 samples/vision/yoloe/conversion/export.py \
  --weights /work/checkpoints/yoloe-26n-seg-pf.pt --variant 26n \
  --output-dir /work/export26n --test-image samples/vision/yoloe/test_data/office_desk.jpg
```

`--weights`、`--variant`、`--output-dir` 必填；支持 11s/m/l、26n/s/m/l/x 八种变体。`--threads` 默认 2，必须为正整数。不传 `--test-image` 时使用 seed 0 的 CPU 随机输入；建议提供图片，让对照结果更有实际意义。不隐式下载权重或编译，拒绝复用已有输出目录。导出失败非零退出（已处理的输入/校验错误为 2）；开始准备后产生的失败会在 `export.json` 留下阶段状态，不写成成功。

E11 使用源 cv2/cv3/cv5 分支、DFL16 和 opset 11；E26 使用 one2one 分支、直接 LTRB 和 opset 17。Linear 词表权重以等价 1×1 卷积执行，不替换权重参数；已有词表卷积直接保留。原始头返回全部 anchor 和十个 NHWC tensor，不做 proposal 筛选。骨干网络的层连接和 `model.<index>` 节点名保留，供 X5 attention 配置识别。

两类模型都先将全部原始 anchor、Top-K 前的解码 tensor 和原型与上游静态 PF 头比较（rtol/atol 1e-4），再检查 ONNX 图，并将十个 ONNX Runtime CPU 输出与 PyTorch 比较（rtol/atol 2e-3）。比较时设置 `ORT_DISABLE_ALL`，在关闭 CPU 融合的条件下检查导出图。生成 `yoloe_<variant>_seg_pf.onnx`、`yoloe_<variant>_seg_pf.names`、`export.json`；记录包含权重、输入、ONNX、词表哈希、依赖版本、逐输出最大绝对误差和 `status=float_checked`。导出与平台无关、不写入硬件 march；march 由后续按目标准备与编译步骤选择。

E26 还要求选中的 `(anchor, class)` 集合与上游完全一致，再按这个身份比较选中值。浮点舍入引起的名次变化记录在 `order_identical=false` 和 `reordered_rows` 中。即使分数接近，只要新增/遗漏 anchor 或类别就失败。导出接口提供稠密 tensor；最终 Top-K 排序通过运行时及数据集评估器测量。

以 E11s/m/l 与 E26n/s/m/l/x 八种权重及随附 `office_desk.jpg` 进行的参考性浮点导出对照，在 E26 m/l/x 上分别观察到 2/4/2 行 Top-K 名次变化，选中的 `(anchor, class)` 集合保持一致。该对照在 CPU 上使用单张图片。数据集精度及编译后 BIN/HBM 推理指标通过评估器测量。

`prepare.py` 执行 ONNX checker，拒绝外置 tensor 文件和动态形状，并要求词表与 [classes.names](../test_data/classes.names) 逐字节一致。输出按唯一形状识别，不依赖物理输出排列：

| 角色 | E11 NHWC 形状 | E26 NHWC 形状 |
| --- | --- | --- |
| 分类，步长 `s` | `[1,640/s,640/s,4585]` | 同左 |
| 框，步长 `s` | `[1,640/s,640/s,64]` | `[1,640/s,640/s,4]` |
| 掩码系数，步长 `s` | `[1,640/s,640/s,32]` | 同左 |
| 原型 | `[1,160,160,32]` | 同左 |

`s` 取 8、16、32，共十个 float32 输出。形状一致证明的是 tensor 接口；权重尺寸、架构、标签语义与精度由导出器检查和评估器验证。报告用 `variant_declared` 标记此阶段的尺寸为调用者声明；请将导出 metadata 与准备目录一并保存。

<a id="calibration"></a>
## 校准

通过 `--cal-images` 提供有代表性的 JPG/JPEG/PNG/BMP 图片。脚本递归按相对路径排序，以等间隔索引选择最多 `--sample-count` 张（默认 100）。不足 20 张会记录警告；被选中的图片无法读取则失败。合成测试图片不能作为模型精度校准集。

几何处理与运行时共用：E11 按截断后的缩放尺寸 letterbox、填充 127；E26 对缩放尺寸四舍五入、填充 114。均从 BGR 转 RGB、HWC 转 NCHW。工具链要求的范围与格式不同：

- **X5：** `.rgb` 为无文件头的 float32 原始二进制，范围 0..255。`preprocess_on=False`、`norm_type=data_scale`、`scale_value=1/255` 描述 Mapper 融合归一化的方式，见 [X5 校准规则](https://developer.d-robotics.cc/oe_x5_doc/cn/oe_mapper/source/ptq/ptq_usage/prepare_calibration_data.html)。
- **S：** `.npy` 保存与原浮点模型一致的 0..1 float32。运行时 NV12 图像仍需要 `scale_value=1/255`；校准 NPY 和运行时图片是不同输入阶段，不是重复归一化，见 [S 校准规则](https://developer.d-robotics.cc/oe_s_doc/guide/ptq/ptq_usage/prepare_data)。

```bash
# cwd: repository root; all /work paths are user-supplied inputs or new outputs
python3 samples/vision/yoloe/conversion/prepare.py \
  --onnx /work/export26n/yoloe_26n_seg_pf.onnx \
  --names /work/export26n/yoloe_26n_seg_pf.names \
  --target s100 --variant 26n --cal-images /work/calibration-images \
  --sample-count 100 --output-dir /work/yoloe26n-s100-config
```

该命令只做准备，报告 `status=config_only`，不调用编译器。E11 改用对应导出的 ONNX/names，选择 `--variant 11s --target x5` 或 `s100`，其他尺寸按表选择。每次调用都要求新的输出目录，后续加编译也不能复用旧目录；失败产生的部分目录保留供排查。

<a id="compile"></a>
## 编译

在匹配的 OE 环境加 `--compile`。可选 `--compiler` 仅接受可执行文件路径，不接受一段 shell 命令。例如：

```bash
# cwd: repository root inside the S OE environment
python3 samples/vision/yoloe/conversion/prepare.py \
  --onnx /work/export26n/yoloe_26n_seg_pf.onnx \
  --names /work/export26n/yoloe_26n_seg_pf.names \
  --target s100 --variant 26n --cal-images /work/calibration-images \
  --output-dir /work/yoloe26n-s100-build --compile
```

实际命令为 S 的 `hb_compile -c <absolute config.yaml>` 或 X5 的 `hb_mapper makertbin --model-type onnx --config <absolute config.yaml>`，cwd 为准备目录。YAML 以绝对路径引用 ONNX、词表和校准输入；请在编译器可见的文件系统/容器内完成准备，确保所有被引用的输入在容器内可访问。

E26 配方验证所用的 S OE 3.7.0 CPU 容器：

```bash
REPO_DIR=/path/to/rdk_model_zoo
docker run --rm -it --shm-size=2g \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

| 策略 | X5 E11 | S E11 | S E26 |
| --- | --- | --- | --- |
| 校准 | default | default，Softmax int8 | KL，全部节点 int8 |
| 编译 | latency、O3、core 1、jobs 4 | latency、O2、core 1、jobs 15、advice 1 | latency、O2、core 1、jobs 4 |
| Padding | pyramid 输入 | 输入/输出无 padding | 输入无 padding，允许输出 padding |
| 删除输出节点请求 | 无 | 无 | 无 |

X5 只为真实 ONNX 中存在的节点添加 int16 attention 配置：所有 E11 尺寸使用 `/model.10/m/m.0/attn/Softmax`，11l 另需 `/model.10/m/m.1/attn/Softmax`。每个缺失节点都会在 `source attention node absent:...` 警告中单独列出。不能把另一节点改名来冒充。S E11 量化配方 `config_ultralytics_YOLOE_Seg_YUV420SP_NV12.yaml` 的设置为：NV12 运行时输入、`scale_value 0.003921568627451`、默认校准加 softmax-int8 的 `quant_config`、latency/O2、`jobs: 15`、`advice: 1`、输入/输出无 padding；其生效的 `remove_node_name` 名单沿用 v8 头部编号（`/model.23/...`），而 YOLOE-11 的头在 `/model.22/...` 下，这些删除名与 11 系列图不匹配。本浮点路线直接不请求删除。相关选项删除边界算子的含义见 [S 模型修改规则](https://developer.d-robotics.cc/oe_s_doc/guide/model_deployment_guidance/model_deployment_principle_process/model_modify)。不删除表达浮点输出意图，最终仍须读取编译制品的真实 metadata。

<a id="validation"></a>
## 验证

退出码：准备完成或编译退出 0 且产生非空文件时为 0；捕获到编译失败或缺失制品为 1；输入无效、依赖/编译器缺失为 2。编译成功仅报告 **`compiled_unverified`**，保留 `observed_output_dtype=null`，`board` 与 `dataset_accuracy` 不设值。文件名 `_float` 是预期协议，不是实际精度证明。

对已发布的 X5 E11 制品，用 `hb_perf` 可视化加 `hrt_model_exec model_info` 检查验证（准备 BIN 后在匹配的 X5 镜像中运行）：

```bash
hb_perf samples/vision/yoloe/model/x5/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
hrt_model_exec model_info --model_file samples/vision/yoloe/model/x5/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
```

检查编译制品的实际 metadata，并用代表性输入比较十个输出与浮点模型。统一运行时进一步检查 target、NV12 输入和十个 NHWC float32 输出角色；整数输出、形状或平台不符直接拒绝，不在后处理中加入手动反量化。使用 `conversion.json` 中制品摘要显式指定本地文件：

```bash
# cwd: repository root on the matching board; replace the path and digest
python3 samples/vision/yoloe/runtime/python/main.py \
  --target s100 --variant 26n --model-path /work/model.hbm \
  --local-float-sha256 REPLACE_WITH_64_HEX_SHA256
```

主机测试使用合成图执行真实 ONNX 校验，检查校准像素、14 种配置和模拟编译器成功/失败捕获。安装准备依赖后运行：

```bash
# cwd: repository root
python3 -m unittest discover -s samples/vision/yoloe/tests
# Additionally, in the export environment:
python3 -m unittest discover -s samples/vision/yoloe/conversion/tests
```

<a id="artifacts"></a>
## 产物

每个准备目录包含 `source/model.onnx`、`source/classes.names`、`calibration/`、`calibration.json`、`config.yaml`、`conversion.json`。记录以 SHA-256 绑定原图、生成 tensor、ONNX、词表和 YAML。编译额外保存 stdout/stderr 合并完整日志 `compile.log`，记录精确 argv/cwd、UTC 起止时间和退出码；成功时在 `conversion.json` 写制品路径、哈希及字节数。

预期路径为 `compiler_output/yoloe_<variant>_seg_pf_<march-without-hyphen>_640x640_nv12_float.bin`（X5）或 `.hbm`（S）。本地转换与[原始发布制品](../model/README_cn.md)分开保存，不会自动建立新的发布身份或板测记录。

<a id="known-gaps"></a>
## 准备要求

浮点 S HBM 通过上述本地路线生成；当前没有发布浮点 S HBM，已发布的量化 S 制品属于不同输入。导出对照在未优化的 ONNX Runtime CPU 上执行；优化引擎行为、更多输入与 X5 11m/11l 编译需自行运行验证。图接口校验不识别错误标注的权重尺寸或标签语义，这部分由导出器检查和评估器覆盖。[C++ 运行时](../runtime/cpp/README_cn.md)的 SDK 构建与板端执行见其指南。[统一评估器](../evaluator/README_cn.md)提供显式数据集映射与计分。

每次转换保存本地权重、导出 ONNX 和生成 HBM 的 SHA-256。原始量化输出路线与浮点输出路线分别保存制品记录。
