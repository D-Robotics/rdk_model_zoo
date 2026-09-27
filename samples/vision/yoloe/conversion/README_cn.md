# YOLOE PF 模型转换

<a id="source-model"></a>
## 源模型

本目录从**已经导出到本地**的 YOLOE PF ONNX 生成校准数据和可核查的 OE 配置，不自动下载权重。11 系列使用 DFL16，26 系列直接输出 LTRB；两者都有固定顺序的 4585 类词表、三个步长（8/16/32）、32 个掩码系数和原型输出。文本提示、视觉提示模型不适用。

原始配方保留在 [X5 E11](../../../../platforms/x5/samples/vision/yoloe/conversion/README_cn.md)、[S E11](../../../../platforms/s/samples/vision/yoloe11_seg/conversion/README.md)、[S E26](../../../../platforms/s/samples/vision/yoloe26_seg/conversion/README_cn.md)。统一转换路线保留浮点输出节点，不设置 `remove_node_type` 或 `remove_node_name`。S 原始发布制品输出为量化数据，不能冒充本路线新生成的浮点模型。

<a id="toolchain-targets"></a>
## 工具链与目标

| `--target` | `--variant` | March | 编译器 | 校准数据 |
| --- | --- | --- | --- | --- |
| x5 | 11s、11m、11l | bayes-e | `hb_mapper makertbin` | RGB float32 原始二进制，0..255 |
| s100 | 11s | nash-e | `hb_compile` | NPY RGB float32，0..1 |
| s100 | 26n、26s、26m、26l、26x | nash-e | `hb_compile` | NPY RGB float32，0..1 |
| s100p | 26n、26s、26m、26l、26x | nash-m | `hb_compile` | NPY RGB float32，0..1 |

ONNX 输入固定为 `[1,3,640,640]` RGB float32，运行时输入为 NV12。拒绝 S600、X5 E26、S100P E11，不自动回退到其他平台。这 14 种选择已有主机配置检查，**不等于 OE 编译验收**。X5 源分支只有 11s YAML，将该策略用于 11m/11l 仍需实际编译验证。S26 源记录使用 OE 3.7.0；其他路线的最低可用工具链版本尚未核定。

主机准备使用 Python 3.10+，建议独立环境：

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/conversion/requirements-host.txt
python3 samples/vision/yoloe/conversion/prepare.py --help
```

这些依赖仅用于 ONNX 检查、图片处理和 YAML，不安装 OE、PyTorch、Ultralytics 或 ONNX Runtime。编译时进入对应 X5/S OE 环境，按需补充准备依赖；不要直接覆盖工具链环境自身的编译依赖。

<a id="export"></a>
## 导出

权重导出器尚在统一迁移中。当前保留 [E11 导出器](../../../../platforms/x5/samples/vision/yoloe/conversion/onnx_export/export_yoloe11seg_bpu.py) 和 [E26 导出器](../../../../platforms/s/samples/vision/yoloe26_seg/conversion/onnx_export/export_yoloe26_seg_pf.py) 作为源实现；独立依赖和版本要求见上述源转换 README，其中 E26 固定 Ultralytics 8.4.127。把权重放入新的工作目录，脚本拒绝覆盖已有输出。

```bash
# cwd: repository root; local PF checkpoint and source export dependencies required
python3 platforms/x5/samples/vision/yoloe/conversion/onnx_export/export_yoloe11seg_bpu.py \
  --weights /work/export11/yoloe-11s-seg-pf.pt --imgsz 640 --opset 11
python3 platforms/s/samples/vision/yoloe26_seg/conversion/onnx_export/export_yoloe26_seg_pf.py \
  --weights /work/checkpoints/yoloe-26n-seg-pf.pt --size n \
  --output-dir /work/export26n
```

E11 默认在权重旁生成 `.onnx`、`.names`、`.export.json`，但不证明精度一致性。E26 生成 `yoloe_26n_seg_pf.onnx/.names/.json`，检查原始头与上游输出并比较 ONNX Runtime 浮点结果；旧 metadata 固定写 nash-e，但导出的图并非 HBM，准备时仍须显式选择真实 target。本轮迁移尚未用真实权重执行这两个导出器。

`prepare.py` 执行 ONNX checker，拒绝外置 tensor 文件和动态形状，并要求词表与 [classes.names](../test_data/classes.names) 逐字节一致。输出按唯一形状识别，不依赖物理输出排列：

| 角色 | E11 NHWC 形状 | E26 NHWC 形状 |
| --- | --- | --- |
| 分类，步长 `s` | `[1,640/s,640/s,4585]` | 同左 |
| 框，步长 `s` | `[1,640/s,640/s,64]` | `[1,640/s,640/s,4]` |
| 掩码系数，步长 `s` | `[1,640/s,640/s,32]` | 同左 |
| 原型 | `[1,160,160,32]` | 同左 |

`s` 取 8、16、32，共十个 float32 输出。形状一致仅证明接口，不能证明权重尺寸、架构、标签语义或精度。报告用 `variant_declared` 明确这是调用者声明；请将导出 metadata 与准备目录一并保存。

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

实际命令为 S 的 `hb_compile -c <absolute config.yaml>` 或 X5 的 `hb_mapper makertbin --model-type onnx --config <absolute config.yaml>`，cwd 为准备目录。YAML 使用绝对路径，请在编译器可见的文件系统/容器内生成；仅移动 YAML 不会同时迁移其引用的输入。

| 策略 | X5 E11 | S E11 | S E26 |
| --- | --- | --- | --- |
| 校准 | default | default，Softmax int8 | KL，全部节点 int8 |
| 编译 | latency、O3、core 1、jobs 4 | latency、O2、core 1、jobs 15、advice 1 | latency、O2、core 1、jobs 4 |
| Padding | pyramid 输入 | 输入/输出无 padding | 输入无 padding，允许输出 padding |
| 删除输出节点请求 | 无 | 无 | 无 |

X5 仅在真实 ONNX 存在 `/model.10/m/m.0/attn/Softmax` 时添加源 int16 attention 配置，否则报告 `source attention node absent`。不能把另一节点改名来冒充。S11 源 YAML 的删除名单残留 v8 节点，本浮点路线直接不请求删除。相关选项删除边界算子的含义见 [S 模型修改规则](https://developer.d-robotics.cc/oe_s_doc/guide/model_deployment_guidance/model_deployment_principle_process/model_modify)。不删除表达浮点输出意图，最终仍须读取编译制品的真实 metadata。

<a id="validation"></a>
## 验证

退出码：准备完成或编译退出 0 且产生非空文件时为 0；捕获到编译失败或缺失制品为 1；输入无效、依赖/编译器缺失为 2。编译成功仅报告 **`compiled_unverified`**，保留 `observed_output_dtype=null`、`board=not-run`、`dataset_accuracy=not-run`。文件名 `_float` 是预期协议，不是实际精度证明。

验收前必须读取真实制品 metadata，并用代表性输入比较十个输出与浮点模型。统一运行时进一步检查 target、NV12 输入和十个 NHWC float32 输出角色；整数输出、形状或平台不符直接拒绝，不在后处理中加入手动反量化。使用 `conversion.json` 中制品摘要显式指定本地文件：

```bash
# cwd: repository root on the matching board; replace the path and digest
python3 samples/vision/yoloe/runtime/python/main.py \
  --target s100 --variant 26n --model-path /work/model.hbm \
  --local-float-sha256 REPLACE_WITH_64_HEX_SHA256
```

主机测试使用合成图执行真实 ONNX 校验，检查校准像素、14 种配置和模拟编译器成功/失败捕获；不运行真实模型，也不证明 OE 配方可用。安装准备依赖后运行：

```bash
# cwd: repository root
python3 -m unittest discover -s samples/vision/yoloe/tests
```

<a id="artifacts"></a>
## 产物

每个准备目录包含 `source/model.onnx`、`source/classes.names`、`calibration/`、`calibration.json`、`config.yaml`、`conversion.json`。记录以 SHA-256 绑定原图、生成 tensor、ONNX、词表和 YAML。编译额外保存 stdout/stderr 合并完整日志 `compile.log`，记录精确 argv/cwd、UTC 起止时间和退出码；成功时在 `conversion.json` 写制品路径、哈希及字节数。

预期路径为 `compiler_output/yoloe_<variant>_seg_pf_<march-without-hyphen>_640x640_nv12_float.bin`（X5）或 `.hbm`（S）。本地转换与[原始发布制品](../model/README_cn.md)分开保存，不会自动建立新的发布身份或板测记录。

<a id="known-gaps"></a>
## 已知缺口

本统一路线尚未执行真实权重导出、OE 编译、编译输出检查、数据集精度或板端推理；也没有已发布浮点 S HBM。源导出器仍待统一收编，X5 11m/11l 策略和 attention 节点缺失警告仍需真实模型检查。图接口校验不能识别错误标注的权重尺寸或标签语义。C++ 迁移、统一数据集评估器是另外的未完成工作，准备入口不代表它们已经完成。
