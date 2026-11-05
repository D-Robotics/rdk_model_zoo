# OSNet S 系列 Workbench 验收

本流程沿用 GUIDELINE.md v2.8 的目录、版本冻结和验收规则，任务指标改为 OSNet x1.0 的 Market1501 Rank-1/mAP。S100、S100P、S600 分别使用 nash-e、nash-m、nash-p；各自保留转换 run、HBM、板端 campaign 和候选包。平台名称不能代替 Runtime 兼容性测试。

使用 model/artifacts.json 锁定的 x1.0 Market1501 权重，经 `conversion/prepare.py export-s` 导出 opset19 空间头 ONNX，保留导出回执。输入固定为 batch=1、F32 RGB NCHW `[1,3,256,128]`：Pillow 双线性缩放，除以 255，再按 ImageNet mean/std 归一化。输出为 F32 `[1,512,1,1]`。采集原始特征，再做 L2 归一化、平方欧氏距离检索，不做 reranking。

校准使用 seed=20261029 锁定的 128 张训练集行人裁剪图，由 `conversion/prepare_s.py data` 生成 F32 NPY。评测使用全部 3,368 张 query 和 15,913 张有效 gallery；同身份同摄像头排除规则及 junk 处理见 metrics.py。提前冻结有序图片字节清单；Rank-1 和 mAP 相对 FP32 的绝对下降均不得超过 1 个百分点。

先用当前 OE 的 `hb_config_generator -f -m MODEL.onnx --march MARCH` 生成完整模板，再调用 `conversion/prepare_s.py config` 填写目标、输入、校准目录和输出目录；`--precision int16` 只设置 `quant_config.model_config.all_node_type=int16`。之后在锁定镜像 ID 中执行 `hb_compile -c`。编译器实际节点类型及回退行为保留在原始 CSV。默认 INT8 未过精度门槛的实验单独保留。必须从编译日志确认实际校准 batch=8；jobs=8 是编译并行度，不能作为校准 batch 的证据。最终 Runtime 输入仍为 batch=1。

在对应开发板直接使用原生样例：

```bash
python3 samples/vision/osnet/runtime/python/main.py \
  --target s100 --variant s100-f32-int16 --model-path /absolute/model.hbm \
  --test-img /absolute/person_crop.jpg --output feature.npy
python3 samples/vision/osnet/evaluator/main.py \
  --backend s100 --variant s100-f32-int16 --model-path /absolute/model.hbm \
  --dataset /absolute/Market1501 --output-dir /absolute/new-capture --capture-only
```

S100P/S600 需同时替换 target/backend 和 variant。主机调用 evaluator 的 `--capture-dir ... --dataset ... --output-dir ...` 对保存的完整特征打分。主机打分回执中的 board_test=not-run 表示该进程没有执行板端推理；板端 capture.json 单独记录真实推理、模型 SHA 和特征 SHA。

Runtime 使用安装的 `hrt_model_exec perf`、合法 F32 固定输入、`--core_id 1`，分别实测并发 1/2，每种条件 3 轮，每轮总计 200 帧。先核对板端帮助。3.13.6 的布尔 warmup 只忽略首帧，因此单独运行 20 帧设备预热，计时轮使用 `--enable_warmup=false`，明确记录跨进程预热口径。吞吐用 profiler 的真实帧数和墙钟，不能用并发延迟倒数推算。

C++ 使用 UCP，每个 worker 独占模型和缓冲区，编译参数包含 `-DOSNET_UCP -I/usr/hobot/include -L/usr/hobot/lib -ldnn -lhbucp`。原生程序命令：

```bash
osnet --benchmark MODEL.hbm images.dat reference.f32 result.json 1 20 200 3
osnet --benchmark MODEL.hbm images.dat reference.f32 result.json 2 20 200 3
```

私有 fixture 含 32 张已解码裁剪图和精确 resize 参考字节；reference.f32 由相同 HBM 的 Python SDK 独立生成。每个 worker 在计时前验证全部 32 张。每轮每流预热 20 帧、计时 200 帧，3 轮合计 600/1,200 帧。前处理包含 resize 和 F32 归一化，Runtime 包含 UCP、cache 和输出拷贝，后处理为 L2；保存同帧四阶段数据及共同墙钟吞吐。JPEG 解码、检测和检索不计时。run.sh 支持 TARGET、VARIANT、MODEL_PATH、FIXTURE_DIR、BENCHMARK、STREAMS、WARMUP、RUNS、ROUNDS。

测量前、中、后保存板型、OS/kernel、Runtime/HBRT、CPU 在线核数、governor/频率、BPU 频率和温度，不修改时钟策略。S100P 根分区空间有限，数据使用已有 `/mnt/ssd`。连接路径为本机 Mac → Tailscale dev-s600 → S100/S100P；S600 直接使用 dev-s600。

执行前在板端 campaign 中冻结源码归档和控制脚本 SHA。`evaluator/stage_s.py --workspace WB --build-id BUILD_ID` 检查全量精度、模型/数据/源码身份、原生烟测、6 轮 Runtime、两种 C++ 流水线、原始统计和环境证据，保留原始 OE HTML、节点 CSV 和编译 JSON。候选包包含模型、release.json、报告和 SHA256SUMS，路径为 `releases/models/osnet/osnet/reid/x1.0/TARGET`。技术验收通过后仍不自动批准发布。

[英文流程及官方参考文档](S_SERIES.md)。本次查阅的官方网页正文和板端工具帮助随私有 Workbench 证据归档。

固定输入 ONNX 阶段比对遇到本 OE 镜像 GPU PTQ 推理的 cuDNN 执行失败/非法内存访问。改用已安装的 `hmct.executor.ORTExecutor(model).to("cpu").create_session().forward({"images": input})`，与 PyTorch 分进程运行并复制返回特征；CPU FP32/校准/PTQ 检查及官方 HBRuntime BC 仿真均成功，模型和输入字节没有改变。PTQ 校准本身仍在 GPU 上以 batch=8 执行。只有 ONNX 模型 SHA 和输入 SHA 完全相同才能跨目标复用阶段回执；各目标的 BC 独立仿真，再与各板 SDK 的原始特征比较。

冻结源码的导出复现为 `osnet-s-20261030-export-r02`，ONNX 与编译输入逐字节相同；冻结的全量 FP32 基线为 `osnet-s-float-20261030-r02`，特征与此前探索性基线逐字节相同。
