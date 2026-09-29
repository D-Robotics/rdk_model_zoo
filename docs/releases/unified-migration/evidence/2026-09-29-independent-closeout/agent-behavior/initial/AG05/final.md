# 结论

**不能只替换模型路径。** 输入侧碰巧对得上，输出侧在代码里有硬性协议校验，自有模型（`(1,1024,50)` int8）会在 `bind_model` 处被拒绝；此外"50 类"牵动 sample 内多处写死的四类假设。以下是按 Skill 流程读取的实际代码与调用链。

# 实际调用链（当前统一入口 = `samples/vision/pointnet`）

```
run.sh → python3 -m samples.vision.pointnet.runtime.python.main
  main.main()                                     runtime/python/main.py:33
  ├─ resolve_selection(...)                       model_binding.py:67  # 只接受 target=s100；--model-path 必须同时给 --asset-id s:pointnet:s100/pointnet.hbm
  ├─ np.loadtxt(--test-pts) → (N,3) float32       main.py:54
  ├─ RuntimeModelRunner(selection).load()         single_array_runner.py:68
  │   ├─ require_execution_target(target)         # 板卡身份门
  │   ├─ verify_asset_file(asset, model_path)     assets.py:98         # 清单 sha256=null 时仅记录观测哈希，不拒绝外来文件
  │   ├─ hbm_runtime.HB_HBMRuntime(model_path)    single_array_runner.py:179  # 板端 SDK，懒加载
  │   └─ bind_model(selection, metadata)          model_binding.py:93  # ★ 协议硬校验点
  ├─ task.pre_process(points)                     pointnet.py:31       # 要求 (N,3)、N==metadata 输入维[2]；质心/最大半径归一化；不重采样
  ├─ task.forward(tensors)                        pointnet.py:54       # runner 校验 shape/dtype 后原样返回 raw (1,N,4)
  └─ task.post_process(raw)                       pointnet.py:58       # int8 需 metadata 的 SCALE 描述符，float64 反量化后 argmax → int32 (N,)
```

另有 `platforms/s/samples/vision/pointnet/runtime/python/`（仅 `main.py`+`pointnet.py`）为历史镜像，当前入口以上面的 `samples/` 树为准（README 与 `run.sh` 均指向它）。

# I/O 对照（按 integration-contract 逐项）

| 项 | 自有模型（fixture.json） | sample 期望（代码实证） | 判定 |
|---|---|---|---|
| 输入 shape | `(1,3,1024)` float32 | 固定 `(1,3,N)`、float32，`N` 来自编译 metadata（model_binding.py:117-120） | **布局兼容**；但 N=1024 与自带 `chair.pts` 的 **2776 点**不符，`pre_process` 不采样/补点，需另备 1024 点输入 |
| 输入名 | 未知 | 恰好 1 个输入（model_binding.py:113） | 缺信息 |
| 输出 shape | `(1,1024,50)` int8 | **必须恰为 `(1,N,4)`**（model_binding.py:121-122；tests/test_pointnet.py:21,155 同样钉死） | **硬阻断**：50 通道触发 `MetadataMismatchError`，`main.py` 捕获后 rc=2 |
| 输出 dtype | int8 | 允许 int8..int32，但**必须带有效 SCALE 描述符**（model_binding.py:124-127 → quantization.py:157，scale 有限正、axis 与通道匹配） | fixture `quantization: null` → 缺 scale/zero_point/axis，阻断 |
| 输出语义 | 未知 | 逐点 logits，argmax 即标签，0..3=back/seat/leg/arm（pointnet.py:84、main.py:69、visualization.py:10） | 50 通道的类别集合/顺序/是否 logit 未知 |
| 预处理 | 未知 | 减质心、除最大欧氏半径（pointnet.py:44-49），点序不变 | 自有模型训练时的归一化约定未提供 |

# 代码内必须改动的位置（若类别数确为 50）

`model_binding.py:121`（shape 校验）、`main.py:69`（四类计数）、`visualization.py:10,76`（CHAIR_PARTS）、`runtime/python/README*.md` 与 `README_cn.md` 支持矩阵/标签表、`tests/test_pointnet.py` 契约——这已是一整套 wrapper/接口适配（本 Skill 范围内可做），而非"只换路径"。

# 还缺的信息

1. **模型产物本体**：文件路径/格式（是否已编译为目标平台 HBM）、哈希、上游训练 commit/框架、权重许可——fixture 全部未给。
2. **目标平台矛盾**：fixture `board_records.target=x5`（且 scope "single image"，对点云任务不适用），但本 sample 仅发布 s100，`resolve_selection` 直接拒绝非 s100（model_binding.py:78-79）。需确认自有模型到底为哪个 SoC/哪个 OE 版本编译。
3. **产物真实 metadata**：输入/输出张量名、输出是否恰一维 50 通道、int8 的 scale/zero_point/axis（per-tensor 或 per-channel）。
4. **输出语义**：50 类的类别名与索引顺序、logits 还是概率、通道维位置。
5. **训练前处理一致性**：归一化方式、坐标系/轴序、是否要求固定采 1024 点、点序语义。
6. **转换与验证状态**：fixture 的 conversion 收据产生于 `before_preprocess_change`，当前代码已是 `after_preprocess_and_postprocess_change`——按 evidence-contract，旧收据不能替代修改后的端到端验证；且 `conversion/README_cn.md` 自述缺编译配方。`latest_run.rc=2、new_outputs=[]` 与"binding 拒绝→rc=2"的失败特征一致；`existing_result`（2020-01-01）是陈旧产物，不能作为本次运行证据。
7. **精度协议**：浮点基线输出、数据集、逐点精度指标与容差——均未提供。

# 对 fixture 中 proposed_change 的判断（不建议采纳）

- "load JPEG, run model, NMS and save result.jpg in forward"：本任务是点云 `.pts` 输入、逐点 argmax，无 NMS；且 sample 设计明确"文件 IO 与绘图在推理阶段之外"（main.py:3、pointnet.py:26）。
- "readme: infer all targets work from one X5 run"：一次 X5 记录不能外推其他 target（evidence-contract：分 target 记录；该 sample 仅 s100，x5/s100p/s600 为"无已发布制品"）。

# 操作与限制说明

- 实际读取路径：SKILL.md 及其 4 个 references/assets；fixture.json；`rdk-b7-board-integration` 下 `samples/vision/pointnet/`（main.py、pointnet.py、model_binding.py、model_runner.py、run.sh、visualization.py、tests/test_pointnet.py、README_cn.md、runtime/python/README_cn.md、conversion/README_cn.md、model/download.py）、`samples/_shared/`（single_array_runner.py、runtime_meta.py、quantization.py、assets.py）、`docs/release/s/models.yaml`、`test_data/chair.pts`（行数=2776）。
- 未运行任何转换、推理、下载或 shell 命令；全程只读，未修改任何文件。**工具链产物状态：无可核对的新产物（收据过期）；样例接入状态：被输出协议硬阻断，缺上文 1–7 项信息前无法推进。**
