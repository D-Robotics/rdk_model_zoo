# 转换(YOLO26 Depth,RDK-S)

用 RDK-S OpenExplorer 工具链(`hb_compile`)将 YOLO26 Depth 五个规格量化为 `.hbm`。在 OE Docker 镜像内、本 `conversion/` 目录下执行。

## 混合发布 Profile

发布默认采用**按规格选型的两个编译 Profile**：

| 规格 | Profile | ONNX 边界 | 输入契约 | 校准 |
| --- | --- | --- | --- | --- |
| `n`/`s`/`m` | **NV12** | 校准 log-depth(图内 `clip → scale/bias → exp → resize4x`) | letterbox 后 NV12,`data_scale=1/255` | `max`,分位 `0.9999` |
| `l`/`x` | **lite** | 192×192 原始深度 logit | scale-fill float32 NCHW featuremap `/255` | `default`(KL) |

之所以拆分：NV12 编译的 `l`/`x` 输出被定标上界截断(quant max 钉在校准 max 上)，lite 编译的 `n`/`s`/`m` 达不到 0.999 raw-cosine 线。**不要把两个 Profile 的校准资产混用**——NV12 的 letterbox RGB 张量与 lite 的 scale-fill featuremap 张量是不同输入契约，不能互换判断精度。

## 前置

- OE Docker 镜像:`ai_toolchain_ubuntu_22_s100_s600_gpu`(提供 `hb_compile`)。
- 各规格 Ultralytics YOLO26 权重(`yolo26{n,s,m,l,x}-depth-log.pt`)。
- SUNRGBD 校准图像(用 `extract_sunrgbd_subset.py` 备子集)。

## 步骤

### 1. 导出各规格 ONNX

```bash
# lite 边界(l/x):raw logit 输出
python3 export.py --weights /path/to/yolo26l-depth.pt --variant l --output-dir ./onnx
# -> ./onnx/yolo26l-depth_op11_lite.onnx

# NV12 边界(n/s/m):校准 log-depth 输出
python3 export.py --weights /path/to/yolo26n-depth.pt --variant n --output-dir ./onnx
# -> ./onnx/yolo26n-depth-log.onnx
```

### 2. 备校准数据

```bash
python3 extract_sunrgbd_subset.py --src /path/to/sunrgbd --out ./sunrgbd_subset

# lite 档位(l/x):scale-fill,/255
python3 prepare_calibration.py --images ./sunrgbd_subset --contract lite \
  --output ./calibration_lite --manifest ./calibration_manifest_lite.json \
  --report ./calibration_report_lite.md

# NV12 档位(n/s/m):114-letterbox,/255
python3 prepare_calibration.py --images ./sunrgbd_subset --contract nv12 \
  --output ./calibration_nv12 --manifest ./calibration_manifest_nv12.json \
  --report ./calibration_report_nv12.md
```

`--contract` 选择校准预处理；两份输出必须放不同目录——两契约不可互换。

### 3. 用 hb_compile 量化

`ptq_yamls/` 下 24 份 committed YAML(9 份 NV12 对应 n/s/m + 15 份 lite；lite 的 n/s/m YAML 保留作实验用)。

```bash
python3 scripts/quantize.py --variant n --march nash-e   # 自动选 NV12 profile
python3 scripts/quantize.py --variant l                  # lite profile
python3 scripts/quantize.py                               # 发布全集:5 规格 × 3 march
```

### 4. 拷贝 .hbm 到模型目录

```bash
cp bpu_model_output_yolo26n_nv12_nashe/yolo26n_depth_nashe_768x768_nv12.hbm ../model/nash-e/
cp bpu_model_output_yolo26l_lite_nashe/yolo26l_depth_lite_nashe_768x768.hbm ../model/nash-e/
```

产物文件名即 `runtime/python/main.py` 期望名。

## 说明

- int8 量化(CNN 友好)。
- NV12 档位的 `/255` 由图内 `data_scale` 完成；lite 档位的 `/255` 在校准生成脚本和板端运行时执行。
- 各 config 已提交(路径相对本 `conversion/` 目录),无需运行时生成 YAML 即可复现。

## 精度调优（可选）

发布默认保持混合 Profile。精度设置只能作为受控实验，不能视为可互换开关：

- NV12 编译的 `l`/`x` **不可交付**：深度输出被校准 max 截断(bus.jpg 上约 17% 饱和像素，三板 cosine 0.9938/0.9944)。修复需要校准集覆盖更高 logit，改 YAML 无效——`max + 0.9999` 当时已启用。
- lite 编译的 `n`/`s`/`m` 低于 0.999 raw-cosine 线(S100 bus.jpg 上 0.9903/0.9854/0.9529)。`max + 0.9999` 校准把 `n` 提到 0.9975,仍未过线。
- 在 nash-e/S100 上，为 `/model.23/head/head.3/Conv` 设置
  `node_info ... OutputType: int16` 后，`bus.jpg` 的 lite raw cosine 与 int8
  基线相同，均为 0.985449。
- 在 nash-e/S100 上设置
  `calibration_parameters.optimization: set_all_nodes_int16` 后，HBM 从约
  13 MB 增至 25 MB、实测推理从约 5.8 ms 增至 23.0 ms，raw cosine 反而降至
  0.982804，因此**不作为本模型默认方案**。

若未来模型或数据集未满足验收指标，应固定 ONNX、校准图片、预处理、板卡和测试
输入；每次只改一个量化选项，同时对比 raw-logit cosine 与后处理深度图。只有记录
好所选 profile 和板端证据后，才替换已发布 HBM。
