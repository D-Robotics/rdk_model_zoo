[English](README.md) | [简体中文](README_cn.md)

# 模型准备

<a id="artifacts"></a>
## 已发布制品

按目标和变体选择 20 条独立清单记录，默认 n。BIN 属于 X5，HBM 必须匹配对应 Nash march。
S100P 有独立 nash-m 制品，不会隐式复用 nash-e。

| 目标 | 变体 | 方案 | 相对本 model 目录的路径 |
|---|---|---|---|
| x5 | n | nv12 | `yolo26n_depth_bayese_768x768_nv12.bin` |
| x5 | s | nv12 | `yolo26s_depth_bayese_768x768_nv12.bin` |
| x5 | m | nv12 | `yolo26m_depth_bayese_768x768_nv12.bin` |
| x5 | l | nv12 | `yolo26l_depth_bayese_768x768_nv12.bin` |
| x5 | x | nv12 | `yolo26x_depth_bayese_768x768_nv12.bin` |
| s100 | n | nv12 | `nash-e/yolo26n_depth_nashe_768x768_nv12.hbm` |
| s100 | s | nv12 | `nash-e/yolo26s_depth_nashe_768x768_nv12.hbm` |
| s100 | m | nv12 | `nash-e/yolo26m_depth_nashe_768x768_nv12.hbm` |
| s100 | l | lite | `nash-e/yolo26l_depth_lite_nashe_768x768.hbm` |
| s100 | x | lite | `nash-e/yolo26x_depth_lite_nashe_768x768.hbm` |
| s100p | n | nv12 | `nash-m/yolo26n_depth_nashm_768x768_nv12.hbm` |
| s100p | s | nv12 | `nash-m/yolo26s_depth_nashm_768x768_nv12.hbm` |
| s100p | m | nv12 | `nash-m/yolo26m_depth_nashm_768x768_nv12.hbm` |
| s100p | l | lite | `nash-m/yolo26l_depth_lite_nashm_768x768.hbm` |
| s100p | x | lite | `nash-m/yolo26x_depth_lite_nashm_768x768.hbm` |
| s600 | n | nv12 | `nash-p/yolo26n_depth_nashp_768x768_nv12.hbm` |
| s600 | s | nv12 | `nash-p/yolo26s_depth_nashp_768x768_nv12.hbm` |
| s600 | m | nv12 | `nash-p/yolo26m_depth_nashp_768x768_nv12.hbm` |
| s600 | l | lite | `nash-p/yolo26l_depth_lite_nashp_768x768.hbm` |
| s600 | x | lite | `nash-p/yolo26x_depth_lite_nashp_768x768.hbm` |

文件名、URL 和摘要以 [X5 清单](../../../../platforms/x5/docs/release/models.yaml) 与
[S 清单](../../../../platforms/s/docs/release/models.yaml)为准。
下方 list 命令打印精确 ID 和 URL。源清单声明可下载，不代表本轮已经下载或验证推理。

<a id="preparation"></a>
## 显式准备

从仓库根目录执行：

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --list-models
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/model/download.sh --target s100p --variant l
```

每次下载一个制品。其他条目使用对应目标、变体或精确 `--asset-id`。
`download_model.sh` 委托同一个统一下载器，参数也是显式旗标，不沿用旧 shell 位置参数。
下载与推理分开，本轮主机迁移未实际下载模型。

<a id="accompanying-files"></a>
## 附属文件

深度任务不需要标签文件。样例只附带逐字节保留的 `../test_data/bus.jpg`，不附带校准集或评估集。
S lite l/x 校准系数位于 binding 模块，是声明权重契约的一部分。
原始权重、生成 ONNX、BIN/HBM 和 SUNRGBD 数据保存在外部，准备步骤及缺口见[转换](../conversion/README_cn.md)。

<a id="local-paths"></a>
## 本地路径与外部副本

默认路径从本 model 目录解析，不受 Python 当前目录影响。X5 文件平放，S 保留 nash-e/m/p 子目录。
下载器 `--output-dir` 只改变下载位置，不修改运行时默认路径。例如：

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n \
  --output-dir /work/models
python -m samples.vision.yolo26_depth.runtime.python.main --target x5 \
  --asset-id x5:yolo26_depth:yolo26n_depth_bayese_768x768_nv12.bin \
  --model-path /work/models/yolo26n_depth_bayese_768x768_nv12.bin \
  --output /work/depth/external-published
```

新编译模型显式声明自转换来源：

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --target x5 \
  --asset-id x5:yolo26_depth:yolo26n_depth_bayese_768x768_nv12.bin \
  --model-path /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin \
  --converted-model --output /work/depth/custom-x5-n
```

此时 ID 参照输入/输出契约，不表示官方字节身份。报告中 `asset_id=null`、
`artifact_origin=user-converted`，保留 `contract_reference` 和实际模型摘要。
板卡身份及 metadata 检查仍生效。任意形状或边界变化需要新绑定，改名不会使不兼容模型兼容。
S lite 的运行时系数必须与导出权重校准参数一致。

<a id="formats-checksums"></a>
## 格式、摘要与来源

五个 X5 发布方摘要保持不变：

| 变体 | 发布方 SHA-256 |
|---|---|
| n | `e55091eb594e20e37e6c36a36cce42a94ad80ec651ae893a2143cd2273ed9b0b` |
| s | `0e43958195f504d7a8ac48b1c99f4802cd9a4c3580321bfb251d0e0f892ccf4c` |
| m | `f4f2f1958dc16324932b4492490209c817cf7565c3c29240bcf4f0012f9c0be0` |
| l | `6a5fa40bda20ee56208ca6e594ecfd9781329385d0baf1b15c9eaa9625286d14` |
| x | `61798227fb7e0772a739b483ae5b5acd58a8e785dd7fd9aec5dcac7db0903d91` |

十五个 S 摘要全部为 `null`（未知）。下载器和运行时记录实际摘要，但在没有可信预期值时，
它只能标识本地字节，不能独立认证发布方来源。原生启动器在构建或运行前校验 X5 发布摘要。
显式自转换模式不会将参照制品的发布摘要当成新生成文件的摘要。

摘要不证明精度或板卡兼容性。运行时加载时按声明方案校验 metadata；本轮主机迁移尚未观察
真实制品 metadata，也没有新增板测结果。
