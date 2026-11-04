[English](README.md) | 简体中文

# LPRNet

<a id="overview"></a>
## 算法与来源

LPRNet 无需独立字符检测器即可识别车牌文字。Python 入口在所选目标上运行准备好的车牌张量，并使用 CTC 解码字符序列。

参考：[LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447).

<a id="directory"></a>
## 目录结构

```text
lprnet/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持的模型

| 目标 | 变体 | Python | C++ | 说明 |
|---|---|---|---|---|
| X5 | `lpr.bin` | 支持 | 不支持 | 用内置 `test_input.dat` 验证 |
| S100/S100P/S600 | `sirius-68class` | 已接入，板端待测 | 不支持 | 2026-11-04 独立转换 HBM |

Runtime 语言：Python。板端运行使用内置 `test_input.dat`，输出原生 `(1,68,18,1)` logits。

<a id="prerequisites"></a>
## 环境前提

板端运行需要 RDK OS `>=3.5.0`、匹配的 `hbm_runtime`、发布的 X5 `lpr.bin` 和带 NumPy 的 Python。随源提供的 `test_input.dat` 已经是 float32 张量，LPR 任务不需要图像包。active 清单中模型校验值为 `sha256: null (unknown)`。

<a id="quickstart"></a>
## 快速体验

在仓库根目录显式准备制品，然后在可识别的 X5 板端运行：

```bash
# cwd：仓库根目录
python3 -m samples.vision.lprnet.model.download \
  --target x5 --output-dir samples/vision/lprnet/model
python3 -m samples.vision.lprnet.runtime.python.main --target x5
```

第一条命令将 `model/lpr.bin` 写入本地并报告观测摘要；发布者摘要未知。第二条命令读取 `test_data/test_input.dat`、执行模型并打印含 `plate` 的 JSON。运行时和 `run.sh` 都不会调用下载器。只做主机选择检查时运行 `python3 -m samples.vision.lprnet.runtime.python.main --dry-run --target x5`。

<a id="expected-results"></a>
## 预期结果

成功推理退出码为 `0`，打印含 `target`、完整 `asset_id` 和解码后 `plate` 的 JSON。具体车牌由模型和输入决定，结果取决于模型和输入；`test_data/example.jpg` 只是源提供的可视参考，实际运行时输入是 `test_input.dat`。

CLI 读取预先打包的 float32 `test_input.dat`，重塑为 `(1,3,24,94)`。生成张量前完成车牌图片解码、缩放和归一化。

<a id="entry-points"></a>
## 入口索引

- [`model/README_cn.md`](./model/README_cn.md)：清单制品、下载、路径和校验值。
- [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)：CLI 与 `LPRNetRecognizer` API。
- [`conversion/README_cn.md`](./conversion/README_cn.md)：源 OE 命令和不可复现项。
- [`evaluator/README_cn.md`](./evaluator/README_cn.md)：完整 raw/text 对照步骤。


## 新转换的 S 系列模型（2026-11-04）

S100、S100P、S600 分别使用 nash-e、nash-m、nash-p 独立编译。源为 [sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch/tree/7c976664b3f3879efabeaff59c7a117e49d5f29e)，Apache-2.0；精确 SHA-256、配置和来源见 [转换登记](conversion/converted_targets_20261104.json)。

先用 `python3 -m samples.vision.lprnet.runtime.python.main --target s100 --list-models` 查看完整 asset-id，再显式下载和运行。新模型使用车牌裁剪图，BGR 线性缩放至 94×24，逐元素 `(pixel-127.5)/128`，float32 NCHW。历史 X5 输入不代表这个独立权重版本的预处理。

```bash
python3 -m samples.vision.lprnet.model.download --target s100
python3 -m samples.vision.lprnet.model.prepare_input --image /path/to/plate.jpg --output /path/to/plate.dat
python3 -m samples.vision.lprnet.runtime.python.main --target s100 --test-bin /path/to/plate.dat
```

下载默认保留 `model/s100/` 平台子目录；外部 `--model-path` 必须同时提供该目标的精确 `--asset-id`。S100P/S600 将目标参数相应替换。浮点全输出检查、编译、真实 HBM ABI 和完整 OE 报告已验收；板端 smoke、识别准确率、Runtime 1/2 并发和 C++ 1/2 流均为 `not-run`。下方历史性能仅描述旧 X5 文件。

<a id="historical-performance"></a>
## 参考性能

源 benchmark 记录：

| 模型 | 测试帧数 | FPS | 平均延迟 | BPU 使用率 | ION 内存 |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="license"></a>
## 许可

源示例和仓库代码遵循仓库 Apache-2.0 许可。LPRNet 论文及上游工程仍归其作者所有；模型来源和未知发布校验值记录在 model README 中。
