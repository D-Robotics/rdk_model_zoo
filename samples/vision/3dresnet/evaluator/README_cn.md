[English](README.md) | 简体中文

# R3D-18 评测记录

<a id="dataset"></a>
## 数据集

随附功能输入是预处理后的 `test_data/video0.npy`，shape `(1,3,16,112,112)`、dtype float32，内容为 16 帧射箭样例。`test_data/kinetics_classnames.json` 是 CLI 使用的 400 条 Kinetics name-to-id 映射；读取后转为 id-to-name 标签（名称中内嵌的引号字符会被去除）。

本目录没有完整 Kinetics-400 数据集、视频解码器、抽帧代码或数据集下载命令。该片段原始获取和预处理命令未记录。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

- 板端功能检查：RDK S100、匹配的 `hbm_runtime`、准备好的 `model/s100/r3d_18.hbm` 和可识别的 S100 身份。
- 下面的性能表为发布记录，由 `hrt_model_exec` 测得；其完整命令、镜像、runtime 版本与 raw 输出未随附，复测时请记录这些条件。

<a id="command"></a>
## 评测命令



S100 功能 smoke 命令：

```bash
# cwd：仓库根目录；前置：显式下载模型和 S100 板卡
bash samples/vision/3dresnet/model/download.sh s100
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm \
  --model-path samples/vision/3dresnet/model/s100/r3d_18.hbm \
  --test-clip samples/vision/3dresnet/test_data/video0.npy \
  --label-file samples/vision/3dresnet/test_data/kinetics_classnames.json \
  --top-k 5 --priority 0 --bpu-cores 0
# 预期：退出码 0 和包含 5 条 predictions 的 JSON
```

性能记录由 `hrt_model_exec` 生成；记录中未包含带输入/输出文件参数的完整命令，因此不提供复现命令。

<a id="metrics"></a>
## 指标

- **Top-1：** 数值稳定 softmax 后按概率降序排列的第一个 class ID。
- **Top-K：** 前 `K` 个 class ID 和 float32 概率，`K` 由 `--top-k` 控制，默认 5。
- **功能标签检查：** `video0.npy` 的参考 Top-1 类别为 `archery`（source 记录）。
- **性能：** thread-performance 记录（S100、`hrt_model_exec`）。“Total Latency”和“Average Latency”单位是毫秒，FPS 是吞吐率。

| 线程数 | 帧数 | 总耗时 (ms) | 平均耗时 (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 18267.76 | 182.68 | 5.47 |
| 2 | 100 | 18291.76 | 182.93 | 10.82 |
| 4 | 100 | 18501.06 | 185.03 | 21.07 |
| 8 | 100 | 24743.56 | 249.19 | 30.74 |

记录中还注明 BPU 占用约 5.2%、ION 内存约 91.9 MB、读带宽 533、写带宽 304。单位和测试环境记录不完整。

<a id="outputs"></a>
## 输出

功能命令将 JSON 报告写到 stdout，不创建结果文件。每条 prediction 包含 `class_id`、`score` 和 `label`；CLI 不保存 raw model output。以下截图展示射箭帧和 Top-5 结果：

![Archery frame](../test_data/readme_img/image-4.png)
![Top-5 result](../test_data/readme_img/image-5.png)

<a id="reference-results"></a>
## 参考结果

| 参考项 | 来源 |
| --- | --- |
| `video0.npy` Top-1 `archery` | source evaluator README 和截图 |
| 四行 thread-performance 表（上表） | source evaluator README |
| BPU/ION/带宽备注（上文） | source evaluator README 和截图 |

<a id="boundaries"></a>
## 适用范围

- 本 sample 没有完整数据集 evaluator 实现；评测即上面的单片段功能命令。
- 没有完整 source `hrt_model_exec` 命令，因此仅凭仓库内容无法复现该性能记录。

同一记录的附加性能指标截图：

![Additional metrics](../test_data/readme_img/image-6.png)
