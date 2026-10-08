# YOLOE 模型制品

<a id="artifacts"></a>
## 制品

下表列出 14 个原始发布制品。S 行为已发布的量化输出模型；在 S 上进行浮点输出推理，请按[转换说明](../conversion/README_cn.md)准备本地浮点模型（当前没有发布浮点 S HBM）。

| Asset ID | Target | Variant | Output route |
| --- | --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | x5 | 11s | floating |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | x5 | 11m | floating |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | x5 | 11l | floating |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 11s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26n | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26m | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26l | quantized / float conversion pending |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | s100 | 26x | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26n | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26s | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26m | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26l | quantized / float conversion pending |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | s100p | 26x | quantized / float conversion pending |

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── download.py  # 准备模型文件
├── download.sh  # 模型准备命令
└── vocabulary.py  # Python 脚本
```

<a id="preparation"></a>
## 准备

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
```

下载失败或哈希不符会非零退出；已存在且哈希不符的文件不覆盖。离线复制原制品后使用精确 `--asset-id` 与 `--model-path`。S 原制品可显式下载，用于量化输出路线与来源对照；浮点入口请使用自行转换的浮点制品。

```bash
# cwd: repository root; original quantized publication only
bash samples/vision/yoloe/model/download.sh --target s100p --variant 26n
```

<a id="accompanying-files"></a>
## 配套文件

必需词表 [classes.names](../test_data/classes.names)，4585 行顺序与模型类别 ID 对齐；CLI 拒绝哈希变化。示例图 [office_desk.jpg](../test_data/office_desk.jpg) 可以替换，但不能更改词表来获得新类别。

<a id="local-paths"></a>
## 本地路径

默认路径为 `model/<target>/<manifest filename>`，S26 保留 `nash-e/` 或 `nash-m/` 子目录；`--output-dir` 则在指定目录后直接拼接 manifest filename。`--model-path=null` 按选择推导。X5/S100 默认 11s，S100P 默认 26n；显式 asset-id 在未指定 variant 时优先。

自行转换浮点 HBM 时必须同时指定 `--model-path` 和 `--local-float-sha256`：摘要用于精确识别本地字节并完成选择，`source_asset_id` 记录该转换的协议来源。加载时核验目标平台和全部十个 NHWC float32 输出角色，目标兼容性由这些加载检查与板端运行共同确认。

ONNX 检查、平台校准与可选编译见[转换准备说明](../conversion/README_cn.md)。生成文件名中的 `_float` 表示目标协议；`compiled_unverified` 表示编译产物还需按验证步骤完成元数据与数值检查。

<a id="formats-checksums"></a>
## 格式与校验和

X5 为 BIN，S 为 HBM。已知值取自活跃发布清单；S11 发布哈希未知，相邻制品摘要不互用。

| Asset ID | SHA-256 | Authority |
| --- | --- | --- |
| `x5:yoloe:yoloe_11s_seg_pf_bayese_640x640_nv12.bin` | `8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11m_seg_pf_bayese_640x640_nv12.bin` | `ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31` | `docs/release/x5/models.yaml` |
| `x5:yoloe:yoloe_11l_seg_pf_bayese_640x640_nv12.bin` | `a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2` | `docs/release/x5/models.yaml` |
| `s:yoloe11_seg:yoloe_11s_seg_pf_nashe_640x640_nv12.hbm` | `null (unknown)` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm` | `bf4a03cb75fcfe67cdf6549ef0dec9541b5ae009cdef05e7879a4af9c75a5439` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm` | `a27c313dc922b778d3537187c7b55a82f87b826edc66abdef027e15770243ee6` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm` | `d257d502a5245de823cdb76cc734974a1568e68bff02bece232895db29aceeef` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm` | `695cd10aba09873bb9cd4f838113882945e979c745514185c7ab6f5cb17690c4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm` | `119a6b069b32282915948e161f548b7b22136b765b70101c33d22680ac6fef24` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26n_seg_pf_nashm_640x640_nv12.hbm` | `58922c669bcccec7a00c11719db6c3b9c4eaa4e0388b58bd120fda914fe9d048` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26s_seg_pf_nashm_640x640_nv12.hbm` | `73f838c5d52a73451b8af50002c020d571d8fd75beeb29c54e0a6c956e0c45a5` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26m_seg_pf_nashm_640x640_nv12.hbm` | `8621baa458a94d3d1946665609633f1ecdd930efeeebe91d975f822ffd273af4` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26l_seg_pf_nashm_640x640_nv12.hbm` | `314df1d0d60c839f9ff4248ab0d4ff2adc3cc8e514a8e9cf9acca62527696738` | `docs/release/s/models.yaml` |
| `s:yoloe26_seg:nash-m/yoloe_26x_seg_pf_nashm_640x640_nv12.hbm` | `2abf9d30dd487f2d5eb53444b7272a71d7be18265a39bd463cfd5b0cd6c14539` | `docs/release/s/models.yaml` |
