[English](README.md) | 简体中文

# LiteAvatar 模型产物

[Python 运行时](../runtime/python/README_cn.md) · [转换](../conversion/README_cn.md)

LiteAvatar 在 **S100** 上运行，需要五文件包：四个本地准备的 HBM 产物加上
已打包的 CMVN 文件。[S 清单](../../../../docs/release/s/models.yaml)记录了
发布摘要；准备走下面两条路线之一。S100P、S600 与 X5 没有发布集合。

<a id="artifacts"></a>
## 产物

| `model/` 下文件 | 目标 | 作用 | 来源 |
| --- | --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | s100 | 音频编码器（paraformer，50 层隐状态） | 本地包或自行编译（[转换](../conversion/README_cn.md)） |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | s100 | 隐状态到嘴部参数 | 本地包或自行编译（[转换](../conversion/README_cn.md)） |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | s100 | 参考帧人脸编码器 | 本地包或自行编译（[转换](../conversion/README_cn.md)） |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | s100 | 人脸解码器（INT16 BPU + FP32 CPU InstanceNorm） | 本地包或自行编译（[转换](../conversion/README_cn.md)） |
| `am.mvn` | s100 | 前端 CMVN 统计 | 已打包在本目录 |

只需准备四个 HBM 文件（`am.mvn` 已随仓库提交）；四个模型在运行时边界都是
FLOAT32 输入/输出，INT16 量化在产物内部。

<a id="directory"></a>
## 目录结构

```text
model/
├── README.md     # 英文说明
├── README_cn.md  # 中文说明
├── verify.py     # 已准备包的金标摘要检查
├── am.mvn        # 打包的 CMVN 统计（固定摘要）
└── s100/         # 准备好的 HBM 产物
```

<a id="preparation"></a>
## 准备

两条路线最终都得到 `model/s100/` 下的四个文件；用下面的命令校验：

```bash
# 输入：model/s100/ 下的四个 HBM 文件
# 成功：退出码 0 并逐项打印摘要匹配
python3 samples/vision/lite_avatar/model/verify.py
```

`verify.py` 逐文件对照[格式与校验和](#formats-checksums)中的清单摘要校验
SHA-256，任何缺失或不匹配都会以非零码退出。

**路线 A —— 本地包。** 把四个已编译产物的副本以发布文件名放到
`model/s100/` 下，然后运行 `verify.py`。

**路线 B —— 自行编译。** 按[转换](../conversion/README_cn.md)全程用
OpenExplorer 工具链自行编译，再把四个 HBM 以发布文件名拷入 `model/s100/`。
自编译集合与金标摘要不同是正常的；请用本地哈希清单显式标识，保证校验仍然
精确：

```bash
cd samples/vision/lite_avatar/model
python3 - <<'PY' > lite-avatar-rebuild.json
import hashlib, json, pathlib
print(json.dumps({
    f"s100/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(pathlib.Path("s100").glob("*.hbm"))
}, indent=2))
PY
bash ../runtime/python/run.sh --target s100 \
    --models-manifest lite-avatar-rebuild.json \
    --audio ../test_data/asr_example.wav \
    --output outputs/lite_avatar/avatar.mp4
```

清单是 JSON 对象，键为发布文件名（可以是其子集），值为 64 位十六进制
SHA-256。每个列出的文件都按记录的摘要校验，SDK 加载时还会验证模型的
输入/输出形状与目标。

<a id="accompanying-files"></a>
## 伴随文件

`am.mvn` 是 Paraformer 前端的 CMVN 统计文件，已提交在本目录（与
Paraformer 示例的副本同字节），由运行时 `--cmvn-path` 默认值选用——无需
准备。数字人数据（背景视频、人脸框、参考帧）是独立的，按
[测试数据](../test_data/README_cn.md)准备。

<a id="local-paths"></a>
## 本地路径

准备完成后，四个产物位于示例根目录下的 `model/s100/`；自编译集合还会在
`model/lite-avatar-rebuild.json` 携带自己的哈希清单。[准备](#preparation)中的
`--models-manifest` 示例与
[Python 运行时](../runtime/python/README_cn.md)中的路径都指向这些位置。

<a id="formats-checksums"></a>
## 格式与校验和

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `s100/lite_avatar_encoder_400x560_s100.hbm` | nash-e `.hbm`，F32 feats `[1,400,560]` 输入 / F32 隐状态 `[1,400,50,512]` 输出 | `0a6833a1e24d9f407e4a7662c305c6b292f2d947aef03adce2043bec1091aef0` |
| `s100/lite_avatar_audio2mouth_30x50x512_s100.hbm` | nash-e `.hbm`，F32 `[1,30,50,512]` 输入 / F32 嘴部参数 `[1,30,32]` + feat 输出 | `ddded3350ef1e9c8e068263d841b3a15a1a5f59b027c7d50f6dd3c8985fa50e5` |
| `s100/lite_avatar_net_encode_384x384_s100.hbm` | nash-e `.hbm`，F32 RGB `[1,3,384,384]` 输入 / F32 特征金字塔 `f0..f3` 输出 | `b244f61b63bbfdfb219bf03270cb19b9a7a0d8e1604dbbc6fb0591956b673ecb` |
| `s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | nash-e `.hbm`，F32 特征金字塔 + 嘴部参数输入 / F32 嘴部图像 `[1,3,384,384]` 输出（21 个 InstanceNorm 节点在 CPU） | `27e46cf9d6514f587d54c16cfa813583d89e59c9d5b5d4138efdf51e4d20df3c` |
| `am.mvn` | Kaldi CMVN 统计（文本） | `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5` |

每个 HBM 都是固定上游 TorchScript 导出的训练后量化（INT16 激活、INT8
权重）产物，按 nash-e 编译。[S 清单](../../../../docs/release/s/models.yaml)
中的 `lite_avatar` 行携带这些摘要，是发布权威。
