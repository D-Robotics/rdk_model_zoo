[English](README.md) | 简体中文

# PointNet 模型文件

<a id="artifacts"></a>
## 制品清单

| Asset ID | Target | 文件 | 来源 |
| --- | --- | --- | --- |
| `s:pointnet:s100/pointnet.hbm` | s100 | `s100/pointnet.hbm` | 公开下载 |

[发布清单](../../../../docs/release/s/models.yaml)提供
[HBM 下载地址](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointNet/pointnet.hbm)。
没有发布本例的 X5、S100P 或 S600 制品。该模型只分割椅子部件，算法的其他能力不等于已提供变体。

<a id="preparation"></a>
## 准备步骤

```bash
# cwd: repository root
bash samples/vision/pointnet/model/download.sh --target s100
python3 samples/vision/pointnet/runtime/python/main.py --list-models
```

有网络和 PyYAML 的主机或板端均可下载。下载器先写临时文件，再原子安装；不覆盖已有文件。
发布方未给出 digest，因此无法独立认证已有文件。失败后恢复网络并重试显式命令，或向发布方
取得同一个 HBM 放到默认位置。后缀或重命名不能证明目标兼容。

<a id="accompanying-files"></a>
## 伴随文件

`../test_data/chair.pts` 是 XYZ 示例输入，不是校准集或模型参数。CLI 固定标签
back/seat/leg/arm 对应 0/1/2/3，无需额外下载类别词典、MVN 或配置文件。

<a id="local-paths"></a>
## 本地路径

默认 `samples/vision/pointnet/model/s100/pointnet.hbm` 相对 sample 自身定位，不依赖 cwd。
下载器 `--output-dir` 修改 `s100/pointnet.hbm` 的父目录。使用另一个确切制品路径时：

```bash
# cwd: repository root; copy the published S100 HBM to /tmp/pointnet.hbm first
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --asset-id s:pointnet:s100/pointnet.hbm --model-path /tmp/pointnet.hbm
```

<a id="formats-checksums"></a>
## 格式与校验值

格式 `.hbm`；发布清单 `sha256: null (unknown)`。下载器输出的是观测 SHA-256，
用于追踪，不能当作发布方校验值。加载要求单模型、float32 `(1,3,N)` 输入、`(1,N,4)` 输出。
输出 dtype 取自 metadata 并校验；整数 logits 必须有有效 SCALE 参数。
这些检查不能替代待补板测，也不证明任意 HBM 均兼容。
