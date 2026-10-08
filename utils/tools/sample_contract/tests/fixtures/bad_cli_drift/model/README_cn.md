# 模型制品（fixture）

<a id="artifacts"></a>
## 制品清单

| 文件 | Target | 阶段 | 来源 |
| --- | --- | --- | --- |
| `fixture1_224x224_nv12.bin` | x5 | 单阶段 | 下载 |
| `fixture1_224x224_nv12.hbm` | s100 | 单阶段 | 下载 |

<a id="preparation"></a>
## 准备步骤

在仓库根目录执行：

```bash
bash samples/tools/fixture/good_sample/model/download.sh --target x5
```

校验失败即报错退出，无静默回退。

<a id="accompanying-files"></a>
## 伴随文件

`labels.txt` 将输出索引映射为 fixture 类名，运行时必需。

<a id="local-paths"></a>
## 本地路径

准备完成后制品位于本目录；运行时 `--model-path` 与 `--asset-id` 配对后
默认在本目录解析。

<a id="formats-checksums"></a>
## 格式与校验值

| 文件 | 格式 | SHA-256 |
| --- | --- | --- |
| `fixture1_224x224_nv12.bin` | bayes-e `.bin` | null (unknown) |

fixture 制品未发布校验值；未知值保持 `null (unknown)`，禁止跨制品复制。
