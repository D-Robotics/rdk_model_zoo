<!-- 模板：model/ README（中文）。契约：readme-contract.md §4.2。
     保持锚点；替换 ⟪…⟫；完成后删除引导。sha256 纪律：未知值写
     `sha256: null (unknown)`——禁止猜测或从其他制品复制。 -->

# 模型制品 — ⟪模型名⟫

<a id="artifacts"></a>
## 制品清单

> **必须回答：** 每个制品文件一行，对应其服务的 target 与 pipeline 阶段；获取方式
> （下载/手动/转换产出）。必须与本 sample 的 manifest 行一致。

| 制品 | 格式 | Target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| ⟪file⟫ | ⟪.bin/.hbm/…⟫ | ⟪x5 / s100 / …⟫ | ⟪single / det / rec / …⟫ | ⟪download \| manual \| conversion⟫ |

<a id="preparation"></a>
## 准备步骤

> **必须回答：** 确切命令（cwd、`--target`）或手动步骤；hash 不匹配时的行为；
> 主途径失败时的替代途径。

```bash
# cwd：仓库根目录
bash samples/⟪domain⟫/⟪name⟫/model/download.sh --target ⟪target⟫
# 预期：上表文件落在 model/ 下，sha256 与 manifest 校验通过
```

⟪仅手动准备的制品：获取渠道（内部档案/供应商门户）、所需版本、放置位置。⟫

<a id="accompanying-files"></a>
## 伴随文件

> **必须回答：** 运行所需的每个非制品文件（词典/labels/mvn 统计/配置）的一句话作用
> 与是否必需。

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| ⟪file⟫ | ⟪一句话⟫ | ⟪是/否⟫ |

<a id="local-paths"></a>
## 本地路径

> **必须回答：** 准备完成后文件位于何处；runtime 默认参数指向哪里
> （两者一致性会被机器校验）。

- 制品：`samples/⟪domain⟫/⟪name⟫/model/⟪…⟫`
- runtime 默认 `--model-path`（或等价参数）：⟪path⟫

<a id="formats-checksums"></a>
## 格式与校验值

> **必须回答：** 每个制品的格式与已知 SHA-256 **及数值来源**。未知值写
> `sha256: null (unknown)`——禁止伪造、禁止从同模型其他制品复制。

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| ⟪file⟫ | ⟪format⟫ | ⟪hash 或 `null (unknown)`⟫ | ⟪发布记录 / manifest / unknown⟫ |
