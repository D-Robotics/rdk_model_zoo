[English](README.md) | 简体中文

# Catalog 数据发布器

本模块属于 `rdk_model_zoo`，是 `model_zoo_doc/catalog` 看板数据的唯一生成器。

## 本地开发

使用 Node 22 系列的 Node.js 22.12 或更新版本。

```sh
cd utils/tools/catalog-publisher
npm ci
npm run check
```

检查会验证平台清单和源码引用，运行测试与 TypeScript 检查，然后重建并验证可复现产物。输出为 `dist/catalog.json` 和 `dist/catalog.meta.json`（包含 SHA256、字节长度、数据版本和来源信息）。

## 数据来源

`sources.json` 为每个平台定义一个输入。X5 和 S 使用 `worktree` 模式，读取当前检出的清单。X3 使用 `commit` 模式，固定在 `6fcef2b87c12435e11fbd7327ea70d4efd917b1c`；Git 通过 `git show` 读取该源码树。需要完整 Git 历史。`tag` 模式选择不可变的带注释发布标签；缺少对象时会报告所需的 `git fetch origin <sha>` 命令。

| 平台 | 读取的清单 | 版本文件 | 生成的源码链接 |
| --- | --- | --- | --- |
| `x5` | `docs/release/x5/models.yaml`, `benchmarks.yaml` | `docs/release/x5/VERSION` | `blob/<resolved commit>/...` |
| `s` | `docs/release/s/models.yaml`, `benchmarks.yaml` | `docs/release/s/VERSION` | `blob/<resolved commit>/...` |
| `x3` | `6fcef2b8…/platforms/x3/release/models.yaml`, `benchmarks.yaml` | `6fcef2b8…/platforms/x3/VERSION` | `blob/6fcef2b87c12435e11fbd7327ea70d4efd917b1c/platforms/x3/...` |

在 `docs/release/{x5,s}` 中维护 X5 和 S 的模型与基准数据，并维护对应平台的 `VERSION`。构建会检查版本文件与清单发布版本是否一致。X3 catalog 输入读取 `sources.json` 配置的提交树。

### 工作区读取与生成链接

`link_ref`/`link_prefix` 不决定读取内容。它们只标注产物输出：每条记录的 `source_ref`/`source_path_prefix`（消费者将其渲染为 `blob/<ref>/<prefix>/...` 仓库链接），以及 `catalog.meta.json` 来源信息中每个平台的 `ref`。因此，工作区构建读取当前检出内容，链接则使用不可变引用；`--pin` 构建读取固定标签的源码树，链接使用该标签。

对于工作区来源，`sources.json` 配置 `link_ref: "HEAD"`。加载器在输出任何内容前，将其解析为实际构建所用检出的完整 40 位十六进制提交（`git rev-parse HEAD^{commit}`）。因此，无论在 `develop`、`main` 还是 PR 检出中生成，链接都会指向该确切提交；产物不会包含字面值 `HEAD`、分支名或其他可变引用。缺少 Git 上下文，或配置的引用无法解析时，构建会明确失败。历史来源不受影响：X3 固定提交和所有 `--pin` 标签保留原有不可变链接。引用解析仅改变这些标注，不改变任何模型、资源或基准值。实际读取内容仍由来源信息中每个平台的 `manifest_sha256` 固定，可复现性检查以它为依据。

## 数据标识与历史发布

Catalog 格式版本独立于平台版本。任何输出模型、观测或来源信息变化时，`catalog-v1.0.0-<content fingerprint>` 都会变化，即使平台发布标签没有变化。元数据 SHA256 校验整个序列化产物。平台标签另外记录。

可为每个平台选择历史带注释标签；保留其原始根目录布局：

```sh
npm run catalog:build -- --pin x5=x5-v1.1.2 --out dist/historical
```

若标签源码树的平台位于 `platforms/<id>` 前缀下，在冒号后指定此前缀：`--pin x5=<annotated-tag>:platforms/x5`。同样的覆盖方式适用于 `s` 和 `x3`。

固定标签构建从标签实际布局中解析各平台的 VERSION：与清单目录同级，或旧布局的平台根目录；工作区配置的 `version_file` 不用于标签。发布了清单却没有 VERSION 的标签会被拒绝，解析出的版本必须与清单发布版本一致。源码布局变化时，不要重写已发布标签或重命名现有下载 URL。

## 导入文档仓库

源码检查成功后，从 `model_zoo_doc/catalog` 执行：

```sh
npm run catalog:import -- /path/to/rdk_model_zoo/utils/tools/catalog-publisher/dist
npm run check
```

在文档仓库提交导入的快照与锁定文件。常规文档构建验证其自身已提交的快照，不会读取同级检出或生成模型数据。

工作流 `.github/workflows/model-catalog-data.yml` 在 PR、推送至 `develop` 和 `main`、手动触发时运行 `npm run check`。推送和手动运行会将 `catalog.json` 与 `catalog.meta.json` 上传为 `model-catalog-data`；PR 运行验证。网站导入按上述文档仓库命令执行。

## 添加硬件

先添加完整平台发行内容，再扩展 `sources.json`、catalog 平台/硬件类型及归一化映射，并添加测试。选择新的推荐开发板属于展示层选择，无需移动看板或维护中的清单。

静态清单检查不能替代板端推理测试。缺少的测量值保持缺失，不合成数值。
