# 项目文档与维护说明

模型浏览、硬件性能、精度与模型下载使用仓库首页的在线模型目录入口。本目录集中保存开发与维护资料。文档随目标 ref 阅读：Skills 的 `rdk_x5` 是维护源，不代表当前 checkout 或用户任务一定面向 X5；目标可能是 X5、S100/S100P/S600、X3 或 legacy。

## 开发文档

- [仓库规范](Model_Zoo_Repository_Guidelines.md)：样例目录、代码和文档约定。
- [Model Zoo Skills](../skills/README.md)：Skills Pack 的维护、许可、独立安装与验证说明。
- [源码参考](source_reference/README.md)：接口与底层组件说明。
- TROS 集成说明：[中文](tros/README_cn.md) / [English](tros/README.md)。

## 维护资料

| 内容 | 位置 |
| --- | --- |
| 网站源码与开发命令 | [catalog/README.md](catalog/README.md) |
| 模型与 Benchmark 清单说明 | [manifests/README.md](manifests/README.md) |
| 模型清单 | [manifests/models.yaml](manifests/models.yaml) |
| Benchmark 清单 | [manifests/benchmarks.yaml](manifests/benchmarks.yaml) |
| 发布规范 | [中文](RELEASE_cn.md) / [English](RELEASE.md) |
| 版本与变更记录 | [VERSION](../VERSION) / [CHANGELOG.md](../CHANGELOG.md) |
| 各版发布说明 | 统一保存在根目录 [CHANGELOG.md](../CHANGELOG.md) |

当前目标 ref 的清单优先从 `docs/manifests/` 读取。历史 ref 可能保留 `docs/release/` 或根 `release/`；多个候选时以目标 ref、Manifest 的 `release.platform`/`source_ref` 和 README 交叉核对，不用维护源的清单替代目标数据。分支或 Tag 只是定位线索，不能单独证明硬件兼容。

```text
docs/
├── assets/           # 文档图片
├── catalog/          # 在线目录源码、构建和测试
├── manifests/        # 当前分支 Manifest 与 schema
├── source_reference/ # 接口参考
└── tros/             # TROS 集成说明
```

当前维护线把网站源码与发布数据放在 `docs/catalog/` 和 `docs/manifests/`。旧 Release Tag 保留其原始目录；目录生成器需要兼容目标历史 ref 中的 `docs/release/` 和根 `release/` 布局。查看 X5、S100/S100P/S600、X3 或 legacy 时，按所选 ref 的实际路径和发布说明读取，文件内容与 Tag 不因目录迁移改变。
