# 项目文档与维护说明

模型浏览、硬件性能、精度与模型下载使用仓库首页的在线模型目录入口。本目录集中保存开发与维护资料。

## 开发文档

- [仓库规范](Model_Zoo_Repository_Guidelines.md)：样例目录、代码和文档约定。
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

```text
docs/
├── assets/           # 文档图片
├── catalog/          # 在线目录源码、构建和测试
├── manifests/        # 当前分支 Manifest 与 schema
├── source_reference/ # 接口参考
└── tros/             # TROS 集成说明
```

网站源码与发布数据已从根目录的 `site/`、`release/` 收拢到这里。旧 Release Tag 保留其原始目录，目录生成器兼容历史 `release/`、`docs/release/` 和当前 `docs/manifests/` 布局；文件内容和 Tag 不因目录迁移改变。
