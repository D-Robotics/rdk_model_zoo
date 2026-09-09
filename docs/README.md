# 项目文档与维护说明

模型浏览、硬件性能、精度与模型下载使用仓库首页的在线模型目录入口。本目录集中保存开发与维护资料。

## 开发文档

- [仓库规范](Model_Zoo_Repository_Guidelines.md)：样例目录、代码和文档约定。
- [源码参考](source_reference/README.md)：接口与底层组件说明。

## 维护资料

| 内容 | 位置 |
| --- | --- |
| 网站源码与开发命令 | [catalog/README.md](catalog/README.md) |
| 模型与 Benchmark 清单说明 | [release/README.md](release/README.md) |
| 模型清单 | [release/models.yaml](release/models.yaml) |
| Benchmark 清单 | [release/benchmarks.yaml](release/benchmarks.yaml) |
| 发布规范 | [中文](RELEASE_cn.md) / [English](RELEASE.md) |
| 版本与变更记录 | [VERSION](../VERSION) / [CHANGELOG.md](../CHANGELOG.md) |
| 各版发布说明 | [releases/](releases/) |

```text
docs/
├── assets/           # 文档图片
├── catalog/          # 在线目录源码、构建和测试
├── release/          # 当前分支 Manifest 与 schema
├── releases/         # 各版发布说明
├── source_reference/ # 接口参考
└── superpowers/      # 设计与实施记录
```

网站源码与发布数据已从根目录的 `site/`、`release/` 收拢到这里。旧 Release Tag 保留其原始目录，目录生成器同时支持两种历史布局；文件内容和 Tag 不因目录迁移改变。
