# 在线模型目录维护

网站源码收录在文档配套目录，仓库根目录仅保留面向模型样例开发的主要内容。发布清单位于 [../release/](../release/)，完整维护入口见 [文档索引](../README.md)。

从仓库根目录运行（Node.js 22）：

```bash
cd docs/catalog
npm ci
npm run dev
```

验证和生产构建：

```bash
npm run check
```

构建输出为 `docs/catalog/dist/`；依赖、构建输出和生成的 `public/data/catalog.json` 不提交。GitHub Actions 使用同一目录安装、验证与发布，线上地址保持不变。

生成器从固定来源读取多平台清单，兼容旧 Tag 的 `release/` 路径与新布局的 `docs/release/`。目录搬迁不重打 Tag，不改变模型文件地址，也不重新计算 Benchmark。
