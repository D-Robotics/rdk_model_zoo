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

## UI 与数据的边界

- `app.ts` 组合目录和独立详情，`ui/navigation.ts` 管理可分享的查询参数。
- `ui/filters.ts` 提供硬件快捷栏、任务侧栏、搜索和手机筛选草稿；任务分组集中在 `catalog/task-groups.ts`，新增任务使用稳定 ID，未归类项进入其他任务。
- `catalog/card-view-model.ts` 适配现有数据并生成当前硬件范围的卡片摘要；`ui/model-card.ts` 只负责卡片展示。装饰图形不代表实际模型输出。
- 详情由页面、规格表、精度对照、文件下载和来源组件组合；数值格式继续复用 `catalog/metric-display.ts`。
- 基础样式位于 `styles.css`，目录与卡片、详情分别维护局部样式。页面支持系统／浅色／深色主题以及中英文。

本轮只调整 UI 和程序结构，保留现有清单、指标、文件关联和下载地址。后续数据刷新请遵循 [设计与数据交接要求](../superpowers/specs/2026-09-07-model-labs-ui-framework-design.md)，不要在 DOM 组件中填入模型特例或 Benchmark 数字。
