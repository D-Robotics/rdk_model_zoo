<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# rdk-model-zoo-web-sample 治理卡

| 字段 | 内容 |
| --- | --- |
| Owner | RDK Model Zoo Web maintainers |
| Skill 版本 | 1.0.0 |
| 文档许可证 | CC-BY-4.0；frontmatter 的 Apache-2.0 为 Hub 兼容字段，不覆盖内容许可 |
| 数据分类 | public |
| 主要用途 | 新增或补齐 Model Zoo Web 示例记录时，判断哪些字段必填、哪些可缺、什么叫待补、什么叫完整 |
| 执行边界 | 只写能从示例、发布清单或转换配置核对到的事实；不编造性能、精度、体积或下载地址 |
| 验收状态 | 目录构建与 `npm --prefix model_zoo_web run check` 校验记录形状；板端实测不由本卡宣称通过 |
| 已知风险 | 把 `sourced_benchmarks` 里的历史数字当成网页完整基准；把 S 系列总表当成每一块具体板卡 |
| 更新触发 | `model_zoo_web/scripts/build_catalog.py`、`task-registry.json`、`accuracy-metrics.json` 或详情页展示口径变化 |
