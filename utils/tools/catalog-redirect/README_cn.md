[English](README.md) | 简体中文

# Catalog 重定向文件（仅供本地使用，尚未部署）

`https://d-robotics.github.io/rdk_model_zoo/` 是此前发布的模型看板地址。它是单页应用，因此其所有链接都使用同一路径上的查询字符串：`?model=…`、`?task=…`、`?benchmark=performance` 及其组合。这些链接已出现在聊天、问题跟踪和发布说明中，无法收回。

本目录保留此前将看板转发到 `/model_zoo_doc/models/` 的方案。
当前网站在 `main` 的 [`model_zoo_web/`](../../../model_zoo_web/README.md)
中维护，继续使用 `/rdk_model_zoo/` 地址。

本目录中的 `index.html` 实现该转发。

## 状态：尚未发布

**本仓库没有工作流部署此文件；未经明确决策，也不应部署。** 文件保留在这里供评审，以便后续有计划地发布重定向：

- `rdk_model_zoo` 的 GitHub Pages 站点根路径必须继续提供内容，因此需要向该 Pages 环境部署。发布即向该 Pages 环境部署，属于仓库发布方式的变更，数据构建 workflow 不涉及部署。
- 发布会影响在线 URL，是用户可见的对外操作。

专用的 `model-zoo-web-pages.yml` workflow 发布网站；
`model-catalog-data.yml` 继续构建、上传独立的 catalog 数据包。
这两个 workflow 都不部署本目录的重定向文件。

## 发布方式（获批后）

重定向必须部署在旧 Pages 站点根目录，使 `index.html` 对应 `/rdk_model_zoo/` 路径前缀。配置该站点的 Pages 来源，将本目录内容发布为站点根目录，然后验证在线深层链接，例如：

```
https://d-robotics.github.io/rdk_model_zoo/?model=yolov8&benchmark=performance
```

确认其跳转到等价的看板视图，并完整保留查询字符串。目标基础 URL 是 `index.html` 内联脚本顶部的单个常量；文档站基础路径变化时需更新它。

## 为什么使用重定向

该方案原本配合文档站使用的 catalog 数据发布器。发布器和 `docs/release/`
暂时独立保留；当前网站读取自己的 `model_zoo_web/data/` 数据。
