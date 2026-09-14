# RDK S 文档索引 / Documentation Index

## 使用与开发

- [仓库规范](./Model_Zoo_Repository_Guidelines.md)：目录、代码和文档约定。
- [Python API 用户手册](./Python_API_User_Guide.md)：`hbm_runtime` 接口说明。
- [UCP 用户手册](./UCP_User_Guide.md)：`libdnn` / `libucp` 接口说明。
- [源码文档](./source_reference/README.md)：接口参考文档及生成方式。
- `assets/`：文档图片资源。

## 版本与发布

- [发布资料说明](./release/README.md)：S 系列清单的范围和数据来源。
- [模型清单](./release/models.yaml) / [Benchmark 清单](./release/benchmarks.yaml)：模型文件、性能和精度记录。
- [当前版本](../VERSION) / [更新记录](../CHANGELOG.md)。
- [S v1.0.0 发布说明](./releases/s-v1.0.0.md) / [历史发布说明目录](./releases/)。
- 统一发布规范：[中文](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/docs/RELEASE_cn.md) / [English](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/docs/RELEASE.md)。三个硬件分支分别维护版本，S 系列使用 `s-vMAJOR.MINOR.PATCH`。

## 在线目录维护

在线目录统一服务 X3、X5、S100、S100P 和 S600。网站源码与维护说明集中在 `rdk_x5` 分支的 [docs/catalog/](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/docs/catalog)，S 分支维护自己的模型与发布数据。

当前分支的发布资料统一位于 `docs/release/`。历史 Tag（例如 `s-v1.0.0`）仍保留当时的 `release/` 路径；查询历史发布时请使用对应 Tag 的链接。
