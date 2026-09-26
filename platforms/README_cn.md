# 平台、分支与发布记录

[English](README.md)

当前 X5/S 整合工作在 `develop` 进行，尚未完成客户发布验收。`rdk_x5`、`rdk_s` 是各平台交付线，历史标签保留其发布时完整布局；本次整合不移动标签，也不把开发状态自动升级为发布。X3 保留为历史分发，不属于本轮新增适配范围。

下表来自本仓库 [registry.json](registry.json)。登记的 release tag 是注册表记录，不是对远端“最新版本”的查询。清单位置相对当前整合树：

| 平台 | 交付线 | 登记标签 | 当前清单 | 原平台资料 |
|---|---|---|---|---|
| RDK X5 | `rdk_x5` | `x5-v1.1.3` | [docs/release/x5](../docs/release/x5) | [x5](../platforms/x5/README_cn.md) |
| RDK S100 / S100P / S600 | `rdk_s` | `s-v1.1.2` | [docs/release/s](../docs/release/s) | [s](../platforms/s/README_cn.md) |
| RDK X3 | `rdk_x3` | `x3-v1.1.2` | [platforms/x3/release](../platforms/x3/release) | [x3](../platforms/x3/README_cn.md) |

## 三种路径各自的用途

- 根 `samples/`：统一维护的 Sample，详见 [完整索引](../samples/README_cn.md)。不再只有最初三个试点。
- 根 `docs/release/x5`、`docs/release/s`：当前制品身份、URL、发布事实及历史指标的维护位置；`tools/catalog-publisher` 从注册表定位这些清单。原平台清单保留作迁移来源记录，不应更新两份产生分叉。
- `platforms/{x5,s}/`：原平台资料、尚未迁移的能力和旧入口兼容层。部分入口已转发到根 Sample，单独复制子目录可能缺少依赖，通常需要完整仓库。

历史标签中的 `samples/`、`docs/release/` 或 `release/` 位于当时仓库根目录。清单 `source.path` 必须按对应标签布局解析，不能在旧标签路径前强加 `platforms/`。

## 运行时和制品差异

X5 与 S 都使用名为 `hbm_runtime` 的模块，但底层 SDK 和模型不通用。X5 常见图像制品是 `.bin`、packed NV12；S 是 `.hbm`，S100/S100P/S600 对应 nash-e/m/p，图像常用 Y/UV 双输入。点云、特征、音频等任务有自己的输入协议，不能概括为所有 Sample 都使用 NV12。X3 历史运行时为 `hobot_dnn` / `bpu_infer_lib_x3`。

[硬件身份注册](../docs/release/platforms.json) 与制品是否发布、语言是否实现、板测是否通过是不同维度。无资产不能静默选择另一板卡；未测不能写作通过。Ultralytics C++ 已有 X5/S 输入适配代码，其实际任务及验证限制见 [C++ 指南](../samples/vision/ultralytics_yolo/runtime/cpp/README_cn.md)，旧“仅 X5”的概括不再适用。

## 保留的开发资料

- X5：[仓库规范](x5/docs/Model_Zoo_Repository_Guidelines.md)、[源码资料](x5/docs/source_reference/README.md)、[数据集](x5/datasets)、robotics 源 Sample。
- S：[仓库规范](s/docs/Model_Zoo_Repository_Guidelines.md)、[Python API](s/docs/Python_API_User_Guide.md)、[UCP](s/docs/UCP_User_Guide.md)、[数据集](s/datasets)、speech/VLA 源 Sample。
- ACT/Pi0 仍由根 [.gitmodules](../.gitmodules) 记录 gitlink 入口；引用上游仓库不等于本次已经迁移或验收。
- X3：保留 `demos/`、`resource/`、`release/` 的历史结构，不强套 X5/S 新目录规范。

原平台 README 保留了硬件背景、模型列表、FAQ、转换/运行说明和社区入口。新增共用 Sample 按 [推理契约](../docs/sample-standards/inference-contract.md) 与 [README 契约](../docs/sample-standards/readme-contract.md) 开发；当前范围见 [计划](../docs/superpowers/plans/2026-09-26-host-completion.md) 与 [迁移台账](../docs/releases/unified-migration/x5-s-migration-map.md)。

## 指标与许可

历史 Benchmark 保留原模型、目标、版本和测试条件。注册表、清单数量、目录去重后的数量可能不同，应由目录构建器重新计算，不把旧快照合计当成当前库存。模型/指标 Schema、构建规则和去重测试见 [catalog-publisher](../tools/catalog-publisher)。

X5/S 保留各自 LICENSE；X3 上游没有随附许可文件，未在迁移中补造。统一代码与第三方模型/数据的许可分别核对，不因目录移动变更其来源。
