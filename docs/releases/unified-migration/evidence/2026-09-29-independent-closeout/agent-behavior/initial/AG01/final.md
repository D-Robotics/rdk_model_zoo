核查完成。以下为按技能(SKILL.md 第 2、4 条 + context-policy)执行的只读检索结果，未运行任何模型或脚本，未修改任何文件。

## 仓库与版本身份

- 目标工作区 `REPO_ROOT`: `/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk-b7-board-integration`
- X5 发布标识来源:`docs/release/x5/models.yaml:2-17` — `platform: x5, version: 1.1.3, tag: x5-v1.1.3, released_at: 2026-09-16`,兼容条件 `RDK X5 / rdk_os >= 3.5.0 / Ubuntu 22.04 aarch64 / TROS-Humble`;`docs/release/x5/VERSION` = `1.1.3`
- 配套性能清单 `docs/release/x5/benchmarks.yaml:2-5` 头部同为 `platform: x5, version: "1.1.3", tag: x5-v1.1.3`,与模型清单标识一致
- 清单自述(`models.yaml:15-17`):仅盘点已发布模型引用，`repository_wide_board_test: false`,不认证运行行为

**清单发现说明**：按 `read_catalog.py` 的发现逻辑，本仓库存在 3 个候选清单(`docs/release/x5/models.yaml`、`docs/release/s/models.yaml`、`platforms/x3/release/models.yaml`),脚本会报 `ambiguous-manifest`;因你明确限定 X5,我显式选择了 `docs/release/x5/` 下的清单对，未依赖分支名推断。

## 已发布的 X5 ResNet 模型

来源:`docs/release/x5/models.yaml:547-559`(记录 `id: resnet`)

| 项目 | 内容 |
|---|---|
| 任务 | image-classification |
| 样例 | `samples/vision/resnet`(目录实际存在，含 python/cpp runtime、conversion、tests) |
| 下载 | `samples/vision/resnet/model/download.sh`,`availability: download` |
| 资产 | **仅 1 个**:`resnet18_224x224_nv12.bin`(224x224 NV12) |

缺项：该资产 `sha256: null` —— 发布哈希未记录，按技能约定视为未知，不得当作校验通过。

ResNet 族相邻条目(按脚本 "resnet" 子串匹配会命中 `resnet` 与 `unet`(名称 "UNet ResNet Family"),以下一并列出但不与 ResNet 分类样本混淆)：
- `resnext`(`models.yaml:560-572`):`ResNeXt50_32x4d_224x224_nv12.bin`,sha256 未记录
- `unet`(`models.yaml:971-999`):resnet18/34/50/101/152 五个 VOC 512x512 backbone 资产，**有** sha256 记录

## 已记录性能(均为发布声明，非本次实测)

**1. ResNet18 分类 —— 唯一直接对应记录**，`docs/release/x5/benchmarks.yaml:479`(记录 `resnet18-x5`):
- latency **2.95 ms**(qualifier: exact)、throughput **449 fps**(qualifier: **lower-bound**),输入 224x224 NV12,hardware RDK X5
- top-1:**71.5%**(float)/ **70.5%**(quantized)
- source:`ref 1e1c64d…`, `samples/vision/resnet/README.md` "## Performance Data",provenance `existing-repository-documentation`

交叉核对(两处一致，且都带保留条件)：
- `samples/vision/resnet/README.md:150-161`:注明数字来自 X5 源发布(rdk_x5 @ac11571, x5-v1.1.3),"not re-measured in this repository",**源未说明 latency/FPS 的线程条件**
- `samples/vision/resnet/evaluator/README.md:89-92`:同一组数字，注明“未在 canonical sample 上重新推导，不代表其结果”

**2. ResNet18 作为 UNet backbone**,`benchmarks.yaml:284-295`:52.72 ms / 18.96 fps @512x512,scope 注明 "historical earlier ResNet18 checkpoint…current board revalidation pending";历史精度记录在 `benchmarks.yaml:2198-2257`(PyTorch FP32 / ONNX Runtime FP32 / X5 PTQ mIoU 0.6197/0.6197/0.6172),均绑定历史 checkpoint、非当前下载复验。

**3. UNet ResNet34/50/101/152**,`benchmarks.yaml:2268-2375`:仅 host FP32 + PTQ 编译精度(mIoU 0.6893/0.6838/0.7094/0.7400),scope 明确 "**board runtime pending**" —— 无板端性能。

**4. ResNeXt50(参考)**，`benchmarks.yaml:474-475`:5.89 ms / 189.61 fps,top-1 76.25%(float)/ 76%(quantized)。

## 缺项与限制汇总

1. `resnet18_224x224_nv12.bin` 发布 sha256 未记录；分类 ResNet 仅发布 resnet18 一个资产，resnet50/152 只有 UNet 变体与转换配置(`conversion/resnet152_config.yaml`),无对应分类 .bin。
2. `resnet18-x5` 记录缺 concurrency/scope 字段，FPS 为 lower-bound;线程条件在各来源中均注明未知。
3. 全部数值是“仓库发布声明”(existing-repository-documentation),**本次未做任何板端实测**，`board_verified` 为 false。
4. 环境限制：本会话未暴露 shell,`read_catalog.py` 无法实际执行，上述为按其逻辑的手动等价只读检索；git HEAD/dirty 状态也无法核实(identity 字段缺失)。
5. 次要漂移：当前工作区 README 标题为 `## Performance data`,清单引用写作 `## Performance Data`,大小写不一致(引用指向 commit ref 下的历史段落)。
6. S 源交付(rdk_s @380e1a2)未发布 ResNet18/50/152 的延迟/精度数字，README 明示不作推断。
