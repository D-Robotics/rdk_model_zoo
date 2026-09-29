# 示例输入与历史展示

**简体中文** | [English](README.md)

本目录保留固定 S 源提交 `380e1a2` 的四张输入图片与四张历史结果图片，不包含模型、校准数据或 golden 张量。

| 路径 | 用途 |
| --- | --- |
| `image1.jpg`–`image4.jpg` | 单图 VLM 提问的示例输入；用户也可提供自己的图片 |
| `results/image.jpg` | 源项目展示图 |
| `results/test1.jpg`、`test2.jpg`、`test3.jpg` | 源项目历史运行截图，作为 README 演示素材 |

## 使用

完成 [模型准备](../model/README_cn.md) 和 [原生构建](../runtime/cpp/README_cn.md#build) 后，从仓库根目录运行：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 demo vlm --image_path ../../test_data/image1.jpg --prompt "Describe this image"
```

在交互式 `main` 中也可输入 `/image ../../test_data/image1.jpg`，随后提问。相对路径按进程工作目录解析，
不是按图片目录解析。运行输出为终端文本，不会覆盖 `results/` 的历史截图。

## 结果边界

四张图片用于功能演示，不是完整精度数据集；生成措辞可能不同，不以截图逐字匹配作为通过标准。
这些图片也不是转换教程要求的 COCO 校准集。板端 golden 对齐另需内部数据，见 [评测说明](../evaluator/README_cn.md)。
本次迁移没有连接板卡或重新生成截图。
