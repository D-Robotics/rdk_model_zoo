[English](README.md) | 简体中文

# 示例输入与演示展示


本目录提供四张输入图片与四张示例结果图片，不包含模型、校准数据或 golden 张量。

| 路径 | 用途 |
| --- | --- |
| `image1.jpg`–`image4.jpg` | 单图 VLM 提问的示例输入；用户也可提供自己的图片 |
| `results/image.jpg` | 源项目展示图 |
| `results/test1.jpg`、`test2.jpg`、`test3.jpg` | 运行会话示例截图，用于演示运行效果 |

## 目录结构

```text
test_data/
├── results/  # results 相关文件
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

## 使用

完成 [模型准备](../model/README_cn.md) 和 [原生构建](../runtime/cpp/README_cn.md#build) 后，从仓库根目录运行：

```bash
cd samples/llm/gemma4-e2b/runtime/cpp
./run.sh --target s600 demo vlm --image_path ../../test_data/image1.jpg --prompt "Describe this image"
```

在交互式 `main` 中先输入 `/image ../../test_data/image1.jpg` 加载图片，再提问。相对路径按进程工作目录解析，
不是按图片目录解析。生成文本显示在终端；`results/` 目录存放运行会话示例截图。

## 结果边界

可用这些图片进行单图提问；生成文本取决于提示词和模型输出。评估数据集准确率时，准备带标签的样本并通过[评估器](../evaluator/README_cn.md)比较预测与参考文本。COCO 校准数据见转换教程。
