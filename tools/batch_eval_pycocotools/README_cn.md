[English](README.md) | 简体中文

# 批量 COCO 评测

这些脚本用于生成 COCO 格式的预测结果，并通过 `pycocotools` 在 COCO2017 验证集上评测目标检测、实例分割和姿态估计结果。数据准备见[数据集准备指南](cn_COCO2017val.md)。

- `eval_batch_python.py` 和 `eval_batch_cpp.py` 为目录中的 `.bin` 模型运行指定的样例评测程序。
- `eval_pytorch_generate_labels*.py` 及其批量封装脚本生成 PyTorch 模型的预测结果。
- `eval_pycocotools.py`、`eval_pycocotools_seg.py` 和 `eval_pycocotools_pose.py` 分别评测检测框、分割掩码和关键点。

在已安装 `pycocotools` 且准备好真值标注文件的环境中运行评分脚本。目标检测评测程序支持预测 JSON 文件或包含 JSON 文件的目录：

```bash
python3 tools/batch_eval_pycocotools/eval_pycocotools.py \
  --truth /path/to/instances_val2017.json \
  --json /path/to/predictions.json
```

生成预测结果需要对应模型的运行环境、依赖和资源。运行前通过所选脚本的 `--help` 查看参数。
