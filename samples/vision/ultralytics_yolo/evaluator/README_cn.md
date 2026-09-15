# 评测

`eval_yolo_det.py`、`eval_yolo_seg.py`、`eval_yolo_pose.py`、`eval_yolo_cls.py` 共用当前运行类和平台选择；`--help` 查看参数。COCO 评测需要 pycocotools、图片目录及标注 JSON；分类需要 ImageNet `--val-txt` 或 synset `--label-file`。不自动安装依赖。

COCO 类别索引使用标准 COCO 顺序，标注子集也不改变模型类别映射；本评测器不支持自定义类别顺序。不提供标注时，预测导出要求数字图片文件名作为 image_id。`--limit` 只评估所选图片。空检测结果会保存空 JSON 并明确提示跳过指标计算，不能当作精度验收通过。评测 NMS 默认均为 0.70，与 S 运行 CLI 的 0.45 不同；报告指标时需记录阈值和数据集。

原 Benchmark 表格仍由平台 Sample README/Manifest 提供证据，本轮没有重新生成实测指标。板端精度、时延和 C++ 编译仍需发布前验证。

## YOLO26

```bash
python eval_yolo_det.py --family yolo26 --platform s600 --model-path /path/to/yolo26n_detect.hbm --image-dir /data/coco/val2017 --annotation /data/coco/annotations/instances_val2017.json
python eval_yolo_obb.py --family yolo26 --platform x5 --model-path /path/to/yolo26n_obb.bin --image-dir /data/dota/images
```

detect/seg/pose/cls 复用现有评测入口。旧 X5 分类路径自动采用 ordered 标签和 -1 偏移，S 使用文件名标签；共用入口可显式选择 `--val-format ordered --label-offset -1`。旧路径保留数据集位置及阈值默认值。OBB 仅输出预测 JSON，不计算 DOTA AP；`--label-path` 仅兼容旧参数。
