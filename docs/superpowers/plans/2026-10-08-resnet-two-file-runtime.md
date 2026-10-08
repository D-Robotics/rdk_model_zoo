# ResNet Runtime 职责精简实施计划

目标：保留完整可读的模型三阶段，入口只负责参数和调用，通用能力在公共目录复用。
用户已明确不限制文件数；YOLO 后续可以按任务分文件。本轮先交付 ResNet。

规范：[Python Runtime](../../sample-standards/python-runtime.md)。
本地基线为 develop 的 dc237a332232fe9e8a40913b71c989b4ee56ac64，使用独立工作树。
不修改主任务工作树，不连接板卡，不合并或推送。

1. main.py 保留显式构造模型和 predict；cli.py 聚合参数、制品选择、列表、dry-run 和展示。
2. classify.py 保留模型初始化及 preprocess/infer/postprocess/predict，模型特有默认参数就近声明。
3. 在现有 _shared/image.py、labels.py、model_runner.py 内提供图片读取、标签校验和本地分类加载；无新共享框架或通用模块文件。
4. 用独立共享函数测试、ResNet 行为测试和共享 runner 调用方回归确认边界与数值。移除文件数量断言。
5. 同步中英文 README、规范、接口调整说明和验收记录；编译配方保留，仅更新运行时调用示例与提示。

复核重点：不能为减少文件隐藏模型算法；公共模块不包含 ResNet 专用制品名；
模型帮助与查看模式依赖轻量；NV12、softmax、Top-K 及错误报告保持；板端结果单独记录。
