[English](README.md) | 简体中文

# OSNet 固定输入

<a id="inputs"></a>
## 输入

不提交 Market1501 图像。model/download.py --fixtures 准备私有 images.dat，包含已解码 RGB 人体块及有符号平面参考字节，另有顺序清单和匹配的 32×512 浮点参考。Python 默认读首图，C++ 启动入口检查全部 32 个；也可用 --test-img 提供自有图像。

S 系列使用各平台私有候选包中的 HBM，按 SHA256SUMS 和 artifacts.json 校验后以 --model-path 指定；X5 回读 ZIP/TAR 不含这些 HBM。命令、校准和固定输入要求见 [S 系列 Workbench 流程](../evaluator/S_SERIES_cn.md)。
