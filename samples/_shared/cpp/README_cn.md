# 共用原生身份与摘要工具

[English](README.md)

这些 C++17 工具不依赖 Python、OpenCV、板端 SDK 或加密库。将本目录加入编译器
头文件路径；读取实际本机身份时链接 `platform_identity.cc`。SHA-256 和纯身份
匹配函数仅需头文件。

| API | 契约 |
| --- | --- |
| `rdk::sha256_hex(data, size)` | 对恰好 `size` 个可读字节计算小写 64 字符摘要；仅在调用期间借用输入 |
| `rdk::sha256_file(path)` | 按 64 KiB 分块读取普通文件；空字符串代表打开/读取/类型失败；空普通文件仍有非空合法摘要 |
| `rdk::identify_target(NativeIdentity)` | 匹配观测字符串，返回 `x5`、`s100`、`s100p`、`s600`，未知时返回空字符串 |
| `rdk::read_native_identity` | 读取固定本地 sysfs/device-tree 文件并返回独立字符串；没有 SSH、联网、环境变量或 CLI 身份覆盖 |

身份规则对齐 [platforms.json](../../../docs/release/platforms.json) 和共用
[Python 实现](../platforms.py)。非空 SoC 名优先，包括 S100 配合 S100P board_type
的细分；缺失时才读取 socinfo，最后是精确 device-tree 型号。高优先级信息未知时
不会回退到低优先级 X5 别名。SoC/board 字符串忽略大小写，device-tree 型号匹配
区分大小写。识别身份不构成制品兼容性或板测通过的结论。

流式 SHA 实现从 YOLOv5 原生运行证据模块提取，原模块委托这里计算。读取错误、
目录或设备路径不能误报为空内容摘要。哈希相等仅证明与预期摘要对应的字节一致；
没有发布方校验和的观测哈希不能认证发布来源或转换工具链。空模型仍由 sample
预检明确拒绝。

[YOLOE 主机测试](../../vision/yoloe/tests/test_cpp_preflight.py)逐一对照平台注册表
全部别名、识别优先级与 Python 行为，并对二进制/填充边界/分块读取边界数据使用
`hashlib` 对照。原生预检还验证已知 SHA 测试向量，包括一百万个 `a`。
这些是主机检查；两个工具均不执行推理，也不认证板端 SDK。
