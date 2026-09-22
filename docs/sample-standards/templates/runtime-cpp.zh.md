<!-- 模板：runtime/cpp README（中文）。契约：readme-contract.md §4.4。
     保持锚点；替换 ⟪…⟫；完成后删除引导。本 README 只在存在 C++ 实现时存在——
     缺席时不得在其他层级声称双语言支持。 -->

# C++ 运行 — ⟪模型名⟫

<a id="supported-boards"></a>
## 适用板卡

> **必须回答：** 明确列出本构建可在哪些板卡运行、哪些被排除及原因。“全平台”需要
> 逐板证据，否则不得写。

| 板卡 | 状态 | 说明 |
| --- | --- | --- |
| ⟪board⟫ | ⟪supported-verified / supported-not-run / not-supported⟫ | ⟪原因/链接⟫ |

<a id="dependencies"></a>
## 依赖

> **必须回答：** 交叉编译/主机工具链、板端库（libdnn、libhbucp 等）、头文件搜索路径、
> CMake 版本。版本具体。

- 工具链：⟪如 OE 交叉工具链 ⟪版本⟫⟫
- 板端库：⟪libdnn / libhbucp + 版本⟫
- CMake ≥ ⟪版本⟫

<a id="build"></a>
## 构建

> **必须回答：** 完整命令序列与 cwd；说明 SoC 宏检测（configure 时读取
> `/sys/class/boardinfo/soc_name`）及按 target 的 CMake 开关。

```bash
# cwd：samples/⟪domain⟫/⟪name⟫/runtime/cpp
mkdir -p build && cd build
cmake .. && make -j
# 预期：⟪二进制路径/名称⟫
```

<a id="run"></a>
## 运行

> **必须回答：** cwd、前置条件（制品已准备）、默认与自定义命令、可观察的预期结果。

```bash
# cwd：samples/⟪domain⟫/⟪name⟫/runtime/cpp/build
./⟪binary⟫ --model_path=⟪artifact⟫ ⟪其他参数及实际默认值⟫
# 预期：⟪可观察结果 / 输出文件⟫
```

<a id="parameters"></a>
## 参数

> **必须回答：** 全部 gflags 参数（snake_case）及代码实际定义的默认值——静态机器核对。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_path` | ⟪default⟫ | ⟪…⟫ |

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

> **必须回答：** 对外接口面（config 结构体＋模型类）、资源分配/释放时序（构造函数 vs
> `init()`/`deinit()`）、自由函数数据流（pre_process/infer/post_process 引用传参）、
> 线程边界。引用头文件中的真实符号。

- 配置：`⟪XxxConfig⟫` —— ⟪字段与默认值⟫
- 模型：`⟪XxxModel⟫` —— ⟪init() 加载模型并分配张量；资源在 ⟪deinit()/析构⟫ 释放⟫
- 数据流：`pre_process(…) → infer(…) → post_process(…)`，张量引用传参

<a id="results-interpretation"></a>
## 结果解释

> **必须回答：** stdout/输出文件的含义——格式、坐标约定、退出码。

⟪如每个检测一行：[x1,y1,x2,y2] score class_id（像素坐标）；成功退出码 0；
结果图位于 ⟪path⟫⟫
