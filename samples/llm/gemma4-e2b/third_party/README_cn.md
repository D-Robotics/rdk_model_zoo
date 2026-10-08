# 第三方依赖

**简体中文** | [English](./README.md)

本目录用于存放 Gemma4-E2B 示例所依赖的第三方源码。

## 目录结构

```text
third_party/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── install_tokenizers_cpp.sh  # Shell 脚本
```

## tokenizers-cpp

HuggingFace tokenizers 的 C++ 绑定 + sentencepiece，用于推理时的原生 C++
分词（原生推理无需 Python；启动器使用 Python 3）。

**不随 git 提交。** 显式执行 `install_tokenizers_cpp.sh` 时会从
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) 拉取
固定 commit 的源码。

启动器和 CMake 不会自动运行安装脚本。从 sample 根目录显式准备：

```bash
bash third_party/install_tokenizers_cpp.sh --dry-run
bash third_party/install_tokenizers_cpp.sh
```

准备源码需要 Git、外网访问，以及显式安装的稳定版 Rust 1.80+ 工具链
（`rustc`、`cargo` 位于 PATH 或 `$HOME/.cargo/bin`）。脚本只检查前置条件，不安装或升级 Rust。
`--dry-run` 打印锁定的源码提交与目标目录，不联网、不创建文件，也不要求已有编译工具；`--help` 打印用法。

Rust 缺失或过旧时，请先自行准备工具链。已有 rustup 时，可显式执行
`rustup toolchain install stable --profile minimal` 和 `rustup default stable`，
再用 `rustc --version`、`cargo --version` 确认。后一个命令会更改活动 Rust 工具链；
其他项目需要不同版本时，应改用项目级 override。

## 固定来源与本地变更

- 上游：`mlc-ai/tokenizers-cpp`，源码固定提交 `c586c52f93f7b060753bd2388eb96a105cb7374d`。
- Rust binding：`tokenizers 0.21.2` + `onig`；还需初始化 sentencepiece/msgpack 子模块。
- 下载目录：本目录下 `tokenizers-cpp/`，不提交到 Model Zoo Git 仓库。
- 安装脚本会把 Rust `Cargo.lock` 格式 4 改为 3，这是本地兼容改动，不是新的上游提交。

新源码先在同级临时目录中准备，根提交及递归子模块固定版本检查通过后才移入目标目录。
复用现有目录时，必须匹配固定提交、具有所需 CMake 文件，且无本地改动；仅允许上述精确的 Cargo.lock 格式补丁。
修改过、不完整、版本不符或符号链接形式的目标目录均拒绝并保留。请先检查、另行保存自己的内容，再重试。
克隆失败只清理本次创建的临时目录，不覆盖或重置已有源码。

代理可通过 `HTTP_PROXY`/`HTTPS_PROXY` 配置。成功输出 `tokenizers-cpp ready` 或
`already prepared` 及完整提交号，表示源码准备完成；原生构建在下一步进行。
编译 Rust binding 时仍可能访问包仓库，离线编译需要提前缓存依赖。

后续从 sample 根目录执行 `bash runtime/cpp/run.sh --target s600 --build`；详见
[构建与运行](../runtime/cpp/README_cn.md#build)。依赖及其子模块各自的许可保留在下载的源码中。
