# 第三方依赖

**简体中文** | [English](./README.md)

本目录用于存放 Gemma4-E2B 示例所依赖的第三方源码。

## tokenizers-cpp

HuggingFace tokenizers 的 C++ 绑定 + sentencepiece，用于推理时的原生 C++
分词（原生推理无需 Python；启动器使用 Python 3）。

**不随 git 提交。** 显式执行 `install_tokenizers_cpp.sh` 时会从
[mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) 拉取
固定 commit 的源码。

启动器和 CMake 不会自动运行安装脚本。从 sample 根目录显式准备：

```bash
bash third_party/install_tokenizers_cpp.sh
```

依赖 `curl` 及外网访问；编译过程还需要 `cargo`（Rust 工具链）来构建
tokenizers 的 Rust binding。 Rust 版本需不低于 1.80；若系统版本过旧，安装脚本会在 `$HOME/.cargo` 下引导安装当前稳定版 rustup 工具链。

## 固定来源与本地变更

- 上游：`mlc-ai/tokenizers-cpp`，源码固定提交 `c586c52f93f7b060753bd2388eb96a105cb7374d`。
- Rust binding：`tokenizers 0.21.2` + `onig`；还需初始化 sentencepiece/msgpack 子模块。
- 下载目录：本目录下 `tokenizers-cpp/`，不提交到 Model Zoo Git 仓库。
- 安装脚本会把 Rust `Cargo.lock` 格式 4 改为 3，这是本地兼容改动，不是新的上游提交。

脚本检查到现有 CMake 文件时会复用目录，没有重新核对其 Git 提交；若来源不明，先保留并核对本地改动，
不要把“目录已存在”当作版本已验证。缺少上述 CMake 文件的目录会被源脚本重新创建，执行前应保存自有内容。
代理可通过 `HTTP_PROXY`/`HTTPS_PROXY` 配置。准备成功应输出 `tokenizers-cpp ready` 或已存在提示；
这只表示依赖准备结束，不表示原生构建或板端推理已成功。

后续从 sample 根目录执行 `bash runtime/cpp/run.sh --target s600 --build`；详见
[构建与运行](../runtime/cpp/README_cn.md#build)。依赖及其子模块各自的许可保留在下载的源码中。
