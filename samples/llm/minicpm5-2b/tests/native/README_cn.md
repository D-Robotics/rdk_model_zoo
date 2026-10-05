[English](README.md) | [简体中文](README_cn.md)

# 原生核心主机测试

本目录下的 C++ 驱动将生产源码与 `fixtures/` 中的 SDK 替身共同编译并在主机运行。每个替身头部均已注明身份，**不是**厂商 SDK：不做分词、不做 BPU 运算、不做生成，也不代表任何板卡证据。`tests/test_native_core.py` 负责编译与运行；legacy CLI 场景还会执行真实的 `src/main.cc`。

覆盖内容：legacy 单次使用生命周期（R1 回归）、请求构建与贪心采样参数、模板大小限制、S600 临时配置文件在成功/错误/异常路径的清理（R3 回归）、指标有限性与非负校验且保留零值（R2 回归）、阶段边界与两轮会话字段，以及测试自身隔离（CORE-R4 回归）：每个 S600 驱动通过 `scratch_dir.hpp` 以 `mkdtemp` 原子创建唯一 scratch 目录、仅清理自持目录、恢复 `TMPDIR`（含原本未设置状态）；套件还验证敌意 `TMPDIR` 下既有 `model/` 哨兵在所有驱动直连运行后保留，以及多个编译后二进制共享同一 `TMPDIR` 并发运行互不冲突。

```bash
../rdk_model_zoo/.venv/bin/python -m unittest discover -s tests -v
```

环境变量覆盖：`MINICPM_CXX` 指定编译器；`MINICPM_JSON_INCLUDE` 指向 `nlohmann` 头文件目录。未设置覆盖时，由 `tools/host_validation/native_dependencies.py` 依次通过 `pkg-config nlohmann_json` 或标准系统包含根（`/usr/include`、`/usr/local/include`、`/opt/homebrew/include`）发现头文件，不读取仓库外个人路径。无头文件的机器仅跳过依赖 JSON 的 S600 驱动（记录原因，且两个覆盖全部驱动的辅助检查跳过而不是部分执行）；无效覆盖则直接失败。
