[English](./README.md) | 简体中文

# ConvNeXt C++ 运行时

本目录用于保留 `ConvNeXt` 样例的 C++ 运行时占位结构。

## 说明

当前样例实际维护的运行路径是 `runtime/python`。

目前 `runtime/cpp` 目录并未提供完整的 C++ 推理实现，仅保留占位文件以保持目录结构与仓库模板一致。

## 目录结构

```text
.
├── CMakeLists.txt
└── run.sh
```

## 构建

当前 `CMakeLists.txt` 仅输出占位提示，不会生成可运行的推理程序。

```bash
mkdir -p build
cd build
cmake ..
```

## 运行

当前 `run.sh` 会直接提示该样例尚未提供 C++ 推理实现。

```bash
chmod +x run.sh
./run.sh
```

## 备注

- 实际推理请使用 `runtime/python/main.py`。
- 不应将本目录视为当前维护中的 C++ 运行路径。
