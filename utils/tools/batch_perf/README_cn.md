# Batch Perf

本工具用于对指定目录下所有以 `.bin` 结尾的模型文件批量执行 `perf` 测试。

## 使用方式

可以将脚本配置为 alias，便于在任意模型目录中调用：

```bash
alias perf='python3 <path to your rdk_model_zoo>/utils/tools/batch_perf/batch_perf.py'
```

然后在模型目录中执行：

```bash
perf .
```

## 说明

- 测试顺序按模型文件大小从小到大排序。
- 线程数量通常从 `1` 到 `MAX_NUM`，`MAX_NUM` 一般设置为 `2`。
