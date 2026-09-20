# Python 运行（bad_cli_drift fixture）

中文版参数表与 parser 一致，漂移仅在英文版。

<a id="environment"></a>
## 环境

板端镜像需含 `hbm_runtime`；主机测试仅需 Python。

<a id="usage"></a>
## 使用

在仓库根目录（cwd）执行：

```bash
python3 samples/tools/fixture/bad_cli_drift/runtime/python/main.py --target x5
```

成功判据：退出码 0。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | auto | 执行目标 |
| `--top-k` / `--topk` | int | 5 | 结果数量 |
| `--priority` | int | 0 | 调度优先级 |
| `--bpu-cores` | int 列表 | [0] | BPU 核索引 |
| `--threshold` | float | null | 可选分数阈值 |

<a id="results"></a>
## 结果

打印 top-5 列表。

<a id="integration-example"></a>
## 集成示例

```python
image = load_bgr("test_data/input.jpg")
task = FixtureTask(selection)
result = task.predict(image)
```

示例内变量均有定义。

<a id="stage-io"></a>
## 三阶段 I/O

`pre_process` → 张量；`forward` → 原始输出；`post_process` → top-5。

<a id="troubleshooting"></a>
## 故障排查

- 模型文件缺失：先执行 `model/download.sh`。
