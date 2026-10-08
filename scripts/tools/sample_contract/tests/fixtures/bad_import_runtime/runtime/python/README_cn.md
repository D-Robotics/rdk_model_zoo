# Python 运行（fixture）

<a id="environment"></a>
## 环境

板端镜像需含 `hbm_runtime`、NumPy 与 OpenCV-Python；`hbm_runtime` 仅存在于
板端镜像，`main.py` 懒加载。

<a id="usage"></a>
## 使用

在仓库根目录（cwd）执行：

```bash
python3 samples/tools/fixture/good_sample/runtime/python/main.py --target x5
```

成功判据：退出码 0 且打印 top-5 列表。

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

`main.py` 打印 top-5 `(label, score)` 列表；仅显式给定保存选项时才写文件。

<a id="integration-example"></a>
## 集成示例

```python
image = load_bgr("test_data/input.jpg")  # 定义见下
task = FixtureTask(selection)
result = task.predict(image)

def load_bgr(path):
    with open(path, "rb") as handle:
        return handle.read()
```

示例内所有输入变量均有定义。

<a id="stage-io"></a>
## 三阶段 I/O

`pre_process` 输入单张 BGR `uint8` 图像，返回 resize 后的张量；`forward`
按绑定张量契约喂入并返回原始输出；`post_process` 返回 top-5
`(label, score)` 列表。

<a id="troubleshooting"></a>
## 故障排查

- 模型文件缺失：先执行 `model/download.sh --target <target>`。
- 未知板卡：runtime 报告未解析的 SoC，不回退到其他 target。
