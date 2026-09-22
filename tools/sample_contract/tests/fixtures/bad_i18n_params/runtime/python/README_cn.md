# Python 运行（bad_i18n_params fixture，中文）

本 fixture 故意不含 `main.py`：CLI 默认值检查必须记为 skipped 而不是
通过；双语参数对照照常执行。

<a id="environment"></a>
## 环境

板端镜像需含 `hbm_runtime`。

<a id="usage"></a>
## 使用

在仓库根目录（cwd）执行。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | choice | x5 | 执行目标（与英文版漂移） |
| `--topk` | int | 5 | 结果数量（选项名与英文版不一致） |

<a id="results"></a>
## 结果

打印 top-5 列表。

<a id="integration-example"></a>
## 集成示例

```python
image = load_bgr("test_data/input.jpg")
result = task.predict(image)
```

<a id="stage-io"></a>
## 三阶段 I/O

`pre_process` → 张量；`forward` → 原始输出；`post_process` → top-5。

<a id="troubleshooting"></a>
## 故障排查

- 模型文件缺失：先执行 `model/download.sh`。
