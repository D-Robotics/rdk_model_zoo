# 21 分类样例 forward→infer 委托修正记录（2026-10-05）

> 批次时点记录：最终代码已在本地提交，本文保留执行时的计数和状态。
> 全 51 Sample 的最终源码/主机验收结果见 [Codex 独立验收](2026-10-05-all-sample-codex-review.md)。

> 作者：本地执行者（Claude Code + GLM 委托）。本文是 2026-10-05
> readable-runtime 全样例铺开（提交 `9bc2fdc0`，21 个分类样例批次）
> 的一处委托缺陷的整改记录。**作者自检不冒充独立验收**：该铺开的
> 最终行为验收仍属 Codex 独立评审，本修正同样待其复核。

## 范围（21 样例，仅两个文件面）

convnext / edgenext / efficientformer / efficientformerv2 / efficientnet /
efficientvit / fasternet / fastvit / googlenet / hgnetv2 / mobilenetv1 /
mobilenetv2 / mobilenetv3 / mobilenetv4 / mobileone / repghost / repvgg /
repvit / resnext / vargconvnet / vit，各只改：

- `runtime/python/classify.py`
- `tests/test_predict_entry.py`

不改 shared/、main.py、cli.py、classification.py（3dresnet 等）、README；
参考实现 `samples/vision/resnet/classify.py` 的 forward 本就正确，未动。
`yoloworld.py`/`fcos.py` 存在同形代码行，但不在本委托所有权内，未动。

## 缺陷

新类的 `infer(prepared: PreparedInput)` 调 runner；而兼容别名
`forward` 为同时接受裸 tensor mapping，自行解包后**直接调 runner**：

```python
def forward(self, prepared):                       # 旧
    tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
    return self.runner(tensors)
```

于是子类对 `infer` 的 override / 测试对 `infer` 的 instrumentation 对
`forward` 全部失效——`forward` 成了第二条实现路径，违反"兼容名只是
同一实现的薄委托"约定。既有测试仅比较结果相等（两条路径产出确实
相同），证明不了委托。

## 修正

解包两行归入 `infer`（两种合法输入形式都接受），`forward` 只委托：

```python
def infer(self, prepared):                         # 新
    tensors = prepared.tensors if isinstance(prepared, PreparedInput) else prepared
    return self.runner(tensors)

def forward(self, prepared):                       # 新
    return self.infer(prepared)
```

输入集合、数学、标签、结果均不变（`forward` 原本就把两种形式都送进
同一 runner 调用）。`predict` 链与签名注解不变。

## 行为测试（新增，21 份同文）

`test_forward_delegates_to_infer_for_prepared_and_raw_tensors`：注入
fake runner 后以 `mock.patch.object(model, "infer", return_value=sentinel)`
替换 `infer`，分别调 `forward(PreparedInput)` 与
`forward(prepared.tensors)`，断言：两者返回同一 sentinel；patched
`infer` 恰被调 2 次且收到原样参数（`[prepared, prepared.tensors]`）；
**runner 调用数为 0**（forward 不得绕过 infer 自行执行）。

## 红→绿与回归证据

- 红（实现前，代表样例 convnext，其余 20 份文本 grep 证明逐字同构）：
  套件 44 项中恰新测试 1 项 FAIL——`forward` 返回 runner 产出而非
  sentinel，直接证明旁路。日志：
  `local-execution/20261005-all-sample-readable-runtime/classifier_alias_fix/prefail/convnext_unittest.log`（exit=1）。
- 绿（实现后）：21 样例逐个独立进程
  `python -m unittest discover -s samples/vision/<m>/tests -v`，全部
  exit=0（convnext 44、edgenext 42、efficientformer 41、
  efficientformerv2 42、efficientnet 44、efficientvit 43、fasternet 44、
  fastvit 44、googlenet 26、hgnetv2 32、mobilenetv1 33、mobilenetv2 40、
  mobilenetv3 33、mobilenetv4 36、mobileone 26、repghost 24、repvgg 26、
  repvit 26、resnext 26、vargconvnet 26、vit 29）。逐样例 cmd/exit/日志：
  同目录 `postfix/<m>_tests.log`。
- 静态契约：`tools/sample_contract/check.py --sample samples/vision/<m>`
  21× `0 violations`（仅既有 cli.py/main.py 政策 skip；解包行原已在
  通过检查的 `forward` 内，移入 `infer` 不引入新词汇）。日志：同目录
  `contract_checker.log`。

## 边界与未运行项

- 批量替换脚本（逐文件"恰好一次"匹配否则中止）：
  `local-execution/.../classifier_alias_fix/apply_alias_fix.py`；改动面
  经 `git status` 核对为 21×classify.py + 21×test_predict_entry.py，
  无其它文件。
- 无 CLI 变更，按委托指令未重跑 CLI byte-equivalence。
- 未运行真实 SDK / 板端推理 / 模型下载导出编译；runner 均为主机注入
  fixture。主机绿不等于板端验证。
- 未提交、未 stage；最终验收归 Codex 独立评审。
