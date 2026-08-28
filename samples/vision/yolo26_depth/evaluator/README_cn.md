# 评估器(YOLO26 Depth,RDK-S)

## 板端验证状态

`hrt_model_exec perf` 延迟测评已在 S100(nash-e)、S100P(nash-m)、S600(nash-p)三板实测,结果填入下表。

量化 YOLO26 Depth 模型对照 SUNRGBD 基准的板端数值评估。

## 脚本

- `prepare_sunrgbd.py` — 备 SUNRGBD 评估子集。
- `eval_sunrgbd.py` — 用量化 `.hbm` 在子集上推理并计算深度指标(如 AbsRel / RMSE)与 vs float 参考的对数深度余弦。

## 验证结果

三板 × 五规格全部完成板端推理验证;对数深度 vs float 余弦 0.9985–0.9998。

## 延迟

```bash
hrt_model_exec perf --model ../model/nash-e/yolo26n_depth_lite_nashe_768x768.hbm
```

### 实测延迟

> 板端 `hrt_model_exec perf` 跑完后填入,单位 ms(BPU 前向,默认单核)。

| 变体 | S100 (nash-e) | S100P (nash-m) | S600 (nash-p) |
|---|---|---|---|
| n | 3.165 | 2.254 | 1.760 |
| s | 4.490 | 3.244 | 2.363 |
| m | 8.246 | 5.986 | 4.062 |
| l | 9.790 | 7.090 | 4.881 |
| x | 19.059 | 12.853 | 9.097 |
