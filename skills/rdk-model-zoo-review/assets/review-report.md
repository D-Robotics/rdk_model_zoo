<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Model Zoo Review 报告模板

## Findings

没有 finding 时明确说明。存在时按严重程度排列；不要把不同维度合成一个模糊分数。

| 字段 | 记录内容 |
|---|---|
| ID / severity / confidence | 唯一 ID；blocking/major/minor；confirmed/needs-verification |
| Axis | Repository Standards / Delivery Specification / Technical Correctness & Regressions |
| Change relation | introduced / regression / pre-existing / exposed |
| Location | 精确版本:path:line，或缺失文件的要求位置与目录列表 |
| Rule / requirement | 适用规则来源和 required/platform/convention/proposal 分类 |
| Evidence | 具体源码/命令输出/可信日志，不填未执行结果 |
| Impact | 什么输入、平台、任务或消费者受影响 |
| Minimal fix | 修正方向，默认不代用户改代码 |

## Scope

mode、repository、base/head/merge-base 或本地范围、sample/平台/任务/runtime、已读与未读文件；截断 patch/二进制缺失单列。

## Repository Standards

适用条款与冲突；无问题明确说“本维度未发现问题”。不能把模板或旧分支惯例当成硬标准。

## Delivery Specification

需求来源与逐项状态；没有规格时写 `No delivery specification available`。

## Technical Correctness & Regressions

接口/前后处理/异常/公共工具影响与证据。未验证疑点转到 Open Questions。

## Passed Checks

每个通过项列真实文件或命令和观察结果；不能使用“检查全部通过”代替范围。

## Open Questions

仅未决事实、阻断哪些判断及如何验证，不重复已知上下文。

## Verification Matrix

平台 × 变体 × 任务 × runtime × 输入 × 检查；passed/failed/not-run/not-applicable、required、来源与时间。

## Overall Verdict

`review_decision`：pass / changes-required / needs-rework / insufficient-evidence。

`delivery_readiness`：ready / not-ready / unknown。

列未运行项与限制。已读源码不是已执行板测；作者提供结果不能说成 reviewer 亲测。
