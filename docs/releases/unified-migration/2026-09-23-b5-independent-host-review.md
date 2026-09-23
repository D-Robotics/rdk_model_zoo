# B5 独立主机评审

review_decision=pass（host）；delivery_readiness=not-ready。Board=not-run、Closed=no。审阅基于 develop 的未提交工作树，base=16c5d04d0c71ebd160400c14fb2be1c2eb2513f0，精确文件身份见[快照](evidence/2026-09-23-b5-working-tree-snapshot.json)。

独立 reviewer：`review_b4_first_four`；与五个sample实现作者不同。主任务据 reviewer 原始结论整理本报告，作者自检和新增fixture不冒充 reviewer 执行。

## Findings 与关闭

| Sample | Finding | 级别 | 关闭依据 |
| --- | --- | --- | --- |
| SigLIP | 自定义相对图片路径被误写成从sample解析 | P2 | 两语言明确默认绝对路径/显式路径按cwd；reviewer逐行确认 |
| DINOv2 | conversion缺固定源码、checkpoint及ONNX依赖的准备命令 | P2 | 两语言恢复完整准备与SHA命令；补源五行量化矩阵；reviewer确认 |
| DINOv2 | ONNX精度流程只有描述却容易理解为可直接复现 | P2 | 迁移baseline自包含代码完整；ONNX精度另标manual/not-implemented；reviewer确认 |
| CLIP | 默认和显式路径语义不完整 | P2 | 默认sample绝对位置、用户相对路径按cwd；reviewer确认 |
| CLIP | 短边/长边round描述错误 | P2 | 短边224、长边比例round；reviewer确认 |
| CLIP | evaluator先构造legacy SDK后验证身份 | P2 | gate先于legacy import/factory；reviewer确认 |
| 3DResNet | 下载README使用flags而shell只接受位置参数，客户无法准备模型 | P1 | 十份README同步位置参数；fake-python shell测试实际校验argv；reviewer确认 |
| 3DResNet | discover测试数量文案过时 | P2 | 移除数字承诺，实际16 tests由reviewer复跑；确认关闭 |

主任务另发现并由作者修复3DResNet裸模块导入冲突与错误embedding命名；独立审阅确认完整importlib/相对导入及classification职责。主任务撤回evaluator的近似tie自动豁免，加入实际执行README的全零tie负例；reviewer确认任何ID不一致均失败并保留完整证据。

## 三维审查

- Repository standards：任务三阶段与predict，binding/runner/IO职责、平台身份、精确manifest、五级双语README、完整源资源均核对。没有新增宽泛豁免。
- Delivery specification：五sample保持分类/图文/特征/视频差异，保留CLIP文本+BPE、SigLIP两子模型、DINO三march与双输出、ViT两量化十分类、3D视频和400JSON标签；不以目录归位代替整合。
- Technical correctness：源前后处理对照、raw纯度、每调用context、default与外部身份、实际CLI/API fixture、异常/shape/dtype、文档命令与本地链接已测试。

独立运行记录：ViT13，SigLIP19，CLIP最终15，DINO代码14/文档17，3D最终16；DINO第18项（文档baseline成功/故意raw差异失败）由主任务补充并执行。最终主机覆盖详见[回归证据](evidence/2026-09-23-b5-local-regression.json)。源码/文档无未关闭主机阻断。

## 未验证与过程偏差

真实板端SDK/模型加载/输出、数据集精度、延迟、OE转换/校准未执行；历史表不是本轮证据。DINO历史ONNX精度仍需独立工具链/数据准备。所有支持声明为source支持+host契约，不能升级客户验收。

reviewer误调用一次本机真实模型下载，root后来实际核对并清理；不是零下载审查。完整事实见[事件证据](evidence/2026-09-23-b5-review-download-incident.json)。模型未执行，未连接板卡，未使用远程电脑。后续shell检查用fake python3，禁止以真实下载器试命令。
