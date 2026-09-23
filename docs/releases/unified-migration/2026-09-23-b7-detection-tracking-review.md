# B7 本地主机迁移记录

状态：实现中；Board=not-run、Closed=no、delivery=not-ready。B6 主机独立评审已完成；依用户授权继续本地基础开发。固定源、完整能力和缺口见 [source audit](2026-09-23-b7-source-audit.md)。不下载真实模型、不加载真实 SDK、不接板卡或远程电脑，不提交/push/发布。

## 实施和评审边界

- YOLOv5：X5/S 共享三阶段接口和 manifest 身份选择，明确 packed/split NV12、源默认 resize、输出变换、输出排列/NMS差异；ByteTrack 使用其 S detector，跟踪状态独立。C++ 保留两套硬件调用与内存实现，入口统一身份和参数。
- FCOS：三尺寸、15个输出按shape匹配、显式dequant与五层解码；不套普通 YOLO decode。
- YOLOWorld：离线词表与32槽 prompt，图像scale/prompt ID逐调用context，拒绝空prompt。
- LPRNet：保留已打包 .dat 输入与68字符CTC，不编造图像预处理。
- MODNet：手动资产获取、source归一化与几何、独立背景合成；无对应源转换脚本时披露具体缺口。
- ByteTrack：detector与tracker分层，明确状态/reset/非线程安全，修复源显式阶段参数错误；缺失视频由用户提供，runtime不隐式下载。

每样例触及 root/model/runtime/python/conversion/evaluator 双语 README；YOLOv5 另有 C++。各层负责算法/支持与快速路径、资产准备、CLI/API、真实转换范围、数值对照及历史性能；父索引由主任务最后统一更新。入口 main 仅CLI/IO，binding只约束制品metadata，runner只适配原生SDK容器，pre/forward/post与predict可逐调用对照。文档按模板逐项填真实内容，不用章节齐全代替可执行性。

先运行失败契约测试再写实现；以真实源函数+注入SDK做离线数字对照、metadata正负、A/B/A context和README实际API/命令检查。作者自检后交换独立review，再写本批主机结论。所有板端/模型/转换结果保持not-run，校准及复验逐项进入交接队列。

主机依赖补充：为执行真实 CPU ByteTrack，在项目隔离 .venv 安装源版本 lap==0.5.12 / cython-bbox==0.1.5。仅访问包源，不下载模型、不接板卡；结果与版本将写入本批证据。

## 用户暂停（2026-09-23）

用户要求停止并交给先前 Claude Code。全部实现任务已中断，B7 保持 in-progress / changes-required，未做最终全量回归或独立验收。最新中途改动、已知问题、验证快照及剩余六大步详见 [交接文档](2026-09-23-claude-code-handoff.md)。仅整理交接，不继续开发；未提交/push。
