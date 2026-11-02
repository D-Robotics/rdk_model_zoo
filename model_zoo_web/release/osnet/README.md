# OSNet 研究用途正式目录

正式记录位于 `model_zoo_web/data/vision/osnet/osnet/reid.yaml`，包含 OSNet x1.0 的 X5、S100、S100P、S600 四个平台。输入清单位于 `release/inputs.json`，轻量 OE 数据位于 `release/reports/osnet-x1-0-<platform>-oe-data.json`，封面为 `release/assets/osnet-reid-cover-ai.png`。

Archive 目录沿用服务器既有结构：`downloads/rdk_model_zoo_web/rdk_<platform>/osnet/<filename>`。四套模型和报告共 20 个文件均完成服务器回读与匿名 HTTPS 哈希、字节数核验；模型与原始 OE 报告未改变。Workbench 的逻辑对象键不作为 Archive 的实际物理目录。

`approval.json` 保存用户明确同意的研究用途登记范围。用户未提供额外授权；原作者的研究用途说明保留为模型许可依据，BIN/HBM 的额外再分发条款没有被假定已获授权。数据集、原始权重和私有 fixture 不随此记录分发。

Catalog 的 sample_revision 固定到本地 develop 的原生集成提交；每个平台另保留原始转换提交和验收源码归档身份。原生提交需先推送到远端，再推送网站目录，保证上线后的源码链接可访问。本次未推送、创建 Tag 或部署网站。
