# OSNet 研究用途正式目录

正式记录位于 `model_zoo_web/data/vision/osnet/osnet/reid.yaml`，包含 OSNet x1.0 的 X5、S100、S100P、S600 四个平台。输入清单位于 `release/inputs.json`，轻量 OE 数据位于 `release/reports/osnet-x1-0-<platform>-oe-data.json`，封面为 `release/assets/osnet-reid-cover-ai.png`。

Archive 模型目录沿用服务器既有结构：`downloads/rdk_model_zoo_web/rdk_<platform>/osnet/<filename>`，模型 URL、字节及哈希不变，目录校验清单只登记模型。

release/OE 附属文件位于 `downloads/rdk_model_zoo_web/assets/rdk_<platform>/osnet/<model-stem>/`。每个平台包含派生 `release.json`、逐字节保留的 `release.source.json`、原始 `oe_report.html`、轻量 `oe_report_data.json` 和 `SHA256SUMS`。派生清单只更新位置并记录原始清单身份，原始状态、研究用途许可和实测数据保留。校验完整下载包时，将模型与四个附属文件放在同一个本地目录。

`approval.json` 保存用户明确同意的研究用途登记范围。用户未提供额外授权；原作者的研究用途说明保留为模型许可依据，BIN/HBM 的额外再分发条款没有被假定已获授权。数据集、原始权重和私有 fixture 不随此记录分发。

Catalog 的 sample_revision 固定到本地 develop 的原生集成提交；每个平台另保留原始转换提交和验收源码归档身份。原生提交需先推送到远端，再推送网站目录，保证上线后的源码链接可访问。本次未推送、创建 Tag 或部署网站。
