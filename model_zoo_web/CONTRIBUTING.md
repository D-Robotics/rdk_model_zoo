# RDK Model Zoo Web 提交指南

本文说明如何把 `samples/` 中已经完成转换、精度验证和板端测试的模型发布到
RDK Model Zoo Web。

## 1. 适用范围

Model Zoo Web 只接收当前仓库 `samples/` 下维护的模型，不从历史
`platforms/` 目录导入模型、指标或下载地址。

提交对象按以下层级组织：

```text
模型来源 / 模型系列 / 任务 / 参数规模 / 芯片平台
```

组织规则：

- 不同模型系列分别建档，例如 YOLO11 和 YOLO26 分开。
- 不同任务分别建档，例如 detect、pose 和 segment 分开。
- 同一系列、同一任务的不同参数规模放在同一个 YAML 中，例如 n、s、m、l、x。
- 同一参数规模的不同芯片平台放在同一个 variant 中，例如 X5、S100 和 S600。
- 同一系列、同一任务共用一张封面图，不为每个参数规模或平台重复制作封面。

对应目录为：

```text
model_zoo_web/data/<domain>/<source>/<series>/<task>.yaml
```

例如：

```text
model_zoo_web/data/vision/ultralytics_yolo/yolo26/detect.yaml
```

## 2. 发布门槛

模型必须满足以下条件后，才能将平台状态标记为 `released`：

- `samples/` 中存在可复现的转换、评测和板端运行入口。
- 已生成目标平台可部署模型，并上传到稳定的 OSS 路径。
- 已完成目标数据集全量精度测试，不使用少量图片结果代替正式精度。
- 已完成 Runtime 单路和 2 路并发测试。
- 已完成 C++ 端到端单路和 2 路流水线测试。
- 已记录输入合同、模型参数量、GFLOPs、Runtime 版本和测试环境。
- 已上传原始 OE HTML 报告，并提供可直接访问的 HTTPS URL。
- 已确认模型许可证名称和官方许可证页面。
- 已准备一张与任务匹配的 16:9 封面图。

缺少任一正式发布证据时，不应伪造数值、URL 或占位下载链接，也不应将状态标记为
`released`。

## 3. 仓库与 OSS 边界

提交到 Git：

- `model_zoo_web/data/` 下的 YAML 元数据。
- 网页源代码、构建器、校验器和文档。
- 对网页固定公共资源的必要修改。

不提交到 Git：

- `.bin`、`.hbm`、`.onnx`、`.pt` 等模型制品。
- 完整 OE HTML、转换工作目录和转换日志。
- COCO、ImageNet 等数据集。
- `model_zoo_web/build/` 和 `model_zoo_web/dist/`。
- `node_modules/`、`__pycache__/` 和本地 workbench 文件。
- AccessKey、Secret、临时签名 URL 或其他凭据。

模型、发布清单、校验和文件和 OE 报告放在 OSS：

```text
models/<source>/<series>/<task>/<size>/<platform>/
├── <deployment-model>
├── release.json
├── SHA256SUMS
└── oe_report.html
```

对象路径中不加入日期、Git 提交或 OE 版本层级。构建日期、提交和工具链版本记录在
发布元数据中。

## 4. Catalog YAML

新模型应复制现有记录作为起点，例如
[`data/vision/ultralytics_yolo/yolo26/detect.yaml`](data/vision/ultralytics_yolo/yolo26/detect.yaml)。

基础结构如下，省略号必须替换为真实数据，不能原样提交：

```yaml
schema_version: 1
id: <source>/<series>/<task>
name: <display name>
domain: vision
source: <sample source>
provider: <upstream provider>
family: <series>
task: <task>
description:
  zh: <只描述模型架构特点>
  en: <architecture-only description>
license:
  name: <license name>
  url: <official HTTPS license page>
sample_path: samples/<path>

variants:
  - size: <n|s|m|l|x|other>
    model:
      parameter_count: <integer>
      gflops: <number>
    input:
      width: 640
      height: 640
      source:
        format: rgb
        dtype: float32
        layout: NCHW
        shape: [1, 3, 640, 640]
        scale: 0.003921568627451
    platforms:
      - platform: x5
        status: released
        released_at: YYYY-MM-DD
        artifact:
          format: bin
          march: bayes-e
          runtime_input: nv12
          url: https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com/models/<source>/<series>/<task>/<size>/<platform>/<file>.bin
          release_manifest_url: <same-directory>/release.json
          checksums_url: <same-directory>/SHA256SUMS
          size_bytes: <integer>
          sha256: <64 lowercase hexadecimal characters>
        reports:
          oe_conversion_url: <same-directory>/oe_report.html
        accuracy:
          dataset: <dataset name>
          task: <evaluation task>
          images: <evaluated image count>
          float_onnx:
            label: <exact baseline identity>
            map_50_95: <number>
            map_50: <number>
            map_75: <number>
          runtime:
            map_50_95: <number>
            map_50: <number>
            map_75: <number>
        performance:
          tool: hrt_model_exec perf
          implementation: native_cpp_cli
          timing_scope: model_runtime
          thread_semantics: runtime_submission_concurrency
          core_id: 1
          warmup_frames_per_condition: 20
          runs_per_condition: 3
          frames_per_run: 200
          stages:
            preprocess: not_measured
            runtime: measured
            postprocess: not_measured
            end_to_end: not_measured
          measurements:
            - threads: 1
              average_latency_ms: <number>
              observed_min_latency_ms: <number>
              observed_max_latency_ms: <number>
              aggregate_fps: <number>
            - threads: 2
              average_latency_ms: <number>
              observed_min_latency_ms: <number>
              observed_max_latency_ms: <number>
              aggregate_fps: <number>
          end_to_end:
            - <single-stream C++ result>
            - <two-stream C++ result>
        provenance:
          repository_commit: <full commit SHA>
          build_id: <release build identity>
```

字段要求：

- `id` 必须等于 `<source>/<family>/<task>`。
- YAML 路径必须与 `domain/source/family/task` 完全对应。
- `sample_path` 必须位于本仓库 `samples/` 下且真实存在。
- 中文和英文描述只说明架构特点，不写板端性能、宣传语或下载说明。
- `license.url` 必须是官方 HTTPS 页面。
- Source 输入和 Runtime 输入必须分开记录，例如 `NCHW float32 RGB` 与
  `NV12 runtime` 不能混写成一个输入格式。
- 模型参数量使用整数，GFLOPs 使用对应输入尺寸下的真实值。
- OSS URL 必须是无查询参数的稳定 HTTPS 地址。
- Web 构建不会下载模型重新计算 SHA256；Catalog 中的 SHA256 用于制品身份和
  OE 数据关联，只做格式与一致性检查，且不会显示在网页上。
- `released_at` 使用实际发布日期，不填写未来日期。

## 5. 精度与性能口径

### 精度

- 明确数据集、任务和完整评测图片数量。
- 同时记录浮点基线和板端 Runtime 结果。
- 浮点基线必须说明真实模型阶段，例如导出的 FP32 ONNX，不能笼统写成原模型。
- COCO Detect 至少记录 mAP50-95、mAP50 和 mAP75。
- 网页中的精度说明必须能追溯到评测命令和输出文件。

### Runtime 性能

- `threads` 表示 Runtime 提交并发数，不表示 OpenCV 或 CPU 前处理线程数。
- 至少提交单线程和 2 线程条件。
- Runtime 延迟只展示模型执行时间时，前处理、后处理和端到端必须标记为
  `not_measured`。
- 不得把 `hrt_model_exec perf` 的 Runtime 延迟冒充端到端延迟。

### C++ 端到端性能

- 端到端范围从内存中的已解码图像开始，到应用可用检测结果结束。
- 文件读取、图片解码、绘图和结果落盘不计入端到端时间。
- 分别记录 preprocess、runtime、postprocess 和 end-to-end。
- 每项记录 mean、p50、p95、min 和 max。
- 单路和 2 路流水线分别记录，2 路吞吐量按总完成帧数与共同墙钟时间计算。
- `runtime_submission_threads` 必须与 `pipeline_streams` 一致。
- 最大性能测试使用全部在线 CPU 线程，并记录 OpenCV 线程数、CPU governor、
  CPU 频率和 BPU 频率。

## 6. 封面与 OE 报告输入

网页发布构建通过一个审核后的输入清单引入封面和结构化 OE 数据：

```json
{
  "schema_version": 1,
  "models": {
    "<source>/<series>/<task>": {
      "cover": "results/<cover>.png"
    }
  },
  "releases": {
    "<source>/<series>/<task>/<size>/<platform>": {
      "oe_data": "runs/<release>/oe_report_data.json"
    }
  }
}
```

相对路径以输入清单所在目录为基准。

封面要求：

- 16:9，主体清楚，适合网页横向卡片和详情页。
- 同一模型系列和任务共用一张封面。
- 图片应直接体现任务，不使用纯装饰性背景代替模型结果。
- 如果封面展示检测框、分割结果或关键点，叠加结果必须来自真实推理。
- 检测框、标签和置信度样式保持简洁，并确保中英文界面均不遮挡主体。

OE 报告要求：

- 原始 `oe_report.html` 存放在 OSS，通过新页面打开，不提交到 Git。
- 结构化 `oe_report_data.json` 可作为发布构建输入，用于站内报告展示。
- 结构化报告中的制品身份必须与 Catalog 中的平台制品一致。
- 不使用空链接、`#` 或伪造报告页面。

## 7. 本地构建与验证

首次准备环境：

```bash
python3 -m pip install -r model_zoo_web/requirements.txt
npm ci --prefix model_zoo_web
```

验证网页空壳和 Catalog：

```bash
npm --prefix model_zoo_web run check
```

生成正式预览并验证。默认使用仓库中的
`model_zoo_web/release/inputs.json`：

```bash
npm --prefix model_zoo_web run build:release

npm --prefix model_zoo_web run check:release
```

本地审核其他输入清单时，可以使用 `MODEL_ZOO_INPUTS` 覆盖默认路径。

本地查看：

```bash
python3 -m http.server 4173 --directory model_zoo_web/dist
```

访问 `http://127.0.0.1:4173/`。

提交前还应手工检查：

- 中文和英文界面都没有混入另一种语言。
- 首页卡片和详情页名称、描述、芯片和任务一致。
- Get Model 先选择芯片，再选择模型类型，下载地址指向正确制品。
- Model files 只显示可部署模型，不显示转换中间 ONNX 或 PT 文件。
- 网页不显示 SHA256，但能显示文件名、大小和下载操作。
- 许可证名称和链接正确。
- Runtime 与端到端指标口径清楚，单路和 2 路数据没有混淆。
- OE 报告入口可用，并在新页面打开原始报告。
- 桌面和移动端没有文字溢出、控件遮挡或异常换行。

## 8. GitHub Pages 发布

CI 和 Pages workflow 与网页源码一起维护在 `model_zoo_web` 分支：

- 普通 push 和相关 PR 只执行完整构建与验证，不发布网站。
- 只有符合 `web-vMAJOR.MINOR.PATCH` 的 Tag 才会部署 Pages。
- Tag 指向的提交必须属于远端 `model_zoo_web` 分支。
- Pages 上传目录固定为 `model_zoo_web/dist`。

首次发布前，由仓库管理员在 `Settings > Pages` 中将发布源设置为
`GitHub Actions`，并确认 `github-pages` environment 允许 `web-v*` Tag。

发布时先推送分支，再创建带注释的 Tag：

```bash
git push upstream model_zoo_web
git tag -a web-v1.0.0 -m "RDK Model Zoo Web v1.0.0"
git push upstream web-v1.0.0
```

现有旧 Pages workflow 如果仍处于启用状态，后续运行可能重新覆盖站点。切换到新
Web 后应在仓库设置中禁用旧 workflow，或将其改为不再执行 Pages 部署。

## 9. Git 提交规范

每个提交只处理一个逻辑部分。不要把模型元数据、网页重构和无关 sample 修改混在
同一个提交中。

推荐顺序：

```text
feat(sample): add <model> conversion and runtime support
feat(web): publish <model> metadata and benchmark results
fix(web): correct <model> presentation or release metadata
docs(web): update model submission guidance
```

不要提交 `build/` 或 `dist/`。提交前执行：

```bash
git diff --check
git status --short
```

## 10. Pull Request Checklist

PR 描述中复制并完成以下清单：

```markdown
## Model Zoo Web submission

- [ ] 模型来自当前仓库 `samples/`，未读取历史 `platforms/` 数据
- [ ] YAML 路径、模型 ID、系列和任务一致
- [ ] 不同参数规模和平台已合并到正确的 task record
- [ ] 模型描述仅说明架构特点，包含中英文
- [ ] 许可证名称和官方页面已确认
- [ ] 部署模型、release.json、SHA256SUMS 和 OE HTML 已上传 OSS
- [ ] OSS URL 为稳定 HTTPS 地址且可直接访问
- [ ] 已完成全量数据集精度评测
- [ ] 已提供 Runtime 单线程和 2 线程结果
- [ ] 已提供 C++ 端到端单路和 2 路结果
- [ ] 封面为 16:9，同系列同任务共用，结果来自真实推理
- [ ] `npm --prefix model_zoo_web run check` 通过
- [ ] release preview 构建和 `check:release` 通过
- [ ] 页面资源使用相对路径，可部署到 GitHub Pages 项目子路径
- [ ] 已人工检查中文、英文、桌面端和移动端
- [ ] PR 未包含模型制品、OE HTML、数据集、凭据或生成目录
```
