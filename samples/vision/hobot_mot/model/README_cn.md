[English](README.md) | 简体中文

<a id="overview"></a>
## 模型准备

本目录提供 HobotMot 原生库的源码下载入口和跟踪配置。

<a id="artifacts"></a>
## 制品清单

| 制品 | 使用方 | 角色 |
| --- | --- | --- |
| `vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/`（由 `mot-...tar.gz` 解压） | `runtime/cpp`（两个运行时） | 上游源码：包装层由样例构建编译，预编译 ARM64 `iou_based_mot` 核心及依赖库从中链接——所有目标板通用 |
| `configs/iou2.json` | 两个运行时 | 默认 IOU 2.0 配置（不启用卡尔曼滤波） |
| `configs/iou2_kalman.json` | 两个运行时 | 相同默认值，`use_kalman_filter: true` |

<a id="directory"></a>
## 目录结构

```text
model/
├── configs/
│   ├── iou2.json          # 默认跟踪配置（min_score 0.5）
│   └── iou2_kalman.json   # 卡尔曼滤波变体
├── download.py            # 固定版本拉取 + SHA-256 校验 + 解压
├── download.sh            # 下载入口
└── vendor/                # 由 download.py 创建，下载后生成
```

<a id="preparation"></a>
## 准备步骤

在样例根目录（`samples/vision/hobot_mot`）执行：

```bash
python3 model/download.py
```

脚本下载
`https://codeload.github.com/D-Robotics/mot/tar.gz/0120205146d88c3474bfc2936522e349e169fa8d`，
校验 `sha256: a0bf3b9738af2f7c4be5885ac777cdf93dab6d4dd374a297ceebebe58268b54b`，
解压到 `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/`；脚本可重复
执行（复用已校验的压缩包，`--force` 强制重新下载）。离线替代：在其他机器下载
同一压缩包、核对 SHA-256 后放到
`model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d.tar.gz`，再运行脚本解压。

<a id="accompanying-files"></a>
## 跟踪配置

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `tracker_type` | `"IOU_2.0"` | 跟踪算法 |
| `match_type` | `"IOU"` | 按检测框重叠度关联轨迹 |
| `min_score` | `0.5` | 检测置信度下限 |
| `iou_thres` | `0.2` | IoU 关联阈值 |
| `vanish_frame_count`、`time_gap` | `50`、`40` | 轨迹消失等待时间为两者乘积，单位毫秒 |
| `use_kalman_filter` | `false` | 卡尔曼滤波；`iou2_kalman.json` 中为 `true` |

修改配置后重新创建跟踪器。`use_kalman_filter` 使用 JSON 布尔值 `true` 或 `false`。

<a id="local-paths"></a>
## 本地路径

源码默认保存于 `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/`。
构建时可用 `-DHOBOT_MOT_SOURCE_ROOT=<路径>` 指定其他目录。
运行时默认读取 `model/configs/iou2.json`，使用 `--config` 选择其他配置。

<a id="formats-checksums"></a>
## 格式与校验值

源码包：`mot-0120205146d88c3474bfc2936522e349e169fa8d.tar.gz`（gzip tar）。
SHA-256：`a0bf3b9738af2f7c4be5885ac777cdf93dab6d4dd374a297ceebebe58268b54b`。

`configs/` 中的 JSON 配置随 Sample 提供。

