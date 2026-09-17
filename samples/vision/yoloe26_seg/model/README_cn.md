[English](./README.md) | 简体中文

# 模型下载

公开模型目录：

`https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/yoloe26_seg/`

S100 使用 `nash-e/`，S100P 使用 `nash-m/`。每个平台目录包含五个 HBM，
以及匹配的 `.json` 元数据、`.names` 词表和 `manifest.json` 清单。

在本 `model/` 目录执行：

```bash
# 在目标板上自动选择 march。
bash download_model.sh auto n
bash download_model.sh auto all

# 在开发主机上也可以显式指定目标平台。
bash download_model.sh nash-e all
bash download_model.sh nash-m all
```

可选第三个参数为下载目录，默认保存在本示例的 `model/<march>/`。
HBM 文件名为 `yoloe_26<SIZE>_seg_pf_nashe_640x640_nv12.hbm` 或
`yoloe_26<SIZE>_seg_pf_nashm_640x640_nv12.hbm`。

脚本下载的是可部署的 HBM，不是上游 PyTorch 权重。下载通过 HTTPS 进行，
同时校验文件大小和 SHA256；已校验的文件会保留，内容不同的同名文件不会被覆盖。
下载无需账号或密码。

Python 推理会使用匹配的 JSON 校验 HBM；C++ 使用匹配的词表及规范的 HBM 文件名。
请将同一模型的配套文件一起保存。本次发布不包含 S600 模型。
