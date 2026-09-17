# ResNet18 模型制品

此目录不提交模型二进制文件。canonical 下载器读取平台发布 Manifest，
将选择的制品写入本目录，是 canonical sample 唯一维护的 URL/格式/hash
路径。

## 选择并下载

支持的引用和目标路径如下：

| 目标 | Manifest 引用 | 本目录下的目标路径 |
| --- | --- | --- |
| X5 | `x5:resnet:resnet18_224x224_nv12.bin` | `resnet18_224x224_nv12.bin` |
| S100 | `s:resnet18:s100/resnet18_224x224_nv12.hbm` | `s100/resnet18_224x224_nv12.hbm` |
| S600 | `s:resnet18:s600/resnet18_224x224_nv12.hbm` | `s600/resnet18_224x224_nv12.hbm` |

在仓库根目录显式下载一个目标：

```bash
python3 samples/vision/resnet/model/download.py --target x5
python3 samples/vision/resnet/model/download.py --target s100
python3 samples/vision/resnet/model/download.py --target s600
```

Shell 形式等价：

```bash
bash samples/vision/resnet/model/download.sh s100
```

Python 模块在没有 `hbm_runtime` 的主机上也可安全导入；只有调用
`download_target` 时才访问网络。它调用 `samples._shared.assets.resolve_asset`，
从 `platforms/x5/docs/release/models.yaml` 或
`platforms/s/docs/release/models.yaml` 读取 URL 和文件格式，不在本目录复制
第二份注册表。

## 下载和校验行为

`download_target` 按需建立目标子目录，先写同目录临时文件，校验长度和
Manifest 中已有的发布者 SHA-256，并用原子且不覆盖的方式安装。已经存在的
文件会先校验，绝不会静默替换。缺少发布者 hash 时会报告来源限制，同时打印
观察到的 digest，便于保存本地证据。失败或空文件不会留下完整模型假象。

下载器只接受上表三个目标。Manifest 没有 ResNet18 S100P 行，不能把 S100
文件重命名后当成 S100P 制品。运行时在使用自定义 `--model-path` 时同样要求
完整限定引用，避免用 `.bin` 或 `.hbm` 文件名选择错误的 packed/split 协议。

## 兼容路径

旧平台命令仍然可用，并委托到这里：

```bash
(cd platforms/x5/samples/vision/resnet/model && bash download.sh)
(cd platforms/s/samples/vision/resnet18/model && bash download_model.sh s100)
(cd platforms/s/samples/vision/resnet18/model && bash download_model.sh s600)
```

旧输出目录布局得到保留，便于与 canonical 流程对照；脚本不再携带第二份
硬编码 URL 注册表。

下载完成后，在运行命令中使用完整引用和生成路径，例如：

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm
```

如果下载失败，先检查 Manifest URL 和网络访问，只清理可能残留的不完整
`.part` 文件后重试。如果已有文件校验失败，请保留该文件用于调查并换用新
目标路径；下载器有意拒绝覆盖它。当前行没有发布者 SHA-256，因此出现 hash
警告是预期现象。
