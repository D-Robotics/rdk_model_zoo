[English](README.md) | 简体中文

# LiteAvatar 测试数据

[Python 运行时](../runtime/python/README_cn.md) · [模型包](../model/README_cn.md)

这里打包了一个小输入；数字人数据本身需要从 MIT 许可的上游仓库一次性下载。

<a id="bundled"></a>
## 打包文件

| 文件 | 作用 | SHA-256 |
| --- | --- | --- |
| `asr_example.wav` | 示例语音（16 kHz 单声道，30 fps 下输出 166 帧） | `a1bd32dc78493c123f9625a66deee562aed2895f53fbc39f2cca3be7e6f4f20f` |

`asr_example.wav` 随 ModelScope paraformer 包发布（Apache-2.0），是流水线
验证所用的音频。

<a id="avatar-data"></a>
## 数字人数据准备

运行时消费一个数字人数据目录（默认
`samples/vision/lite_avatar/test_data/preload`），包含：

```text
preload/
├── bg_video.h264   # 预处理好的可硬件解码背景（主机一次性转码）
├── bg_video.json   # 对应的已校验 sidecar（几何信息 + 码流 SHA-256）
├── bg_video.mp4    # 原始背景循环视频（MPEG-4 Part 2；转换输入，原样保留）
├── face_box.txt    # 合成 ROI（y1 y2 x1 x2："355 644 278 697"）
└── ref_frames/     # 参考帧 ref_*.jpg（使用前 150 张）
```

下载固定到已验证提交的上游示例数据，校验后解压（一次性，约 5 MB）：

```bash
cd <任意目录>
wget https://github.com/HumanAIGC/lite-avatar/raw/5b7ec850945e03d56fb290b05fb68440c359fa86/data/sample_data.zip
echo "d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b  sample_data.zip" | shasum -a 256 -c -
unzip sample_data.zip -d <repo>/samples/vision/lite_avatar/test_data/
```

自带的 `bg_video.mp4` 为 MPEG-4 Part 2，RDK 视频解码器不支持，因此运行
时读取预处理出的 `bg_video.h264` 与其 `bg_video.json` sidecar。在任意有
FFmpeg 的主机上运行本目录的预处理脚本一次即可——脚本只读原始 `.mp4`，
转码后实测生成的码流并自行写出 sidecar：

```bash
python3 samples/vision/lite_avatar/test_data/prepare_background.py \
    --data samples/vision/lite_avatar/test_data/preload
```

随后把 `bg_video.h264` 与 `bg_video.json` 拷入板上的数字人数据目录。
细节见[运行时背景预处理](../runtime/python/README_cn.md#background-prep)。

该 zip 为 MIT 许可（上游仓库 @`5b7ec850945e03d56fb290b05fb68440c359fa86`）。
除运行时数据外，它还带有网络权重：`net_encode.pt` 与
`net_decode.pt` 是**必需的转换输入**（[转换文档](../conversion/README_cn.md)
中的 ONNX 导出会加载这两个 TorchScript 文件）；`net.pth` 与
`neutral_pose.npy` 本样例不使用。若解压到其他位置，
用 `--avatar-data` 指向该目录。

<a id="directory"></a>
## 目录结构

```text
test_data/
├── README.md              # 英文说明
├── README_cn.md           # 中文说明
├── asr_example.wav        # 示例语音
├── prepare_background.py  # 一次性主机背景预处理
└── preload/               # 解压后的数字人数据（不入库）
```
