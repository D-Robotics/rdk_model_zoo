[English](README.md) | 简体中文

# PP-LiteSeg 转换配方

<a id="source-model"></a>
## 源模型

配方继承自 X5 ac11571：PaddleSeg PP-LiteSeg-STDC1，配置 `configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml`，静态 NCHW RGB `(1,3,512,1024)`。从 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) 或自己的训练取得兼容权重。仓库未随附 checkpoint，也未钉住 PaddleSeg 精确版本；导出前须记录二者。该配方不构成发布 BIN 可逐字节复现的证明。

<a id="toolchain-targets"></a>
## 工具链与目标

仅适用 X5，march=bayes-e，无 S 系列配方。源文档使用 PaddlePaddle 3.0.0（其中建议 Python 3.8–3.10）和 OE 1.2.8。导出包兼容性须按选定的 PaddleSeg 版本确认；paddle2onnx/onnx/onnxsim 没有钉住版本。脚本不再隐式安装依赖。下列保留源环境配置细节，本轮主机迁移未实际执行。

```bash
# Export environment, separate from the board SDK environment
python3 -m pip install paddlepaddle==3.0.0 paddle2onnx onnx onnxsim
# Obtain and select the PaddleSeg revision required by your checkpoint first.
git clone https://gitee.com/paddlepaddle/PaddleSeg.git /data/PaddleSeg
python3 -m pip install -r /data/PaddleSeg/requirements.txt
python3 -m pip install -e /data/PaddleSeg
# cwd: repository root; source OE image location (availability not rechecked)
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker run -it --rm -v "$(pwd):/open_explorer" -w /open_explorer openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
# Inside OE container:
hb_mapper --version
hb_perf --version
```

可选源安装包：[OE SDK](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz)、[中文手册](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip)、[英文手册](https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip)。归档 URL 来自源文档，未重新下载验证。

<a id="export"></a>
## 导出

```bash
# cwd: repository root; export environment, trained weights supplied by user
cd samples/vision/pp_liteseg/conversion
PADDLESEG_DIR=/data/PaddleSeg CHECKPOINT=/data/checkpoints/pp_liteseg_stdc1_cityscapes.pdparams EXPORT_DIR="$PWD/inference_model/pp_liteseg_stdc1_cityscapes_1024x512" ONNX_DIR="$PWD/onnx" bash onnx_export/export_pp_liteseg_stdc1_onnx.sh
```

进入 PaddleSeg 前解析 checkpoint 路径；CONFIG 如非绝对路径则相对于该外部仓库。上例 EXPORT_DIR/ONNX_DIR 使用绝对路径以消除 cwd 歧义。源 Paddle 3 配方假定 tools/export.py 产生 model.json/model.pdiparams，再以 opset 11 执行 paddle2onnx，并用 onnxsim 固定尺寸；须核对所选版本的实际文件名。缺少 checkpoint/config 会在导出前失败。预期 ONNX 为 `onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx`。

编译前核对输出名称、shape 和 dtype。统一运行时要求 `(1,512,1024,1)` int32 类别图；尚未证明外部导出配方能产出这一部署边界。logits 输出需要明确验证的图适配或另行支持的运行时契约，改文件名或再做一次 argmax 不能解决契约不符。

<a id="calibration"></a>
## 校准

使用代表性道路场景；源配方建议 20–50 张。脚本将解码的 BGR 转为 RGB，以 INTER_LINEAR 缩放到 1024×512，写出 little-endian float32 NCHW `(1,3,512,1024)` 原始 0..255 值。不要重复归一化：YAML 使用均值 123.675/116.28/103.53，缩放系数 1/58.395、1/57.12、1/57.375。

```bash
# cwd: repository root; Python + OpenCV + NumPy, no OE/board required
cd samples/vision/pp_liteseg/conversion
python3 prepare_calibration.py --src /data/cityscapes/calibration_images --out calibration_data_rgb_f32_1024x512 --width 1024 --height 512 --num 50 --seed 0
```

每个张量 6,291,456 字节。--num 默认 50，--seed 默认 0；递归收集 jpg/jpeg/png/bmp，排序后按需确定性无放回抽样。唯一文件名保留不同目录下同名图片。输出目录或同级 manifest 已存在时拒绝执行，新一轮须选择新目录。选中图片无法解码时直接失败，不静默减少数据集。同级 `calibration_data_rgb_f32_1024x512.manifest.json` 记录 shape、seed 和输入输出摘要，编译目录仅包含原始张量。这些检查验证数据准备，不证明校准质量。

<a id="compile"></a>
## 编译

```bash
# cwd: repository root, inside OE container; ONNX and calibration already prepared
cd samples/vision/pp_liteseg/conversion
hb_mapper checker --model-type onnx --march bayes-e --model onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
hb_mapper makertbin --config ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml --model-type onnx
# Alternative orchestration of checker + makertbin + hb_perf:
bash build_bin.sh
```

正常构建选择直接命令或 build_bin.sh 其中一条路径，不需要两套都运行。检查 checker 日志中的不支持算子。YAML 保留源相对 ONNX/校准路径、输出前缀和 working_dir。脚本预期输出在 ptq_yamls/..._output 下；实际 OE 容器中须确认路径解析。即使 makertbin 返回零，缺少预期 BIN 仍会失败。CAL_SRC 可先准备数据，因此需要全新的校准目录。覆盖 MODEL 或 CONFIG 时须同步 YAML 的 ONNX/校准路径：仅改 MODEL 只影响 checker 输入。

<a id="validation"></a>
## 验证

主机测试以假工具覆盖校准字节/命名与 shell 失败分支；实际导出、编译及性能测试均为 not-run。须检查实际元数据，并在相同输入下逐像素比较类别 ID。源文档的 logits 余弦阈值 ≥0.95 仅在两侧显式暴露相同 argmax 前 logits 时适用，不能用于整数类别 ID。数据集 mIoU 需要标注验证集和数据集执行器，本示例未提供该执行器。

```bash
# cwd: sample directory inside OE; source expected compiler output location
hb_perf conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
# cwd: sample directory on X5, published model explicitly prepared
hrt_model_exec model_info --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
hrt_model_exec perf --model_file model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin --core_id=0 --thread_num=1 --profile_path="."
```

<a id="artifacts"></a>
## 产物

预期产物链：训练 .pdparams → 推理 model.json/model.pdiparams → 原始／简化 ONNX → OE 日志和 *_output BIN → 本地验证报告。为本地编译模型保留 checkpoint、PaddleSeg 版本、环境版本、校准 manifest、模型摘要和日志。不要因文件名相同就覆盖发布模型。自定义文件可显式指定 runtime asset-id 请求相同张量契约，但未知的发布摘要不能认证文件来源。

<a id="known-gaps"></a>
## 已知缺口

缺少钉住的 checkpoint/PaddleSeg/导出包组合；未观察实际导出图或编译产物；实际输出元数据、数值精度和性能均未验证。源文档预期的约 95 FPS／10.5 ms 不是本轮测量结果。不支持算子需要检查，不能盲目删除 argmax，因为运行时边界已经是类别图。把 calibration_type 改为 mix 或补充代表性数据属于需要重新验证的实验，不是保证有效的修复。
