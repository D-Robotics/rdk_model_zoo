[English](./README.md) | 简体中文

# 模型转换说明

本目录记录当前 sample 所使用 S100 模型产物的原始 R3D-18 转换信息。

## 源模型

转换流程将 PyTorch `r3d_18` 视频动作分类模型导出为 ONNX。模型输入为短视频片段张量，输出为 Kinetics 400 类 logits。

原始 ONNX 结构图如下：

![R3D-18 ONNX graph](../test_data/readme_img/r3d_18_orig.png)

## 工具链

原始转换说明使用 RDK S 算法工具链 OpenExplorer 3.5.0。

转换过程中，工具链支持 `Conv3D`，但不支持原始模型中的 3D `GlobalAveragePooling`。因此转换流程将该 3D pooling 路径替换为等价的 2D `ReduceMean` 后再编译 HBM 模型。

原始转换截图保留如下，便于追溯：

![Original pooling error](../test_data/readme_img/image-1.png)
![Original 3D pooling](../test_data/readme_img/image.png)
![Pooling replacement](../test_data/readme_img/image-2.png)
![Conversion result](../test_data/readme_img/image-3.png)

## 量化说明

原始记录显示，转换后多数算子相似度大于 0.99，最终量化相似度约为 0.99。

## Runtime 使用的模型产物

Runtime 示例使用以下命令下载预编译 HBM 模型：

```bash
cd ../model
bash download_model.sh s100
```

下载后的文件为：

```text
model/s100/r3d_18.hbm
```

## OE 工具链

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成，不建议在板端执行转换。

- OE Docker 下载文档：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链下载：<https://toolchain.d-robotics.cc/>

请从 OE Docker 下载文档获取适配 RDK S100/S100P 的 OpenExplore CPU Docker 镜像，并按实际文件名加载：

```bash
sudo docker load -i ai_toolchain_ubuntu_22_s100_xxx.tar
sudo docker images
```

启动容器时建议挂载当前仓库并增大共享内存：

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```
