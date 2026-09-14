# YOLOv5 C++ Runtime

注：以下命令默认在 `samples/vision/yolov5/runtime/cpp` 目录下执行。

## 依赖说明

RDK OS 已预装 BPU 推理库。若需要安装 OpenCV 开发依赖，可执行：

```bash
sudo apt update
sudo apt install libopencv-dev python3-opencv libopencv-contrib-dev
```

## BPU 推理库

头文件默认位于：

```text
/usr/include/dnn
|-- hb_dnn_ext.h
|-- hb_dnn.h
|-- hb_dnn_status.h
|-- hb_sys.h
`-- plugin
    |-- hb_dnn_dtype.h
    |-- hb_dnn_layer.h
    |-- hb_dnn_ndarray.h
    |-- hb_dnn_plugin.h
    `-- hb_dnn_tuple.h
```

动态库默认位于：

```text
/usr/lib/
|-- libdnn.so
`-- libhbrt_bayes_aarch64.so
```

上述头文件和动态库也可以从 OpenExplorer 发布包获取，参考路径：

```text
package/host/host_package/x5_aarch64/dnn_1.24.5.tar.gz
```

## 编译

```bash
rm -rf build
mkdir -p build
cd build
cmake ..
make
```

## 运行

```bash
./main
```
