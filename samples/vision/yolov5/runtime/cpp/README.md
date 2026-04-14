

注: 所有的Terminal命令基于以下目录`./rdk_model_zoo/demos/detect/YOLOv8/cpp`

安装依赖
```bash
sudo apt update
# OpenCV
sudo apt install libopencv-dev python3-opencv libopencv-contrib-dev
```


BPU推理库已经在RDK平台的RDK OS中自带.
- 头文件
```bash
/usr/include/dnn
.
├── hb_dnn_ext.h
├── hb_dnn.h
├── hb_dnn_status.h
├── hb_sys.h
└── plugin
    ├── hb_dnn_dtype.h
    ├── hb_dnn_layer.h
    ├── hb_dnn_ndarray.h
    ├── hb_dnn_plugin.h
    └── hb_dnn_tuple.h
```

- 推理库
```bash
/usr/lib/
.
├── libdnn.so
└── libhbrt_bayes_aarch64.so
```


上述头文件和动态库也可以通过OpenExploer发布物获取
OE路径: `package/host/host_package/x5_aarch64/dnn_1.24.5.tar.gz`

清空之前的编译产物 (如果有)
```bash
rm -rf build 
```

编译
```bash
mkdir -p build && cd build
cmake ..
make
```

运行
```bash
./main
```