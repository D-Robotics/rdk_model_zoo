# YOLOv5 conversion

<a id="source-model"></a>

## Source model

The X5 source converts Ultralytics YOLOv5 `v2.0` and `v7.0` branch models with matching pretrained weights:

- v2.0: [branch](https://github.com/ultralytics/yolov5/tree/v2.0), weights `yolov5s_tag2.0.pt`
- v7.0: [branch](https://github.com/ultralytics/yolov5/tree/v7.0), weights `yolov5n.pt`

No upstream commit is pinned and no exporter script or checkpoint ships with this sample; the branch/weight pairing and the detection-head edit below are the source procedure.

<a id="directory"></a>
## Directory structure

```text
conversion/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── yolov5_detect_bayese_640x640_nchw.yaml  # Configuration
└── yolov5_detect_bayese_640x640_nv12.yaml  # Configuration
```

<a id="toolchain-targets"></a>

## Toolchain and targets

Conversion runs on an x86 Linux host inside the OE X5 environment (the same environment supplies `hb_mapper`, `hb_perf`, `hrt_model_exec`); the board is not a conversion host.

The two checked-in YAMLs are X5 Bayes-e configurations for 640x640. They preserve `O3`, latency mode, default calibration, and `scale_value: 0.003921568627451`. S100/S600 Nash-e and S100P Nash-m conversion YAMLs are not included. Do not apply the X5 YAML to an S HBM.

<a id="export"></a>

## Export

Work in an external clone of the upstream repository, not in this sample directory.

### YOLOv5 tag v2.0

Clone the official repository, switch to `v2.0`, and download the matching pretrained weights:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -O yolov5s_tag2.0.pt
python3 -m pip install -r requirements.txt
```

Modify `models/yolo.py` so the detection head exports NHWC tensors:

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

Copy `models/export.py` to the repository root and update the default export arguments:

```bash
cp models/export.py export.py
```

```python
parser.add_argument('--weights', type=str, default='./yolov5s_tag2.0.pt', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
```

Replace the ONNX export block so the model is exported with `opset_version=11`, output names `small / medium / big`, and an optional `onnxsim` simplify pass; then run:

Use this export and simplification block:

```python
# ONNX export
try:
    import onnx
    from onnxsim import simplify

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    model.fuse()  # only for ONNX
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['small', 'medium', 'big'])
    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    # simplify
    onnx_model, check = simplify(
        onnx_model,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(onnx_model, f)
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)
```

```bash
python3 export.py
```

### YOLOv5 tag v7.0

Clone, switch to `v7.0`, and download the weights:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

Keep the same NHWC detection-head modification in `models/yolo.py`. Update `export.py` so it exports ONNX only, uses `opset=11`, and sets output names to `small / medium / big`:

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include', nargs='+', default=['onnx'], help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```

Replace the call inside `export_onnx` with:

```python
torch.onnx.export(
    model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
    im.cpu() if dynamic else im,
    f,
    verbose=False,
    opset_version=opset,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['small', 'medium', 'big'],
    dynamic_axes=dynamic or None)
```

Then run `python3 export.py --weights yolov5n.pt`.

<a id="calibration"></a>

## Calibration

The YAMLs point to `./calibration_data_rgb_f32_coco_640` with `cal_data_type: float32`, `calibration_type: default`. No calibration-data generator is included; that directory of representative COCO tensors is an external prerequisite. Do not call the bundled inference images a calibration set.

<a id="compile"></a>

## Compile

From `samples/vision/yolov5/conversion`, after placing a matching external ONNX and the calibration directory at the paths named by the selected YAML:

```bash
# v2.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml

# v7.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5n_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

The expected artifacts are the YAML prefix under the YAML working directory, e.g. `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`. The NCHW YAML is retained as a source reference; its runtime input declaration is NCHW while the published X5 artifact/runtime path is NV12, so select the config intentionally.

<a id="validation"></a>

## Post-conversion validation

In the OE host/board environment, visualize the compiled model and check its inputs and outputs:

```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

Success requires three `(1,H/stride,W/stride,255)` heads, the expected 640 input metadata, and output behavior matching the selected target path. Run the matching detector command in [the runtime guide](../runtime/python/README.md) to check the resulting artifact.

<a id="artifacts"></a>

## Artifacts

`yolov5_detect_bayese_640x640_nchw.yaml` and `yolov5_detect_bayese_640x640_nv12.yaml` define the X5 NCHW and NV12 conversion profiles. Published runtime artifacts are listed in `model/README.md`.


### Model information and Mapper report examples

The following reports describe the X5 v2.0/v7.0 NV12 conversion. Model loading time and per-node cosine similarity belong to these model/toolchain reports; use the runtime and evaluator commands for application latency or dataset accuracy.

```bash
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-09-11,20:21:38.941.75) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 249.627ms.
This model file has 1 model:
[yolov5s_tag_v2.0_detect_640x640_bayese_nv12]
---------------------------------------------------------------------
[model name]: yolov5s_tag_v2.0_detect_640x640_bayese_nv12

input[0]:
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,640,640,)
aligned shape: (1,3,640,640,)
aligned byte size: 614400
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]:
name: small
valid shape: (1,80,80,255,)
aligned shape: (1,80,80,255,)
aligned byte size: 6528000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (6528000,81600,1020,4,)

output[1]:
name: medium
valid shape: (1,40,40,255,)
aligned shape: (1,40,40,255,)
aligned byte size: 1632000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1632000,40800,1020,4,)

output[2]:
name: big
valid shape: (1,20,20,255,)
aligned shape: (1,20,20,255,)
aligned byte size: 408000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (408000,20400,1020,4,)
```

```bash
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.54.0
[DNN] Runtime version = 1.23.10_(3.15.54 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-09-11,20:09:38.351.997) [HorizonRT] The model builder version = 1.23.8
Load model to DDR cost 141.097ms.
This model file has 1 model:
[yolov5n_tag_v7.0_detect_640x640_bayese_nv12]
---------------------------------------------------------------------
[model name]: yolov5n_tag_v7.0_detect_640x640_bayese_nv12

input[0]:
name: images
input source: HB_DNN_INPUT_FROM_PYRAMID
valid shape: (1,3,640,640,)
aligned shape: (1,3,640,640,)
aligned byte size: 614400
tensor type: HB_DNN_IMG_TYPE_NV12
tensor layout: HB_DNN_LAYOUT_NCHW
quanti type: NONE
stride: (0,0,0,0,)

output[0]:
name: small
valid shape: (1,80,80,255,)
aligned shape: (1,80,80,255,)
aligned byte size: 6528000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (6528000,81600,1020,4,)

output[1]:
name: medium
valid shape: (1,40,40,255,)
aligned shape: (1,40,40,255,)
aligned byte size: 1632000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (1632000,40800,1020,4,)

output[2]:
name: big
valid shape: (1,20,20,255,)
aligned shape: (1,20,20,255,)
aligned byte size: 408000
tensor type: HB_DNN_TENSOR_TYPE_F32
tensor layout: HB_DNN_LAYOUT_NHWC
quanti type: NONE
stride: (408000,20400,1020,4,)
```

```bash
ONNX IR version:          6
Opset version:            ['ai.onnx v11', 'horizon v1']
Producer:                 pytorch v2.1.1
Domain:                   None
Model version:            None
Graph input:
    images:               shape=[1, 3, 640, 640], dtype=FLOAT32
Graph output:
    small:                shape=[1, 80, 80, 255], dtype=FLOAT32
    medium:               shape=[1, 40, 40, 255], dtype=FLOAT32
    big:                  shape=[1, 20, 20, 255], dtype=FLOAT32
2024-09-11 15:44:40,195 file: build.py func: build line No: 39 End to prepare the onnx model.
2024-09-11 15:44:40,450 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_original_float_model.onnx.
2024-09-11 15:44:40,450 file: build.py func: build line No: 36 Start to optimize the model.
2024-09-11 15:44:40,800 file: build.py func: build line No: 39 End to optimize the model.
2024-09-11 15:44:40,806 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_optimized_float_model.onnx.
2024-09-11 15:44:40,806 file: build.py func: build line No: 36 Start to calibrate the model.
2024-09-11 15:44:41,009 file: calibration_data_set.py func: calibration_data_set line No: 82 input name: images,  number_of_samples: 50
2024-09-11 15:44:41,009 file: calibration_data_set.py func: calibration_data_set line No: 93 There are 50 samples in the calibration data set.
2024-09-11 15:44:41,012 file: default_calibrater.py func: default_calibrater line No: 122 Run calibration model with default calibration method.
2024-09-11 15:44:42,100 file: calibrater.py func: calibrater line No: 235 Calibration using batch 8
2024-09-11 15:44:46,659 file: ort.py func: ort line No: 179 Reset batch_size=1 and execute forward again...
2024-09-11 15:47:07,752 file: default_calibrater.py func: default_calibrater line No: 140 Select max-percentile:percentile=0.99995 method.
2024-09-11 15:47:10,778 file: build.py func: build line No: 39 End to calibrate the model.
2024-09-11 15:47:10,800 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_calibrated_model.onnx.
2024-09-11 15:47:10,801 file: build.py func: build line No: 36 Start to quantize the model.
2024-09-11 15:47:12,347 file: build.py func: build line No: 39 End to quantize the model.
2024-09-11 15:47:12,408 file: build.py func: build line No: 197 Saving model: yolov5n_tag_v7.0_detect_640x640_bayese_nv12_quantized_model.onnx.
2024-09-11 15:47:12,732 file: build.py func: build line No: 36 Start to compile the model with march bayes-e.
2024-09-11 15:47:12,867 file: hybrid_build.py func: hybrid_build line No: 133 Compile submodel: main_graph_subgraph_0
2024-09-11 15:47:13,097 file: hbdk_cc.py func: hbdk_cc line No: 115 hbdk-cc parameters:['--O3', '--core-num', '1', '--fast', '--input-layout', 'NHWC', '--output-layout', 'NHWC', '--input-source', 'pyramid']
2024-09-11 15:47:13,097 file: hbdk_cc.py func: hbdk_cc line No: 116 hbdk-cc command used:hbdk-cc -f hbir -m /tmp/tmp_prdnrjp/main_graph_subgraph_0.hbir -o /tmp/tmp_prdnrjp/main_graph_subgraph_0.hbm --march bayes-e --progressbar --O3 --core-num 1 --fast --input-layout NHWC --output-layout NHWC --input-source pyramid
2024-09-11 15:49:27,117 file: tool_utils.py func: tool_utils line No: 326 consumed time 133.975
2024-09-11 15:49:27,211 file: tool_utils.py func: tool_utils line No: 326 FPS=288.9, latency = 3461.4 us, DDR = 16197440 bytes   (see main_graph_subgraph_0.html)
2024-09-11 15:49:27,273 file: build.py func: build line No: 39 End to compile the model with march bayes-e.
2024-09-11 15:49:27,408 file: print_node_info.py func: print_node_info line No: 57 The converted model node information:
================================================================================================================================
Node                                                ON   Subgraph  Type          Cosine Similarity  Threshold   In/Out DataType
---------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzPreprocess  0.999967           127.000000  int8/int8
/model.0/conv/Conv                                  BPU  id(0)     Conv          0.999724           1.127231    int8/int8
/model.0/act/Mul                                    BPU  id(0)     HzSwish       0.999239           22.935776   int8/int8
/model.1/conv/Conv                                  BPU  id(0)     Conv          0.996699           20.392370   int8/int8
/model.1/act/Mul                                    BPU  id(0)     HzSwish       0.995564           69.509407   int8/int8
/model.2/cv1/conv/Conv                              BPU  id(0)     Conv          0.996266           61.730789   int8/int8
/model.2/cv1/act/Mul                                BPU  id(0)     HzSwish       0.996379           32.784687   int8/int8
/model.2/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.987249           14.013276   int8/int8
/model.2/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.987381           24.406996   int8/int8
/model.2/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.983127           10.855639   int8/int8
/model.2/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.987979           15.401920   int8/int8
UNIT_CONV_FOR_/model.2/m/m.0/Add                    BPU  id(0)     Conv          0.996379           14.013276   int8/int8
/model.2/cv2/conv/Conv                              BPU  id(0)     Conv          0.992321           61.730789   int8/int8
/model.2/cv2/act/Mul                                BPU  id(0)     HzSwish       0.993225           62.547588   int8/int8
/model.2/Concat                                     BPU  id(0)     Concat        0.993569           17.940353   int8/int8
/model.2/cv3/conv/Conv                              BPU  id(0)     Conv          0.988253           17.940353   int8/int8
/model.2/cv3/act/Mul                                BPU  id(0)     HzSwish       0.989918           10.508739   int8/int8
/model.3/conv/Conv                                  BPU  id(0)     Conv          0.982787           7.924888    int8/int8
/model.3/act/Mul                                    BPU  id(0)     HzSwish       0.988654           8.092022    int8/int8
/model.4/cv1/conv/Conv                              BPU  id(0)     Conv          0.992336           5.621616    int8/int8
/model.4/cv1/act/Mul                                BPU  id(0)     HzSwish       0.993258           3.872177    int8/int8
/model.4/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.987385           2.466782    int8/int8
/model.4/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.989359           5.410062    int8/int8
/model.4/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.982610           4.081088    int8/int8
/model.4/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.989846           5.585560    int8/int8
UNIT_CONV_FOR_/model.4/m/m.0/Add                    BPU  id(0)     Conv          0.993258           2.466782    int8/int8
/model.4/m/m.1/cv1/conv/Conv                        BPU  id(0)     Conv          0.980726           4.777254    int8/int8
/model.4/m/m.1/cv1/act/Mul                          BPU  id(0)     HzSwish       0.983446           4.889826    int8/int8
/model.4/m/m.1/cv2/conv/Conv                        BPU  id(0)     Conv          0.982318           3.377826    int8/int8
/model.4/m/m.1/cv2/act/Mul                          BPU  id(0)     HzSwish       0.985909           7.605175    int8/int8
UNIT_CONV_FOR_/model.4/m/m.1/Add                    BPU  id(0)     Conv          0.992652           4.777254    int8/int8
/model.4/cv2/conv/Conv                              BPU  id(0)     Conv          0.982260           5.621616    int8/int8
/model.4/cv2/act/Mul                                BPU  id(0)     HzSwish       0.985327           7.687179    int8/int8
/model.4/Concat                                     BPU  id(0)     Concat        0.990299           7.244858    int8/int8
/model.4/cv3/conv/Conv                              BPU  id(0)     Conv          0.982896           7.244858    int8/int8
/model.4/cv3/act/Mul                                BPU  id(0)     HzSwish       0.981238           5.655891    int8/int8
/model.5/conv/Conv                                  BPU  id(0)     Conv          0.980117           3.714849    int8/int8
/model.5/act/Mul                                    BPU  id(0)     HzSwish       0.976186           6.713475    int8/int8
/model.6/cv1/conv/Conv                              BPU  id(0)     Conv          0.988606           3.879521    int8/int8
/model.6/cv1/act/Mul                                BPU  id(0)     HzSwish       0.983306           4.371953    int8/int8
/model.6/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.973573           1.150585    int8/int8
/model.6/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.964966           5.939588    int8/int8
/model.6/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.964386           5.097388    int8/int8
/model.6/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.941962           3.874449    int8/int8
UNIT_CONV_FOR_/model.6/m/m.0/Add                    BPU  id(0)     Conv          0.983306           1.150585    int8/int8
/model.6/m/m.1/cv1/conv/Conv                        BPU  id(0)     Conv          0.948590           2.688997    int8/int8
/model.6/m/m.1/cv1/act/Mul                          BPU  id(0)     HzSwish       0.941272           5.648884    int8/int8
/model.6/m/m.1/cv2/conv/Conv                        BPU  id(0)     Conv          0.944876           4.614841    int8/int8
/model.6/m/m.1/cv2/act/Mul                          BPU  id(0)     HzSwish       0.943065           5.309185    int8/int8
UNIT_CONV_FOR_/model.6/m/m.1/Add                    BPU  id(0)     Conv          0.963641           2.688997    int8/int8
/model.6/m/m.2/cv1/conv/Conv                        BPU  id(0)     Conv          0.950161           5.153328    int8/int8
/model.6/m/m.2/cv1/act/Mul                          BPU  id(0)     HzSwish       0.941146           5.703106    int8/int8
/model.6/m/m.2/cv2/conv/Conv                        BPU  id(0)     Conv          0.933608           4.234711    int8/int8
/model.6/m/m.2/cv2/act/Mul                          BPU  id(0)     HzSwish       0.939521           6.980769    int8/int8
UNIT_CONV_FOR_/model.6/m/m.2/Add                    BPU  id(0)     Conv          0.942982           5.153328    int8/int8
/model.6/cv2/conv/Conv                              BPU  id(0)     Conv          0.964446           3.879521    int8/int8
/model.6/cv2/act/Mul                                BPU  id(0)     HzSwish       0.967606           6.328825    int8/int8
/model.6/Concat                                     BPU  id(0)     Concat        0.956309           6.556569    int8/int8
/model.6/cv3/conv/Conv                              BPU  id(0)     Conv          0.973727           6.556569    int8/int8
/model.6/cv3/act/Mul                                BPU  id(0)     HzSwish       0.957020           5.989245    int8/int8
/model.7/conv/Conv                                  BPU  id(0)     Conv          0.948621           4.072040    int8/int8
/model.7/act/Mul                                    BPU  id(0)     HzSwish       0.904375           6.163434    int8/int8
/model.8/cv1/conv/Conv                              BPU  id(0)     Conv          0.981177           4.374038    int8/int8
/model.8/cv1/act/Mul                                BPU  id(0)     HzSwish       0.973906           4.674935    int8/int8
/model.8/m/m.0/cv1/conv/Conv                        BPU  id(0)     Conv          0.896303           1.916546    int8/int8
/model.8/m/m.0/cv1/act/Mul                          BPU  id(0)     HzSwish       0.869613           8.486115    int8/int8
/model.8/m/m.0/cv2/conv/Conv                        BPU  id(0)     Conv          0.839805           8.420576    int8/int8
/model.8/m/m.0/cv2/act/Mul                          BPU  id(0)     HzSwish       0.865418           8.457147    int8/int8
UNIT_CONV_FOR_/model.8/m/m.0/Add                    BPU  id(0)     Conv          0.973906           1.916546    int8/int8
/model.8/cv2/conv/Conv                              BPU  id(0)     Conv          0.911671           4.374038    int8/int8
/model.8/cv2/act/Mul                                BPU  id(0)     HzSwish       0.893293           7.450050    int8/int8
/model.8/Concat                                     BPU  id(0)     Concat        0.866763           6.790506    int8/int8
/model.8/cv3/conv/Conv                              BPU  id(0)     Conv          0.834800           6.790506    int8/int8
/model.8/cv3/act/Mul                                BPU  id(0)     HzSwish       0.760843           7.942293    int8/int8
/model.9/cv1/conv/Conv                              BPU  id(0)     Conv          0.927259           4.785241    int8/int8
/model.9/cv1/act/Mul                                BPU  id(0)     HzSwish       0.932425           5.132613    int8/int8
/model.9/m/MaxPool                                  BPU  id(0)     MaxPool       0.979879           6.074298    int8/int8
/model.9/m_1/MaxPool                                BPU  id(0)     MaxPool       0.993464           6.074298    int8/int8
/model.9/m_2/MaxPool                                BPU  id(0)     MaxPool       0.995697           6.074298    int8/int8
/model.9/Concat                                     BPU  id(0)     Concat        0.985728           6.074298    int8/int8
/model.9/cv2/conv/Conv                              BPU  id(0)     Conv          0.931007           6.074298    int8/int8
/model.9/cv2/act/Mul                                BPU  id(0)     HzSwish       0.844819           6.023237    int8/int8
/model.10/conv/Conv                                 BPU  id(0)     Conv          0.841115           5.126980    int8/int8
/model.10/act/Mul                                   BPU  id(0)     HzSwish       0.858535           6.567430    int8/int8
/model.11/Resize                                    BPU  id(0)     Resize        0.858531           5.970518    int8/int8
/model.11/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8
...el.6/cv3/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8
/model.12/Concat                                    BPU  id(0)     Concat        0.902319           5.970518    int8/int8
/model.13/cv1/conv/Conv                             BPU  id(0)     Conv          0.947352           5.094691    int8/int8
/model.13/cv1/act/Mul                               BPU  id(0)     HzSwish       0.943633           5.366014    int8/int8
/model.13/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.936760           2.916995    int8/int8
/model.13/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.949624           5.177988    int8/int8
/model.13/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.935308           3.382976    int8/int8
/model.13/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.944433           5.233689    int8/int8
/model.13/cv2/conv/Conv                             BPU  id(0)     Conv          0.931966           5.094691    int8/int8
/model.13/cv2/act/Mul                               BPU  id(0)     HzSwish       0.944159           4.878307    int8/int8
/model.13/Concat                                    BPU  id(0)     Concat        0.944225           3.718071    int8/int8
/model.13/cv3/conv/Conv                             BPU  id(0)     Conv          0.941572           3.718071    int8/int8
/model.13/cv3/act/Mul                               BPU  id(0)     HzSwish       0.918719           6.580989    int8/int8
/model.14/conv/Conv                                 BPU  id(0)     Conv          0.957411           3.749895    int8/int8
/model.14/act/Mul                                   BPU  id(0)     HzSwish       0.957810           4.743949    int8/int8
/model.15/Resize                                    BPU  id(0)     Resize        0.957814           4.288212    int8/int8
/model.15/Resize_output_0_calibrated_Requantize     BPU  id(0)     HzRequantize                                 int8/int8
...el.4/cv3/act/Mul_output_0_calibrated_Requantize  BPU  id(0)     HzRequantize                                 int8/int8
/model.16/Concat                                    BPU  id(0)     Concat        0.962476           4.288212    int8/int8
/model.17/cv1/conv/Conv                             BPU  id(0)     Conv          0.974061           4.112339    int8/int8
/model.17/cv1/act/Mul                               BPU  id(0)     HzSwish       0.983879           4.123950    int8/int8
/model.17/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.980654           2.366742    int8/int8
/model.17/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.982723           3.656986    int8/int8
/model.17/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.964849           2.821121    int8/int8
/model.17/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.959107           7.418919    int8/int8
/model.17/cv2/conv/Conv                             BPU  id(0)     Conv          0.961160           4.112339    int8/int8
/model.17/cv2/act/Mul                               BPU  id(0)     HzSwish       0.947822           8.756002    int8/int8
/model.17/Concat                                    BPU  id(0)     Concat        0.951943           4.927598    int8/int8
/model.17/cv3/conv/Conv                             BPU  id(0)     Conv          0.924104           4.927598    int8/int8
/model.17/cv3/act/Mul                               BPU  id(0)     HzSwish       0.948889           20.271854   int8/int8
/model.18/conv/Conv                                 BPU  id(0)     Conv          0.932838           20.202873   int8/int8
/model.18/act/Mul                                   BPU  id(0)     HzSwish       0.929389           5.688267    int8/int8
/model.19/Concat                                    BPU  id(0)     Concat        0.949843           4.288212    int8/int8
/model.20/cv1/conv/Conv                             BPU  id(0)     Conv          0.908277           4.288212    int8/int8
/model.20/cv1/act/Mul                               BPU  id(0)     HzSwish       0.917393           4.524426    int8/int8
/model.20/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.943680           3.775438    int8/int8
/model.20/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.932270           4.696353    int8/int8
/model.20/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.931095           2.750212    int8/int8
/model.20/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.941526           8.871795    int8/int8
/model.20/cv2/conv/Conv                             BPU  id(0)     Conv          0.914818           4.288212    int8/int8
/model.20/cv2/act/Mul                               BPU  id(0)     HzSwish       0.897235           5.465163    int8/int8
/model.20/Concat                                    BPU  id(0)     Concat        0.923493           4.722447    int8/int8
/model.20/cv3/conv/Conv                             BPU  id(0)     Conv          0.924307           4.722447    int8/int8
/model.20/cv3/act/Mul                               BPU  id(0)     HzSwish       0.931528           21.215944   int8/int8
/model.21/conv/Conv                                 BPU  id(0)     Conv          0.892469           21.054279   int8/int8
/model.21/act/Mul                                   BPU  id(0)     HzSwish       0.886928           7.410546    int8/int8
/model.22/Concat                                    BPU  id(0)     Concat        0.871830           5.970518    int8/int8
/model.23/cv1/conv/Conv                             BPU  id(0)     Conv          0.855442           5.970518    int8/int8
/model.23/cv1/act/Mul                               BPU  id(0)     HzSwish       0.805934           7.578112    int8/int8
/model.23/m/m.0/cv1/conv/Conv                       BPU  id(0)     Conv          0.879370           5.911558    int8/int8
/model.23/m/m.0/cv1/act/Mul                         BPU  id(0)     HzSwish       0.848773           7.282393    int8/int8
/model.23/m/m.0/cv2/conv/Conv                       BPU  id(0)     Conv          0.904308           4.925912    int8/int8
/model.23/m/m.0/cv2/act/Mul                         BPU  id(0)     HzSwish       0.890898           11.411656   int8/int8
/model.23/cv2/conv/Conv                             BPU  id(0)     Conv          0.894634           5.970518    int8/int8
/model.23/cv2/act/Mul                               BPU  id(0)     HzSwish       0.867974           7.260511    int8/int8
/model.23/Concat                                    BPU  id(0)     Concat        0.878335           6.109259    int8/int8
/model.23/cv3/conv/Conv                             BPU  id(0)     Conv          0.907323           6.109259    int8/int8
/model.23/cv3/act/Mul                               BPU  id(0)     HzSwish       0.920191           19.757231   int8/int8
/model.24/m.0/Conv                                  BPU  id(0)     Conv          0.996857           20.202873   int8/int32
/model.24/m.1/Conv                                  BPU  id(0)     Conv          0.997903           21.054279   int8/int32
/model.24/m.2/Conv                                  BPU  id(0)     Conv          0.998209           19.711018   int8/int32
```

<a id="known-gaps"></a>

## Additional preparation

- No pinned upstream commit, checkpoint files, export script, calibration producer, or S conversion recipe is included.
- The source X5 Python path and S path have different physical tensor protocols and NMS/dequant behavior; a shared conversion paragraph cannot replace target-specific metadata.
- Manifest publisher SHA-256 values are unknown.
