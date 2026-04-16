English| [简体中文](./README_cn.md)

Yolo World
=======

# 1. Model introduction

LPRNet is a lightweight end-to-end license plate recognition model proposed by Intel. It can directly output character sequences from input license plate images without the need for traditional character segmentation. The network adopts a fully convolutional design and leverages CTC loss for sequence prediction, which reduces parameter size while improving computational efficiency, making it suitable for real-time deployment on embedded or low-power devices. Moreover, LPRNet demonstrates strong robustness against challenges such as varying illumination, angles, and blurriness, and is widely applied in intelligent transportation, parking management, and urban vehicle monitoring.

# 2. Model download link

The RDK X5 heterogeneous .bin model file is only about 800KB, so it is not placed on the cloud server. Instead, it is stored at the same directory level as the current LPRNet folder.

Before using, rename the file:

```
mv lpr.binn lpr.bin
```

# 3. Input and Output Data

## 3.1 Image Encoder

- Input Data

  | Input Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | image    | FLOAT32  | 1 x 3 x 24 x 94 | NCHW           |

- Output Data

  | Output Data | Data Type | Shape                            | Layout |
  | -------- | -------- | ------------------------------- | ------------ |
  | mean_4.tmp_0    | FLOAT32  | 1 x 68 x 18 | NCHW           |

# 4. Onboard Model Performance

- Test frames: 100
- Frame rate: 266 FPS
- Average latency per frame: 3.75 ms
- BPU utilization: 9%
- ION memory usage: 1.11 MB

# 5. Model Testing 

We provide sample license plate images packed as a .bin file.

The image format follows the example in example.jpg.

Rename the test file:

```
mv test.binn test.bin
```

Run inference on the BPU:

```
python infer.py
```