English| [简体中文](./README_cn.md)


# Ultralytics YOLO: You Only Look Once

## Abstract

```bash
D-Robotics OpenExplore Version: >= 3.0.31

Ultralytics YOLO Version: >= 8.3.0
```

## Support Models: 

```bash
- YOLOv5u - Detect
- YOLOv8 - Detect
- YOLO11 - Detect
- YOLO12 - Detect
```

## Introduction to YOLO

![](source/imgs/ultralytics_yolo_detect_performance_comparison.png)


YOLO (You Only Look Once) is a popular object detection and image segmentation model developed by Joseph Redmon and Ali Farhadi of the University of Washington. YOLO was introduced in 2015 and quickly gained popularity due to its high speed and accuracy.

 - YOLOv2, released in 2016, improved upon the original model by incorporating batch normalization, anchor boxes, and dimension clustering.
 - YOLOv3: The third iteration of the YOLO model family, originally by Joseph Redmon, known for its efficient real-time object detection capabilities.
 - YOLOv4: A darknet-native update to YOLOv3, released by Alexey Bochkovskiy in 2020.
 - YOLOv5: An improved version of the YOLO architecture by Ultralytics, offering better performance and speed trade-offs compared to previous versions.
 - YOLOv6: Released by Meituan in 2022, and in use in many of the company's autonomous delivery robots.
 - YOLOv7: Updated YOLO models released in 2022 by the authors of YOLOv4.
 - YOLOv8: The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
 - YOLOv9: An experimental model trained on the Ultralytics YOLOv5 codebase implementing Programmable Gradient Information (PGI).
 - YOLOv10: By Tsinghua University, featuring NMS-free training and efficiency-accuracy driven architecture, delivering state-of-the-art performance and latency.
 - YOLO11 🚀: Ultralytics' latest YOLO models delivering state-of-the-art (SOTA) performance across multiple tasks.
 - YOLO12 builds a YOLO framework centered around attention mechanisms, employing innovative methods and architectural improvements to break the dominance of CNN models within the YOLO series. This enables real-time object detection with faster inference speeds and higher detection accuracy.


## Quick Experience

```bash
# Make Sure your are in this file
$ cd samples/Vision/ultralytics_YOLO_Detect

# Check your workspace
$ tree -L 2
.
|-- README.md     # English Document
|-- README_cn.md  # Chinese Document
|-- py
|   |-- eval_ultralytics_YOLO_Detect_YUV420SP.py # Advance Evaluation
|   `-- ultralytics_YOLO_Detect_YUV420SP.py      # Quick Start
`-- source
    |-- imgs
    |-- reference_hbm_models    # Reference HBM Models
    |-- reference_logs          # Reference logs
    `-- reference_yamls         # Reference yaml configs
```

Run it directly and the model file will be downloaded automatically.

```bash
$ python3 py/ultralytics_YOLO_Detect_YUV420SP.py 
```

If you want to replace other models or use other pictures, you can modify the parameters in the script file.

```bash
$ python3 py/ultralytics_YOLO_Detect_YUV420SP.py -h

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Path to BPU Quantized *.bin Model. RDK X3(Module): Bernoulli2. RDK Ultra: Bayes. RDK X5(Module): Bayes-e. RDK S100: Nash-e. RDK S100P: Nash-m.
  --test-img TEST_IMG   Path to Load Test Image.
  --img-save-path IMG_SAVE_PATH
                        Path to Load Test Image.
  --classes-num CLASSES_NUM
                        Classes Num to Detect.
  --nms-thres NMS_THRES
                        IoU threshold.
  --score-thres SCORE_THRES
                        confidence threshold.
  --reg REG             DFL reg layer.
```

### C++ Inference Experience
Before use, please refer to the Readme in reference_hbm_models to download the corresponding model to the folder. After ensuring the model exists, run the following commands:
```bash
cd rdk_model_zoo_s/samples/Vision/ultralytics_YOLO_Detect/cpp
mkdir build && cd build
cmake .. && make
./main
```

If you want to test your own model, please modify the following macro definitions in the code and recompile:
```c++
#define MODEL_PATH //Model path
#define TEST_IMG_PATH //Test image path  
#define CLASSES_NUM //Number of classes
std::vector<std::string> object_names //Class labels
```


## Result Analysis

![](source/imgs/ultralytics_YOLO_Detect_demo.jpg)

The program automatically downloads the BPU HBM model of YOLO12n-Detect and completes the object detection task of the pictures. The visualization results are saved in the `py_result.jpg` file in the current directory.


## BenchMark - Performance

### RDK S100P

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv5nu | 640×640 | 80 | 1.5 ms / 650.9 FPS (1 thread  ) <br/> 1.7 ms / 1097.7 FPS (2 threads) <br/> 2.3 ms / 1240.9 FPS (3 threads) | 2 ms |  2.6  M  |  7.7   B |  
| YOLOv5su | 640×640 | 80 | 2.1 ms / 461.0 FPS (1 thread  ) <br/> 2.7 ms / 709.4 FPS (2 threads)   | 2 ms |  9.1  M  |  24.0  B |  
| YOLOv5mu | 640×640 | 80 | 3.6 ms / 275.7 FPS (1 thread  ) <br/> 5.7 ms / 346.6 FPS (2 threads)   | 2 ms |  25.1 M  |  64.2  B |  
| YOLOv5lu | 640×640 | 80 | 6.5 ms / 151.2 FPS (1 thread  ) <br/> 11.6 ms / 170.3 FPS (2 threads)  | 2 ms |  53.2 M  |  135.0 B |  
| YOLOv5xu | 640×640 | 80 | 11.6 ms / 85.9 FPS (1 thread  ) <br/> 21.5 ms / 92.4 FPS (2 threads)   | 2 ms |  97.2 M  |  246.4 B |  
| YOLOv8n  | 640×640 | 80 | 1.6 ms / 612.8 FPS (1 thread  ) <br/> 1.8 ms / 1047.2 FPS (2 threads)  | 2 ms |  3.2  M  |  8.7   B |  
| YOLOv8s  | 640×640 | 80 | 2.3 ms / 414.6 FPS (1 thread  ) <br/> 3.2 ms / 599.2 FPS (2 threads)   | 2 ms |  11.2 M  |  28.6  B |  
| YOLOv8m  | 640×640 | 80 | 4.1 ms / 238.6 FPS (1 thread  ) <br/> 6.8 ms / 291.0 FPS (2 threads)   | 2 ms |  25.9 M  |  78.9  B |  
| YOLOv8l  | 640×640 | 80 | 7.8 ms / 127.9 FPS (1 thread  ) <br/> 14.0 ms / 141.4 FPS (2 threads)  | 2 ms |  43.7 M  |  165.2 B |  
| YOLOv8x  | 640×640 | 80 | 11.7 ms / 85.0 FPS (1 thread  ) <br/> 21.7 ms / 91.5 FPS (2 threads)   | 2 ms |  68.2 M  |  257.8 B |  
| YOLO11n  | 640×640 | 80 | 1.6 ms / 585.4 FPS (1 thread  ) <br/> 1.9 ms  / 1028.4 FPS (2 threads) | 2 ms |  2.6  M  |  6.5   B |  
| YOLO11s  | 640×640 | 80 | 2.3 ms / 417.0 FPS (1 thread  ) <br/> 3.2 ms  / 603.2 FPS (2 threads)  | 2 ms |  9.4  M  |  21.5  B |  
| YOLO11m  | 640×640 | 80 | 4.4 ms / 221.6 FPS (1 thread  ) <br/> 7.4 ms  / 266.0 FPS (2 threads)  | 2 ms |  20.1 M  |  68.0  B |  
| YOLO11l  | 640×640 | 80 | 5.5 ms / 178.4 FPS (1 thread  ) <br/> 9.6 ms  / 206.1 FPS (2 threads)  | 2 ms |  25.3 M  |  86.9  B |  
| YOLO11x  | 640×640 | 80 | 9.8 ms / 100.9 FPS (1 thread  ) <br/> 18.1 ms / 109.6 FPS (2 threads)  | 2 ms |  56.9 M  |  194.9 B |  
| YOLO12n  | 640×640 | 80 | 2.5 ms / 395.4 FPS (1 thread  ) <br/> 3.5 ms / 554.0 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO12s  | 640×640 | 80 | 4.0 ms / 247.8 FPS (1 thread  ) <br/> 6.5 ms / 304.6 FPS (2 threads)   | 2 ms |  9.3  M  |  21.4  B |  
| YOLO12m  | 640×640 | 80 | 7.1 ms / 139.5 FPS (1 thread  ) <br/> 12.7 ms / 155.8 FPS (2 threads)  | 2 ms |  20.2 M  |  67.5  B |  
| YOLO12l  | 640×640 | 80 | 11.2 ms / 88.4 FPS (1 thread  ) <br/> 20.9 ms / 95.0 FPS (2 threads)   | 2 ms |  26.4 M  |  88.9  B |  
| YOLO12x  | 640×640 | 80 | 18.9 ms / 52.7 FPS (1 thread  ) <br/> 36.2 ms / 55.0 FPS (2 threads)   | 2 ms |  59.1 M  |  199.0 B |  



### RDK S100

| Model | Size(Pixels) | Classes |  BPU Task Latency  /<br>BPU Throughput (Threads) | CPU Latency<br>(Single Core) | params(M) | FLOPs(B) |
|----------|---------|----|---------|---------|----------|----------|
| YOLOv5nu | 640×640 | 80 | 2.0 ms / 487.0 FPS (1 thread  ) <br/> 2.2 ms / 864.5 FPS (2 threads) <br/> 3.2 ms / 896.6 FPS (3 threads) | 2 ms |  2.6  M  |  7.7   B |  
| YOLOv5su | 640×640 | 80 | 2.9 ms / 340.5 FPS (1 thread  ) <br/> 3.9 ms / 498.2 FPS (2 threads)   | 2 ms |  9.1  M  |  24.0  B |  
| YOLOv5mu | 640×640 | 80 | 5.0 ms / 196.2 FPS (1 thread  ) <br/> 8.2 ms / 239.9 FPS (2 threads)   | 2 ms |  25.1 M  |  64.2  B |  
| YOLOv5lu | 640×640 | 80 | 9.5 ms / 104.0 FPS (1 thread  ) <br/> 17.2 ms / 115.5 FPS (2 threads)  | 2 ms |  53.2 M  |  135.0 B |  
| YOLOv5xu | 640×640 | 80 | 16.9 ms / 58.9 FPS (1 thread  ) <br/> 31.8 ms / 62.6 FPS (2 threads)   | 2 ms |  97.2 M  |  246.4 B |  
| YOLOv8n  | 640×640 | 80 | 2.0 ms / 485.1 FPS (1 thread  ) <br/> 2.4 ms / 798.0 FPS (2 threads)   | 2 ms |  3.2  M  |  8.7   B |  
| YOLOv8s  | 640×640 | 80 | 3.1 ms / 312.7 FPS (1 thread  ) <br/> 4.7 ms / 416.7 FPS (2 threads)   | 2 ms |  11.2 M  |  28.6  B |  
| YOLOv8m  | 640×640 | 80 | 5.8 ms / 170.0 FPS (1 thread  ) <br/> 10.0 ms / 198.3 FPS (2 threads)  | 2 ms |  25.9 M  |  78.9  B |  
| YOLOv8l  | 640×640 | 80 | 11.1 ms / 89.1 FPS (1 thread  ) <br/> 20.4 ms / 97.3 FPS (2 threads)   | 2 ms |  43.7 M  |  165.2 B |  
| YOLOv8x  | 640×640 | 80 | 17.0 ms / 58.6 FPS (1 thread  ) <br/> 31.9 ms / 62.3 FPS (2 threads)   | 2 ms |  68.2 M  |  257.8 B |  
| YOLO11n  | 640×640 | 80 | 2.1 ms / 466.6 FPS (1 thread  ) <br/> 2.6 ms / 741.0 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO11s  | 640×640 | 80 | 3.1 ms / 313.9 FPS (1 thread  ) <br/> 4.7 ms / 419.8 FPS (2 threads)   | 2 ms |  9.4  M  |  21.5  B |  
| YOLO11m  | 640×640 | 80 | 6.3 ms / 157.3 FPS (1 thread  ) <br/> 10.9 ms / 181.9 FPS (2 threads)  | 2 ms |  20.1 M  |  68.0  B |  
| YOLO11l  | 640×640 | 80 | 7.9 ms / 125.8 FPS (1 thread  ) <br/> 14.0 ms / 141.5 FPS (2 threads)  | 2 ms |  25.3 M  |  86.9  B |  
| YOLO11x  | 640×640 | 80 | 14.1 ms / 70.3 FPS (1 thread  ) <br/> 26.4 ms / 75.4 FPS (2 threads)   | 2 ms |  56.9 M  |  194.9 B |  
| YOLO12n  | 640×640 | 80 | 3.3 ms / 293.3 FPS (1 thread  ) <br/> 5.2 ms / 382.1 FPS (2 threads)   | 2 ms |  2.6  M  |  6.5   B |  
| YOLO12s  | 640×640 | 80 | 5.6 ms / 174.7 FPS (1 thread  ) <br/> 9.7 ms / 204.7 FPS (2 threads)   | 2 ms |  9.3  M  |  21.4  B |  
| YOLO12m  | 640×640 | 80 | 10.4 ms / 95.7 FPS (1 thread  ) <br/> 18.9 ms / 104.8 FPS (2 threads)  | 2 ms |  20.2 M  |  67.5  B |  
| YOLO12l  | 640×640 | 80 | 16.6 ms / 60.1 FPS (1 thread  ) <br/> 31.2 ms / 63.8 FPS (2 threads)   | 2 ms |  26.4 M  |  88.9  B |  
| YOLO12x  | 640×640 | 80 | 27.6 ms / 36.1 FPS (1 thread  ) <br/> 53.2 ms / 37.4 FPS (2 threads)   | 2 ms |  59.1 M  |  199.0 B |  



### Performance Test Instructions

1. The performance data tested here are all for models with YUV420SP (nv12) input. There is no significant difference in performance data for models with NCHWRGB input.
2. BPU Latency and BPU Throughput.
- Single-thread latency refers to the delay of processing a single frame by a single thread on a single BPU core, representing the ideal scenario for BPU inference task.
- Multi-thread frame rate means multiple threads simultaneously send tasks to the BPU, where each BPU core can handle tasks from multiple threads. Generally, in engineering projects, controlling the frame delay to be relatively small with 4 threads while fully utilizing the BPU up to 100% can achieve a good balance between throughput (FPS) and frame latency. The BPU of S100/S100P performs quite well, usually requiring only 2 threads to fully utilize the BPU, achieving outstanding frame latency and throughput.
- Data recorded in the table typically reaches a point where throughput does not significantly increase with the number of threads.
- BPU latency and BPU throughput were tested on the board using the following commands:
```bash
hrt_model_exec perf --thread_num 2 --model_file yolov8n_detect_bayese_640x640_nv12_modified.bin

python3 ../../../resource/tools/batch_perf/batch_perf.py --max 3 --file source/reference_hbm_models/
```

3. The test boards were in their optimal state.
- Optimal state for S100P: CPU consists of 6 × A78AE @ 2.0GHz with full-core Performance scheduling, BPU is 1 × Nash-m @ 1.5GHz, delivering 128TOPS at int8.
- Optimal state for S100: CPU consists of 6 × A78AE @ 1.5GHz with full-core Performance scheduling, BPU is 1 × Nash-e @ 1.0GHz, delivering 80TOPS at int8.

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```

## Benchmark - Accuracy

### RDK S100 / RDK S100P
Object Detection (COCO2017)
| Model | Pytorch | YUV420SP<br/>Python | YUV420SP<br/>C/C++ | NCHWRGB<br/>C/C++ |
|---------|---------|-------|---------|---------|
| YOLOv5nu | 0.275 | 0.259 (94.18%) | (%) | (%) |
| YOLOv5su | 0.362 | 0.349 (96.41%) | (%) | (%) |
| YOLOv5mu | 0.417 | 0.400 (95.92%) | (%) | (%) |
| YOLOv5lu | 0.449 | 0.436 (97.10%) | (%) | (%) |
| YOLOv5xu | 0.458 | 0.440 (96.07%) | (%) | (%) |
| YOLOv8n  | 0.306 | 0.291 (95.10%) | (%) | (%) |
| YOLOv8s  | 0.384 | 0.368 (95.83%) | (%) | (%) |
| YOLOv8m  | 0.433 | 0.417 (96.30%) | (%) | (%) |
| YOLOv8l  | 0.454 | 0.437 (96.26%) | (%) | (%) |
| YOLOv8x  | 0.465 | 0.446 (95.91%) | (%) | (%) |
| YOLO11n  | 0.323 | 0.303 (93.81%) | (%) | (%) |
| YOLO11s  | 0.394 | 0.375 (95.18%) | (%) | (%) |
| YOLO11m  | 0.437 | 0.416 (95.19%) | (%) | (%) |
| YOLO11l  | 0.452 | 0.429 (94.91%) | (%) | (%) |
| YOLO11x  | 0.466 | 0.442 (94.85%) | (%) | (%) |
| YOLO12n  | 0.334 | 0.311 (93.11%) | (%) | (%) |
| YOLO12s  | 0.397 | 0.381 (95.97%) | (%) | (%) |
| YOLO12m  | 0.444 | 0.421 (94.82%) | (%) | (%) |
| YOLO12l  | 0.454 | 0.430 (94.71%) | (%) | (%) |
| YOLO12x  | 0.466 | 0.439 (94.21%) | (%) | (%) |

### Accuracy Test Instructions

1. All accuracy data was calculated using Microsoft's unmodified `pycocotools` library, focusing on `Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]`.
2. All test data used the COCO2017 dataset's validation set of 5000 images, inferred directly on-device, saved as JSON files, and processed through third-party testing tools (`pycocotools`) with score thresholds set at 0.25 and NMS thresholds at 0.7.
3. Lower accuracy from `pycocotools` compared to Ultralytics' calculations is normal due to differences in area calculation methods. Our focus is on evaluating quantization-induced precision loss using consistent calculation methods.
4. Some accuracy loss occurs when converting NCHW-RGB888 input to YUV420SP(nv12) input for BPU models, mainly due to color space conversion. Incorporating this during training can mitigate such losses.
5. Slight discrepancies between Python and C/C++ interface accuracies arise from different handling of floating-point numbers during memcpy and conversions.
6. Test scripts can be found in the RDK Model Zoo eval section: [RDK Model Zoo Eval](https://github.com/D-Robotics/rdk_model_zoo/tree/main/demos/tools/eval_pycocotools)
7. This table reflects PTQ results using 50 images for calibration and compilation, simulating typical developer scenarios without fine-tuning or QAT, suitable for general validation needs but not indicative of maximum accuracy.


## Advanced Development

### Introduction to High Performance Computing Processes

![](source/imgs/ultralytics_YOLO_Detect_DataFlow.png)

In the standard processing flow, scores, classes, and xyxy coordinates for all 8400 bounding boxes (bbox) are fully calculated to compute the loss function based on ground truth (GT). However, during deployment, only qualified bounding boxes are needed, not requiring full computation of all 8400 bbox. The optimized processing flow mainly utilizes the monotonicity of the Sigmoid function to first screen and then compute. Meanwhile, by leveraging advanced indexing with Python's numpy, it also screens before computing parts like DFL and feature decoding, saving considerable computation, thus allowing post-processing on a CPU using numpy to be completed in just 5 milliseconds per frame with a single core and thread.

- **Classify Section, Dequantize Operation**
During model compilation, if all dequantization operators are removed, dequantization needs to be manually performed on the three output heads of the Classify section within post-processing. There are multiple ways to check the dequantization coefficients, including viewing the logs from `hb_mapper` or using the BPU inference interface API. Note that each C dimension has different dequantization coefficients; each head has 80 dequantization coefficients, which can be directly multiplied using numpy broadcasting. This dequantization is implemented in the bin model, so the output obtained is in float32.

- **Classify Section, ReduceMax Operation**
The ReduceMax operation finds the maximum value along a certain dimension of a tensor, used here to find the maximum score among the 80 scores of the 8400 Grid Cells. This operation targets the values of each Grid Cell's 80 categories, operating on the C dimension. It's important to note that this step provides the maximum value, not the index of the maximum value among the 80.
Given the monotonic nature of the Sigmoid activation function, the relationship between the sizes of the 80 scores before and after applying Sigmoid remains unchanged.
$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$
$$Sigmoid(x_1) > Sigmoid(x_2) \Leftrightarrow x_1 > x_2$$
Therefore, the position of the maximum value directly outputted by the bin model (after dequantization) represents the final maximum score position, and the maximum value outputted by the bin model, after being processed by Sigmoid, equals the maximum value of the original ONNX model.

- **Classify Section, Threshold (TopK) Operation**
This operation identifies qualifying Grid Cells among the 8400 Grid Cells, targeting the 8400 Grid Cells across the H and W dimensions. For convenience in programming and description, these dimensions are flattened, but there's no essential difference. Assuming the score of a certain category in a Grid Cell is denoted as \(x\), and the integer data after activation is \(y\), the thresholding process involves setting a threshold \(C\). A necessary and sufficient condition for a score to qualify is:
$$y=Sigmoid(x)=\frac{1}{1+e^{-x}}>C$$
From this, we derive that the condition for qualification is:
$$x > -ln\left(\frac{1}{C}-1\right)$$
This operation retrieves the indices and corresponding maximum values of qualifying Grid Cells, where the maximum value, after Sigmoid calculation, represents the score for that category.

- **Classify Section, GatherElements Operation and ArgMax Operation**
Using the indices obtained from the Threshold (TopK) operation, the GatherElements operation retrieves qualifying Grid Cells. The ArgMax operation then determines which of the 80 categories has the highest score, identifying the category of the qualifying Grid Cell.

- **Bounding Box Section, GatherElements Operation and Dequantize Operation**
Similarly, using the indices from the Threshold (TopK) operation, the GatherElements operation retrieves qualifying Grid Cells. Each C dimension has different dequantization coefficients; each head has 64 dequantization coefficients, which can be directly multiplied using numpy broadcasting to obtain bbox information of shape 1×64×k×1.

- **Bounding Box Section, DFL: SoftMax + Conv Operation**
Each Grid Cell uses four numbers to determine the position of its bounding box. The DFL structure provides 16 estimates for one edge of the bounding box based on the anchor's position, applies SoftMax to these estimates, and uses a convolution operation to calculate the expectation. This is central to Anchor-Free design, where each Grid Cell predicts only one bounding box. If the 16 numbers predicting an offset are denoted as \(l_p\) or \((t_p, t_p, b_p)\), where \(p = 0,1,...,15\), the offset calculation formula is:
$$\hat{l} = \sum_{p=0}^{15}{\frac{p·e^{l_p}}{S}}, S =\sum_{p=0}^{15}{e^{l_p}}$$

- **Bounding Box Section, Decode: dist2bbox(ltrb2xyxy) Operation**
This operation decodes each bounding box's ltrb description into an xyxy description. The ltrb represents distances from the top-left and bottom-right corners relative to the Grid Cell center, which are then converted back to absolute positions and scaled according to the feature layer's sampling ratio to yield the predicted xyxy coordinates.

For input images sized at 640, YOLOv8-Detect features three feature maps (\(i=1,2,3\)) with downsample ratios \(Stride(i)\) of 8, 16, and 32 respectively. This corresponds to feature map sizes of \(n_1=80\), \(n_2=40\), and \(n_3=20\), totaling 8400 Grid Cells responsible for predicting 8400 bounding boxes. 

The final detection results include the class (id), score, and location (xyxy).

### Environment and Project Preparation

Note: For any errors such as "No such file or directory", "No module named 'xxx'", "command not found", etc., please check carefully. Do not simply copy and run each command one by one. If you do not understand the modification process, please visit the developer community to start learning from YOLOv5.

- Download the ultralytics/ultralytics repository and configure the environment according to the official Ultralytics documentation.
```bash
git clone https://github.com/ultralytics/ultralytics.git
```
- Enter the local repository and download the official pre-trained weights. Here we use the YOLO11n-Detect model with 2.6 million parameters as an example.
```bash
cd ultralytics
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

### Model Training

- For model training, please refer to the official ultralytics documentation, which is maintained by ultralytics and of very high quality. There are also a great many reference materials on the Internet, and it is not difficult to obtain a model with pre-trained weights like the official one.

- Please note that during training, there is no need to modify any program or the forward method.

Ultralytics YOLO official documentation: https://docs.ultralytics.com/modes/train/


### Export to ONNX

- Uninstall yolo-related command-line commands so that modifications directly in the `./ultralytics/ultralytics` directory take effect.

```bash
$ conda list | grep ultralytics
$ pip list | grep ultralytics # or
# If exists, uninstall
$ conda uninstall ultralytics 
$ pip uninstall ultralytics   # or
```

If it's not straightforward, you can confirm the location of the `ultralytics` directory that needs to be modified using the following Python command:

```bash
>>> import ultralytics
>>> ultralytics.__path__
['/home/wuchao/miniconda3/envs/yolo/lib/python3.11/site-packages/ultralytics']
# or
['/home/wuchao/YOLO11/ultralytics_v11/ultralytics']
```

- Modify the Detect output head to separately output Bounding Box information and Classify information for three feature layers, resulting in a total of six output heads.

File path: `./ultralytics/ultralytics/nn/modules/head.py`, around line 58, replace the `forward` method of the `Detect` class with the following content.

Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for easier switching back during training.

```python
def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result

## If the order of output heads is reversed between bbox and cls, adjust the append order of cv2 and cv3 accordingly,
## then re-export the ONNX model and compile it into a hbm model.

def forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())
    return result
```

- Modify the optimized Attention module (YOLO11 Optional) .

File path: `ultralytics/nn/modules/block.py`, around line 868, replace the `forward` method of the `Attention` class with the following content. The main optimization points are removing some useless data movement operations and changing the Reduce dimension to C for better BPU compatibility, which currently doubles the throughput of the BPU without needing to retrain the model.
Note: It is suggested to keep the original `forward` method, e.g., rename it to `forward_`, for easier switching back during training.

```python
class Attention(nn.Module):
    def forward(self, x):  # original
        pass
    def forward(self, x):  # RDK
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
```

- Modify the optimized AAttn module (YOLO12 Optional)

File path: `ultralytics/nn/modules/block.py`, around line 1159, replace the `forward` method of the `AAttn` class with the following content. The main optimizations involve removing unnecessary data transfer operations and changing the Reduce dimension to C for better BPU compatibility without needing to retrain the model. This currently optimizes BPU throughput to over 60FPS, achieving real-time object detection. Future Bayes BPU will specifically optimize the Area Attention structure to achieve a throughput similar to YOLOv8's nearly 300FPS.
Note: It's recommended to keep the original `forward` method, e.g., rename it to `forward_`, to switch back during training if necessary.

```python
class AAttn(nn.Module):
    def forward(self, x):  # original
        pass
    def forward(self, x):  # RDK
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
        max_attn = attn.max(dim=1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        sum_attn = exp_attn.sum(dim=1, keepdim=True)
        attn = exp_attn / sum_attn
        attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
        x = (v @ attn.transpose(-2, -1))
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)
        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.pe(v)
        x = self.proj(x)
        return x
```

- Run the following Python script to export ONNX 

If there is an error saying No module named onnxsim, simply install it. Note, if the generated ONNX model shows too high IR version, set simplify=False. Both settings do not affect the final bin model but improve the readability of the ONNX model in Netron when enabled.

```python
from ultralytics import YOLO
YOLO('yolov11n.pt').export(imgsz=640, format='onnx', simplify=False, opset=19)
```

### Prepare Calibration Data

Refer to the minimalist calibration data preparation script provided in the RDK Model Zoo S: `https://github.com/D-Robotics/rdk_model_zoo_s/blob/s100/resource/tools/generate_calibration_data/generate_cal_data.py` for preparing calibration data.

### Confirm Removal of Dequantization Node Names

Use the Netron visualization tool: `https://netron.app/`

By viewing the ONNX model through Netron, confirm the names of nodes to remove; a handy rule is to remove nodes that contain '64'. Here, 64 = 4 * REG, where REG = 16. Note, the names exported by different versions of Ultralytics may vary, so do not directly apply them.

![](source/imgs/netron_conv_example.jpeg)

See sizes of [1, 80, 80, 64], [1, 40, 40, 64], [1, 20, 20, 64] for their respective names /model.23/cv2.0/cv2.0.2/Conv, /model.23/cv2.1/cv2.1.2/Conv, /model.23/cv2.2/cv2.2.2/Conv.

Corresponding entries should be added to your YAML configuration file.

```yaml
model_parameters:
  onnx_model: 'ultralytcs_YOLO.onnx'
  march: nash-e  # S100: nash-e, S100P: nash-m.
  layer_out_dump: False
  working_dir: 'ultralytcs_YOLO_output'
  output_model_file_prefix: 'ultralytcs_YOLO'
  remove_node_name: "/model.23/cv2.0/cv2.0.2/Conv;/model.23/cv2.1/cv2.1.2/Conv;/model.23/cv2.2/cv2.2.2/Conv;"
  # Reference remove_node_name
  # YOLOv5u: /model.24/cv2.0/cv2.0.2/Conv;/model.24/cv2.1/cv2.1.2/Conv;/model.24/cv2.2/cv2.2.2/Conv;
  # YOLOv8: /model.22/cv2.0/cv2.0.2/Conv;/model.22/cv2.1/cv2.1.2/Conv;/model.22/cv2.2/cv2.2.2/Conv;
  # YOLO11: /model.23/cv2.0/cv2.0.2/Conv;/model.23/cv2.1/cv2.1.2/Conv;/model.23/cv2.2/cv2.2.2/Conv;
  # YOLO12: /model.21/cv2.0/cv2.0.2/Conv;/model.21/cv2.1/cv2.1.2/Conv;/model.21/cv2.2/cv2.2.2/Conv;

```

### Model Compilation

```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### Exception Handling

If the model outputs differ from the reference models in the Model Zoo, this could be due to incorrect node names removed. You can confirm this by checking the bc model information.

```bash
# Quickly generate a bc model
hb_compile --fast-perf --march nash-e --skip compile --model yolo11n.onnx
# View bc model output node information
hb_model_info yolo11n_quantized_model.bc

```
This provides information about the nodes and helps in debugging discrepancies between your model and the reference models provided in the Model Zoo.



### Model compile

```bash
(bpu_docker) $ hb_compile --config config.yaml
```

### Exception Handling

If the output of the Model is inconsistent with the reference model of Model Zoo, the reason might be that the names of the removed nodes are incorrect. This can be confirmed by checking the information of the bc model.

```bash
# Generate a bc model quickly
hb_compile --fast-perf --march nash-e --skip compile --model yolo11n.onnx
# View the output node information of the bc model
hb_model_info yolo11n_quantized_model.bc
```

The following information can be accessed

```bash
INFO ############# Removable node info #############
INFO Node Name                                          Node Type
INFO -------------------------------------------------- ----------
INFO /model.23/cv3.0/cv3.0.2/Conv                       Dequantize
INFO /model.23/cv2.0/cv2.0.2/Conv                       Dequantize
INFO /model.23/cv4.0/cv4.0.2/Conv                       Dequantize
INFO /model.23/cv3.1/cv3.1.2/Conv                       Dequantize
INFO /model.23/cv2.1/cv2.1.2/Conv                       Dequantize
INFO /model.23/cv4.1/cv4.1.2/Conv                       Dequantize
INFO /model.23/cv3.2/cv3.2.2/Conv                       Dequantize
INFO /model.23/cv2.2/cv2.2.2/Conv                       Dequantize
INFO /model.23/cv4.2/cv4.2.2/Conv                       Dequantize
INFO /model.23/proto/cv3/act/Mul_output_0_HzCalibration Dequantize
```

Model Zoo provides compilation logs, bc Model information logs and hbm model logs for comparing the differences between the models you obtain yourself and the reference models of Model Zoo.

```bash
./samples/Vision/ultralytics_YOLO_Detect/source/reference_logs/
|-- hb_compile_yolo11n.txt
|-- hb_compile_yolo12n.txt
|-- hb_compile_yolov5nu.txt
|-- hb_compile_yolov8n.txt
|-- hb_model_info_yolo11n.txt
|-- hb_model_info_yolo12n.txt
|-- hb_model_info_yolov5nu.txt
|-- hb_model_info_yolov8n.txt
|-- hrt_model_exec_model_info_yolo11n.txt
|-- hrt_model_exec_model_info_yolo12n.txt
|-- hrt_model_exec_model_info_yolov5nu.txt
`-- hrt_model_exec_model_info_yolov8n.txt
```

## References

[ultralytics docs](https://docs.ultralytics.com/)
