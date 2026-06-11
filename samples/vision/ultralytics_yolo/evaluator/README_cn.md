[English](./README.md) | 简体中文

# Ultralytics YOLO 模型评估说明

本目录记录 Ultralytics YOLO 系列模型在 RDK X5、RDK S100、RDK S100P 等平台上的性能与精度评估方法、测试命令和参考数据。评估数据覆盖 COCO2017、ImageNet-1k 等公开数据集，检测、分割、姿态估计和分类任务的指标说明见下文。
## BenchMark Instructions

### Performance Test Instructions
1. `Device`列表示测试的平台, S100P表示RDK S100P, S100表示RDK S100, X5表示RDK X5 (Module).
2. `Model`列表示测试的模型, 与本文`Support Models`章节所列模型是对应关系.
3. `Size(Pixels)`列表示的是模型的算法分辨率, 是导出ONNX模型的输入分辨率, 其他分辨率的图像一般是经过前处理缩放到此分辨率, 再送入网络推理.
4. `Classes`列表示的是模型的检测目标数量, 这里使用的都是Ultralytics YOLO基于COCO2017数据集或者ImageNet-1k数据集训练出来的权重, 类别数量与对应的数据集的类别数量是一致的.
5. `BPU Task Latency / BPU Throughput (Threads)`列列举了BPU延迟与BPU吞吐量的情况.
 - 单线程延迟为单帧,单线程,单BPU核心的延迟,BPU推理一个任务最理想的情况.
 - 多线程帧率为多个线程同时向BPU塞任务, 每个BPU核心可以处理多个线程的任务, 一般工程中2个线程可以控制单帧延迟较小,同时吃满所有BPU到100%,在吞吐量(FPS)和帧延迟间得到一个较好的平衡.
 - 表格中一般记录到吞吐量不再随线程数明显增加的数据.
 - BPU延迟和BPU吞吐量使用以下命令在板端实验, hrt_model_exec工具由OE包提供, 其源码在OE包的`package/board/hrt_model_exec/src`目录下.
```bash
hrt_model_exec perf --thread_num 2 --model_file < model.bin / model.hbm >
```
 - 由于实验的条件不同, 复现的的结果可能不同, 这里统一参照本文的`Platform Details`中的设备状态, 在平台状态最佳时进行实验.
 - hrt_model_exec工具的性能实验中, 充分考虑了缓存预热, 多线程程序设计等性能测试的内容, 测量的时间为用户程序向BPU提交BPU任务到等待BPU任务结束的时间。
 - 在流式数据推理时, 输入输出内存可以开辟一次, 反复使用, 请不要将开辟和回收内存的时间纳入推理时间, 也不要在流式数据推理中反复开辟和回收内存, 这是不科学的程序设计方法.
6. `CPU Latency (Single Core)`指的是后处理时间, 目前对后处理有做性能优化, 后处理时间与有效目标的个数正相关, 这里的后处理时间一般是目标图片中有效目标的个数小于100时的性能数据. Python和C/C++的后处理时间会有一点差距, 但是由于Python后处理程序基本也是numpy深度优化了, 所以两者差距并不是很大.
7. `params(M)`和`FLOPs(B)`是原始浮点模型的参数量和计算量, 使用Ultralytics YOLO软件包在加载完pt模型后, 使用YOLO.export方法时日志中打印的浮点模型参数量和计算量信息. 由于最终生成的BPU定点模型的参数量和计算量与模型结构优化, 图优化, 编译器优化有关系, 与浮点模型的参数量和计算量正相关但是不一定成正比例, 所以这里统一记录浮点计算量为参考.


### Accuracy Test Instructions
1. `Device`列和`Model`列含义与`Performance Test Instructions`章节的含义相同.
2. 精度数据使用微软官方的无修改的`pycocotools`库进行计算, 目标检测(`Obeject Detection`)任务的测评模式为iouType="bbox", 和实例分割(`Instance Segmentation`)任务的测评模式为`iouType="bbox"`和`iouType="segm"`, 人体关键点估计的测评模式为`iouType="keypoints"`.
 - `Accuracy bbox-all mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]`.
 - `Accuracy bbox-small mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]`.
 - `Accuracy bbox-medium mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]`.
 - `Accuracy bbox-large mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]`.
 - `Accuracy mask-all mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]`.
 - `Accuracy mask-small mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]`.
 - `Accuracy mask-medium mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]`.
 - `Accuracy mask-large mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]`.
 - `Accuracy pose-all mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]`.
 - `Accuracy pose-medium mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]`.
 - `Accuracy pose-large mAP@.50:.95` 取自 `Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]`.
3. AP 更关注“质量”：既要找到目标(recall), 又要框得准, 类别对(precision), AR 更关注“数量”: 只要框住就算，不惩罚误检. 一个模型可以有高 AR 但低 AP, 比如疯狂输出大量低质量框, 找得全但不准. 也可以有高 AP 但低 AR, 比如只输出高置信度结果, 很准但漏很多. 这里取的均为AP指标来衡量模型的精度.
4. 测试数据均使用`COCO2017`数据集的val验证集的5000张照片, 在板端直接推理, dump保存为json文件, 送入第三方测试工具`pycocotools`库进行计算, 分数的阈值为0.25, nms的阈值为0.7.
5. pycocotools计算的精度比ultralytics计算的精度会低一些是正常现象, 主要原因是在计算AP曲线下面积时, pycocotools是取矩形面积, ultralytics是取梯形面积, 我们主要是关注同样的一套计算方式去测试定点模型和浮点模型的精度, 从而来评估量化过程中的精度损失.
6. 分类任务使用的数据集为`ImageNet-1k`, 使用TOP1和TOP5两个指标来评估量化过程中的精度损失.
7. BPU模型在量化NCHW-RGB888输入转换为YUV420SP(nv12)输入后, 也会有一部分精度损失, 这是由于色彩空间转化导致的, 在训练时加入这种色彩空间转化的损失可以避免这种精度损失.
8. Python接口和C/C++接口的精度结果有细微差异, 主要在于Python和C/C++的一些数据结构进行memcpy和转化的过程中, 对浮点数的处理方式不同, 导致的细微差异.
9. 本表格是使用PTQ(训练后量化)使用50张图片进行校准和编译的结果, 用于模拟普通开发者第一次直接编译的精度情况, 并没有进行精度调优或者QAT(量化感知训练), 满足常规使用验证需求, 不代表精度上限.

## Performance
### RDK S100P
#### Obeject Detection
| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO12n Detect  | 640×640        |        80 | 1.88 ms / 513.70 FPS (1 thread ) <br/> 3.07 ms / 634.97 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100P    | YOLO12s Detect  | 640×640        |        80 | 3.10 ms / 315.83 FPS (1 thread ) <br/> 5.50 ms / 357.85 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| S100P    | YOLO12m Detect  | 640×640        |        80 | 6.47 ms / 152.80 FPS (1 thread ) <br/> 12.18 ms / 162.62 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| S100P    | YOLO12l Detect  | 640×640        |        80 | 10.23 ms / 97.01 FPS (1 thread ) <br/> 19.67 ms / 101.04 FPS (2 threads) | 2.0 ms                         | 26.4 M      | 88.9 M     |
| S100P    | YOLO12x Detect  | 640×640        |        80 | 17.05 ms / 58.34 FPS (1 thread ) <br/> 33.21 ms / 59.92 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| S100P    | YOLO11n Detect  | 640×640        |        80 | 1.16 ms / 816.50 FPS (1 thread ) <br/> 1.66 ms / 1155.65 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 6.5 M      |
| S100P    | YOLO11s Detect  | 640×640        |        80 | 1.81 ms / 533.50 FPS (1 thread ) <br/> 2.98 ms / 656.31 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| S100P    | YOLO11m Detect  | 640×640        |        80 | 3.90 ms / 252.02 FPS (1 thread ) <br/> 7.10 ms / 278.36 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 68.0 M     |
| S100P    | YOLO11l Detect  | 640×640        |        80 | 4.73 ms / 208.61 FPS (1 thread ) <br/> 8.75 ms / 225.99 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 86.9 M     |
| S100P    | YOLO11x Detect  | 640×640        |        80 | 8.84 ms / 112.05 FPS (1 thread ) <br/> 16.92 ms / 117.39 FPS (2 threads) | 2.0 ms                         | 56.9 M      | 194.9 M    |
| S100P    | YOLOv10n Detect | 640×640        |        80 | 1.12 ms / 837.97 FPS (1 thread ) <br/> 1.58 ms / 1211.72 FPS (2 threads) | 2.0 ms                         | 2.3 M       | 6.7 M      |
| S100P    | YOLOv10s Detect | 640×640        |        80 | 1.75 ms / 548.80 FPS (1 thread ) <br/> 2.81 ms / 692.74 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| S100P    | YOLOv10m Detect | 640×640        |        80 | 3.06 ms / 319.65 FPS (1 thread ) <br/> 5.45 ms / 361.32 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| S100P    | YOLOv10b Detect | 640×640        |        80 | 4.30 ms / 228.16 FPS (1 thread ) <br/> 7.85 ms / 250.93 FPS (2 threads)  | 2.0 ms                         | 19.1 M      | 92.0 M     |
| S100P    | YOLOv10l Detect | 640×640        |        80 | 5.42 ms / 181.96 FPS (1 thread ) <br/> 10.10 ms / 196.04 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| S100P    | YOLOv10x Detect | 640×640        |        80 | 7.33 ms / 135.18 FPS (1 thread ) <br/> 13.90 ms / 142.81 FPS (2 threads) | 2.0 ms                         | 29.5 M      | 160.4 M    |
| S100P    | YOLOv9t Detect  | 640×640        |        80 | 1.29 ms / 736.75 FPS (1 thread ) <br/> 1.90 ms / 1013.70 FPS (2 threads) | 2.0 ms                         | 2.1 M       | 8.2 M      |
| S100P    | YOLOv9s Detect  | 640×640        |        80 | 1.93 ms / 497.53 FPS (1 thread ) <br/> 3.19 ms / 611.75 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| S100P    | YOLOv9m Detect  | 640×640        |        80 | 3.77 ms / 260.19 FPS (1 thread ) <br/> 6.83 ms / 288.82 FPS (2 threads)  | 2.0 ms                         | 20.1 M      | 76.8 M     |
| S100P    | YOLOv9c Detect  | 640×640        |        80 | 4.76 ms / 206.90 FPS (1 thread ) <br/> 8.77 ms / 225.46 FPS (2 threads)  | 2.0 ms                         | 25.3 M      | 102.7 M    |
| S100P    | YOLOv9e Detect  | 640×640        |        80 | 12.27 ms / 81.00 FPS (1 thread ) <br/> 23.73 ms / 83.75 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| S100P    | YOLOv8n Detect  | 640×640        |        80 | 1.10 ms / 851.31 FPS (1 thread ) <br/> 1.52 ms / 1258.50 FPS (2 threads) | 2.0 ms                         | 3.2 M       | 8.7 M      |
| S100P    | YOLOv8s Detect  | 640×640        |        80 | 1.83 ms / 524.95 FPS (1 thread ) <br/> 2.95 ms / 660.43 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| S100P    | YOLOv8m Detect  | 640×640        |        80 | 3.43 ms / 285.34 FPS (1 thread ) <br/> 6.14 ms / 320.93 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| S100P    | YOLOv8l Detect  | 640×640        |        80 | 6.72 ms / 147.19 FPS (1 thread ) <br/> 12.67 ms / 156.40 FPS (2 threads) | 2.0 ms                         | 43.7 M      | 165.2 M    |
| S100P    | YOLOv8x Detect  | 640×640        |        80 | 10.44 ms / 95.08 FPS (1 thread ) <br/> 20.11 ms / 98.81 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| S100P    | YOLOv5nu Detect | 640×640        |        80 | 0.99 ms / 954.28 FPS (1 thread ) <br/> 1.34 ms / 1418.24 FPS (2 threads) | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100P    | YOLOv5su Detect | 640×640        |        80 | 1.60 ms / 602.38 FPS (1 thread ) <br/> 2.56 ms / 763.66 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| S100P    | YOLOv5mu Detect | 640×640        |        80 | 3.06 ms / 319.05 FPS (1 thread ) <br/> 5.43 ms / 363.38 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| S100P    | YOLOv5lu Detect | 640×640        |        80 | 6.04 ms / 163.65 FPS (1 thread ) <br/> 11.36 ms / 174.46 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| S100P    | YOLOv5xu Detect | 640×640        |        80 | 10.74 ms / 92.40 FPS (1 thread ) <br/> 20.67 ms / 96.10 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |
#### Instance Segmentation
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n Seg | 640×640        |        80 | 1.45 ms / 647.24 FPS (1 thread ) <br/> 2.14 ms / 883.83 FPS (2 threads)  | 5.0 ms                         | 2.9 M       | 10.4 M     |
| S100P    | YOLO11s Seg | 640×640        |        80 | 2.31 ms / 413.73 FPS (1 thread ) <br/> 3.88 ms / 501.74 FPS (2 threads)  | 5.0 ms                         | 10.1 M      | 35.5 M     |
| S100P    | YOLO11m Seg | 640×640        |        80 | 5.36 ms / 182.99 FPS (1 thread ) <br/> 9.89 ms / 199.15 FPS (2 threads)  | 5.0 ms                         | 22.4 M      | 123.3 M    |
| S100P    | YOLO11l Seg | 640×640        |        80 | 6.20 ms / 158.60 FPS (1 thread ) <br/> 11.57 ms / 170.73 FPS (2 threads) | 5.0 ms                         | 27.6 M      | 142.2 M    |
| S100P    | YOLO11x Seg | 640×640        |        80 | 11.89 ms / 83.30 FPS (1 thread ) <br/> 22.83 ms / 86.92 FPS (2 threads)  | 5.0 ms                         | 62.1 M      | 319.0 M    |
| S100P    | YOLOv9c Seg | 640×640        |        80 | 6.29 ms / 156.58 FPS (1 thread ) <br/> 11.74 ms / 168.32 FPS (2 threads) | 5.0 ms                         | 27.7 M      | 158.0 M    |
| S100P    | YOLOv9e Seg | 640×640        |        80 | 14.20 ms / 69.78 FPS (1 thread ) <br/> 27.42 ms / 72.35 FPS (2 threads)  | 5.0 ms                         | 59.7 M      | 244.8 M    |
| S100P    | YOLOv8n Seg | 640×640        |        80 | 1.39 ms / 666.05 FPS (1 thread ) <br/> 2.01 ms / 946.83 FPS (2 threads)  | 5.0 ms                         | 3.4 M       | 12.6 M     |
| S100P    | YOLOv8s Seg | 640×640        |        80 | 2.32 ms / 411.56 FPS (1 thread ) <br/> 3.86 ms / 502.69 FPS (2 threads)  | 5.0 ms                         | 11.8 M      | 42.6 M     |
| S100P    | YOLOv8m Seg | 640×640        |        80 | 4.37 ms / 223.73 FPS (1 thread ) <br/> 7.95 ms / 247.14 FPS (2 threads)  | 5.0 ms                         | 27.3 M      | 100.2 M    |
| S100P    | YOLOv8l Seg | 640×640        |        80 | 8.20 ms / 120.19 FPS (1 thread ) <br/> 15.46 ms / 127.84 FPS (2 threads) | 5.0 ms                         | 46.0 M      | 220.5 M    |
| S100P    | YOLOv8x Seg | 640×640        |        80 | 13.01 ms / 76.23 FPS (1 thread ) <br/> 25.11 ms / 79.06 FPS (2 threads)  | 5.0 ms                         | 71.8 M      | 344.1 M    |
#### Pose Estimation
| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n Pose | 640×640        |        80 | 1.23 ms / 770.10 FPS (1 thread ) <br/> 1.74 ms / 1097.27 FPS (2 threads) | 1.0 ms                         | 2.9 M       | 7.6 M      |
| S100P    | YOLO11s Pose | 640×640        |        80 | 1.92 ms / 501.66 FPS (1 thread ) <br/> 3.11 ms / 627.29 FPS (2 threads)  | 1.0 ms                         | 9.9 M       | 23.2 M     |
| S100P    | YOLO11m Pose | 640×640        |        80 | 4.04 ms / 241.82 FPS (1 thread ) <br/> 7.32 ms / 269.96 FPS (2 threads)  | 1.0 ms                         | 20.9 M      | 71.7 M     |
| S100P    | YOLO11l Pose | 640×640        |        80 | 4.87 ms / 202.29 FPS (1 thread ) <br/> 8.99 ms / 220.09 FPS (2 threads)  | 1.0 ms                         | 26.2 M      | 90.7 M     |
| S100P    | YOLO11x Pose | 640×640        |        80 | 9.15 ms / 108.13 FPS (1 thread ) <br/> 17.45 ms / 113.64 FPS (2 threads) | 1.0 ms                         | 58.8 M      | 203.3 M    |
| S100P    | YOLOv8n Pose | 640×640        |        80 | 1.14 ms / 822.46 FPS (1 thread ) <br/> 1.58 ms / 1206.58 FPS (2 threads) | 1.0 ms                         | 3.3 M       | 9.2 M      |
| S100P    | YOLOv8s Pose | 640×640        |        80 | 1.97 ms / 486.85 FPS (1 thread ) <br/> 3.23 ms / 606.41 FPS (2 threads)  | 1.0 ms                         | 11.6 M      | 30.2 M     |
| S100P    | YOLOv8m Pose | 640×640        |        80 | 3.65 ms / 267.74 FPS (1 thread ) <br/> 6.54 ms / 301.30 FPS (2 threads)  | 1.0 ms                         | 26.4 M      | 81.0 M     |
| S100P    | YOLOv8l Pose | 640×640        |        80 | 6.92 ms / 142.52 FPS (1 thread ) <br/> 12.99 ms / 152.18 FPS (2 threads) | 1.0 ms                         | 44.4 M      | 168.6 M    |
| S100P    | YOLOv8x Pose | 640×640        |        80 | 10.67 ms / 92.89 FPS (1 thread ) <br/> 20.48 ms / 96.97 FPS (2 threads)  | 1.0 ms                         | 69.4 M      | 263.2 M    |
#### Image Classification
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                                                                   | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|-------------------------------------------------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100P    | YOLO11n CLS | 640×640        |        80 | 0.40 ms / 2368.83 FPS (1 thread ) <br/> 0.46 ms / 4151.62 FPS (2 threads) <br/> 0.56 ms / 5164.09 FPS (3 threads) | 0.5 ms                         | 2.8 M       | 4.2 M      |
| S100P    | YOLO11s CLS | 640×640        |        80 | 0.52 ms / 1843.47 FPS (1 thread ) <br/> 0.61 ms / 3128.03 FPS (2 threads) <br/> 0.81 ms / 3593.24 FPS (3 threads) | 0.5 ms                         | 6.7 M       | 13.0 M     |
| S100P    | YOLO11m CLS | 640×640        |        80 | 0.78 ms / 1248.81 FPS (1 thread ) <br/> 1.01 ms / 1935.47 FPS (2 threads)                                         | 0.5 ms                         | 11.6 M      | 40.3 M     |
| S100P    | YOLO11l CLS | 640×640        |        80 | 0.90 ms / 1088.57 FPS (1 thread ) <br/> 1.27 ms / 1544.42 FPS (2 threads)                                         | 0.5 ms                         | 14.1 M      | 50.4 M     |
| S100P    | YOLO11x CLS | 640×640        |        80 | 1.45 ms / 676.30 FPS (1 thread ) <br/> 2.34 ms / 844.07 FPS (2 threads)                                           | 0.5 ms                         | 29.6 M      | 111.3 M    |
| S100P    | YOLOv8n CLS | 640×640        |        80 | 0.38 ms / 2470.91 FPS (1 thread ) <br/> 0.45 ms / 4304.69 FPS (2 threads) <br/> 0.55 ms / 5272.31 FPS (3 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| S100P    | YOLOv8s CLS | 640×640        |        80 | 0.49 ms / 1953.98 FPS (1 thread ) <br/> 0.57 ms / 3364.17 FPS (2 threads) <br/> 0.70 ms / 4109.05 FPS (3 threads) | 0.5 ms                         | 6.4 M       | 13.5 M     |
| S100P    | YOLOv8m CLS | 640×640        |        80 | 0.80 ms / 1218.07 FPS (1 thread ) <br/> 1.05 ms / 1876.12 FPS (2 threads)                                         | 0.5 ms                         | 17.0 M      | 42.7 M     |
| S100P    | YOLOv8l CLS | 640×640        |        80 | 1.44 ms / 683.65 FPS (1 thread ) <br/> 2.34 ms / 842.80 FPS (2 threads)                                           | 0.5 ms                         | 37.5 M      | 99.7 M     |
| S100P    | YOLOv8x CLS | 640×640        |        80 | 2.09 ms / 470.34 FPS (1 thread ) <br/> 3.55 ms / 559.63 FPS (2 threads)                                           | 0.5 ms                         | 57.4 M      | 154.8 M    |
## Accuracy
#### Obeject Detection
| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO12n Detect  | 0.338 / 0.313 (92.4 %)                                  | 0.128 / 0.095 (74.0 %)                                    | 0.374 / 0.342 (91.4 %)                                     | 0.524 / 0.515 (98.3 %)                                    |
| S100P    | YOLO12s Detect  | 0.403 / 0.380 (94.2 %)                                  | 0.201 / 0.152 (75.5 %)                                    | 0.450 / 0.432 (95.9 %)                                     | 0.602 / 0.581 (96.5 %)                                    |
| S100P    | YOLO12m Detect  | 0.452 / 0.423 (93.7 %)                                  | 0.251 / 0.204 (81.3 %)                                    | 0.509 / 0.489 (96.0 %)                                     | 0.638 / 0.616 (96.5 %)                                    |
| S100P    | YOLO12l Detect  | 0.463 / 0.429 (92.8 %)                                  | 0.268 / 0.211 (78.6 %)                                    | 0.522 / 0.492 (94.3 %)                                     | 0.646 / 0.630 (97.7 %)                                    |
| S100P    | YOLO12x Detect  | 0.475 / 0.440 (92.7 %)                                  | 0.276 / 0.222 (80.3 %)                                    | 0.536 / 0.509 (94.9 %)                                     | 0.659 / 0.627 (95.1 %)                                    |
| S100P    | YOLO11n Detect  | 0.327 / 0.306 (93.9 %)                                  | 0.130 / 0.104 (80.0 %)                                    | 0.357 / 0.340 (95.2 %)                                     | 0.511 / 0.500 (97.8 %)                                    |
| S100P    | YOLO11s Detect  | 0.400 / 0.380 (95.0 %)                                  | 0.198 / 0.166 (83.9 %)                                    | 0.445 / 0.427 (96.1 %)                                     | 0.587 / 0.579 (98.6 %)                                    |
| S100P    | YOLO11m Detect  | 0.444 / 0.417 (94.0 %)                                  | 0.247 / 0.214 (87.0 %)                                    | 0.497 / 0.478 (96.1 %)                                     | 0.627 / 0.599 (95.6 %)                                    |
| S100P    | YOLO11l Detect  | 0.460 / 0.434 (94.5 %)                                  | 0.267 / 0.227 (85.2 %)                                    | 0.520 / 0.498 (95.9 %)                                     | 0.638 / 0.611 (95.8 %)                                    |
| S100P    | YOLO11x Detect  | 0.474 / 0.446 (94.0 %)                                  | 0.283 / 0.240 (84.7 %)                                    | 0.529 / 0.506 (95.6 %)                                     | 0.652 / 0.627 (96.1 %)                                    |
| S100P    | YOLOv10n Detect | 0.303 / 0.276 (91.3 %)                                  | 0.099 / 0.064 (64.7 %)                                    | 0.330 / 0.302 (91.5 %)                                     | 0.478 / 0.460 (96.2 %)                                    |
| S100P    | YOLOv10s Detect | 0.386 / 0.354 (91.6 %)                                  | 0.175 / 0.126 (72.2 %)                                    | 0.434 / 0.402 (92.5 %)                                     | 0.574 / 0.527 (91.7 %)                                    |
| S100P    | YOLOv10m Detect | 0.425 / 0.368 (86.7 %)                                  | 0.221 / 0.179 (80.9 %)                                    | 0.481 / 0.431 (89.6 %)                                     | 0.603 / 0.472 (78.2 %)                                    |
| S100P    | YOLOv10b Detect | 0.443 / 0.382 (86.3 %)                                  | 0.242 / 0.194 (80.2 %)                                    | 0.498 / 0.437 (87.7 %)                                     | 0.618 / 0.480 (77.7 %)                                    |
| S100P    | YOLOv10l Detect | 0.445 / 0.372 (83.6 %)                                  | 0.258 / 0.202 (78.5 %)                                    | 0.498 / 0.435 (87.4 %)                                     | 0.626 / 0.463 (74.0 %)                                    |
| S100P    | YOLOv10x Detect | 0.459 / 0.409 (89.2 %)                                  | 0.258 / 0.212 (82.1 %)                                    | 0.518 / 0.475 (91.8 %)                                     | 0.639 / 0.535 (83.6 %)                                    |
| S100P    | YOLOv9t Detect  | 0.313 / 0.301 (96.2 %)                                  | 0.113 / 0.105 (93.6 %)                                    | 0.338 / 0.325 (96.3 %)                                     | 0.483 / 0.461 (95.5 %)                                    |
| S100P    | YOLOv9s Detect  | 0.400 / 0.383 (95.8 %)                                  | 0.191 / 0.165 (86.3 %)                                    | 0.444 / 0.431 (97.0 %)                                     | 0.583 / 0.560 (96.1 %)                                    |
| S100P    | YOLOv9m Detect  | 0.449 / 0.432 (96.1 %)                                  | 0.253 / 0.231 (91.2 %)                                    | 0.504 / 0.487 (96.5 %)                                     | 0.617 / 0.602 (97.5 %)                                    |
| S100P    | YOLOv9c Detect  | 0.461 / 0.446 (96.8 %)                                  | 0.269 / 0.250 (93.2 %)                                    | 0.512 / 0.499 (97.4 %)                                     | 0.640 / 0.618 (96.6 %)                                    |
| S100P    | YOLOv9e Detect  | 0.481 / 0.465 (96.6 %)                                  | 0.298 / 0.270 (90.9 %)                                    | 0.538 / 0.520 (96.7 %)                                     | 0.662 / 0.647 (97.7 %)                                    |
| S100P    | YOLOv8n Detect  | 0.309 / 0.292 (94.4 %)                                  | 0.113 / 0.098 (87.2 %)                                    | 0.338 / 0.321 (94.9 %)                                     | 0.473 / 0.457 (96.7 %)                                    |
| S100P    | YOLOv8s Detect  | 0.391 / 0.373 (95.4 %)                                  | 0.195 / 0.166 (85.1 %)                                    | 0.437 / 0.425 (97.3 %)                                     | 0.566 / 0.558 (98.6 %)                                    |
| S100P    | YOLOv8m Detect  | 0.441 / 0.420 (95.4 %)                                  | 0.249 / 0.213 (85.6 %)                                    | 0.494 / 0.478 (96.7 %)                                     | 0.618 / 0.612 (99.1 %)                                    |
| S100P    | YOLOv8l Detect  | 0.461 / 0.442 (95.8 %)                                  | 0.271 / 0.241 (88.9 %)                                    | 0.516 / 0.499 (96.6 %)                                     | 0.651 / 0.628 (96.4 %)                                    |
| S100P    | YOLOv8x Detect  | 0.474 / 0.448 (94.6 %)                                  | 0.280 / 0.245 (87.6 %)                                    | 0.527 / 0.504 (95.7 %)                                     | 0.658 / 0.640 (97.2 %)                                    |
| S100P    | YOLOv5nu Detect | 0.278 / 0.264 (94.7 %)                                  | 0.093 / 0.080 (85.5 %)                                    | 0.309 / 0.293 (94.8 %)                                     | 0.417 / 0.406 (97.5 %)                                    |
| S100P    | YOLOv5su Detect | 0.367 / 0.349 (95.2 %)                                  | 0.169 / 0.141 (83.3 %)                                    | 0.416 / 0.398 (95.8 %)                                     | 0.530 / 0.524 (98.9 %)                                    |
| S100P    | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.194 (86.0 %)                                    | 0.477 / 0.467 (98.0 %)                                     | 0.603 / 0.592 (98.2 %)                                    |
| S100P    | YOLOv5lu Detect | 0.458 / 0.436 (95.1 %)                                  | 0.260 / 0.215 (82.9 %)                                    | 0.516 / 0.500 (96.8 %)                                     | 0.641 / 0.631 (98.4 %)                                    |
| S100P    | YOLOv5xu Detect | 0.466 / 0.445 (95.5 %)                                  | 0.281 / 0.239 (85.0 %)                                    | 0.523 / 0.506 (96.7 %)                                     | 0.645 / 0.638 (99.0 %)                                    |
#### Instance Segmentation
##### bbox
| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Seg | 0.322 / 0.294 (91.4 %)                                  | 0.113 / 0.081 (71.8 %)                                    | 0.352 / 0.324 (92.1 %)                                     | 0.502 / 0.490 (97.6 %)                                    |
| S100P    | YOLO11s Seg | 0.394 / 0.372 (94.4 %)                                  | 0.184 / 0.149 (81.2 %)                                    | 0.442 / 0.424 (96.0 %)                                     | 0.582 / 0.577 (99.1 %)                                    |
| S100P    | YOLO11m Seg | 0.443 / 0.414 (93.3 %)                                  | 0.246 / 0.208 (84.3 %)                                    | 0.497 / 0.473 (95.2 %)                                     | 0.627 / 0.599 (95.6 %)                                    |
| S100P    | YOLO11l Seg | 0.460 / 0.430 (93.5 %)                                  | 0.267 / 0.220 (82.5 %)                                    | 0.520 / 0.493 (94.9 %)                                     | 0.638 / 0.610 (95.6 %)                                    |
| S100P    | YOLO11x Seg | 0.474 / 0.441 (93.0 %)                                  | 0.283 / 0.231 (81.7 %)                                    | 0.529 / 0.501 (94.6 %)                                     | 0.652 / 0.625 (95.8 %)                                    |
| S100P    | YOLOv9c Seg | 0.453 / 0.422 (93.0 %)                                  | 0.254 / 0.206 (81.2 %)                                    | 0.508 / 0.483 (94.9 %)                                     | 0.621 / 0.601 (96.8 %)                                    |
| S100P    | YOLOv9e Seg | 0.481 / 0.450 (93.6 %)                                  | 0.292 / 0.245 (83.9 %)                                    | 0.537 / 0.507 (94.3 %)                                     | 0.650 / 0.628 (96.6 %)                                    |
| S100P    | YOLOv8n Seg | 0.304 / 0.282 (92.9 %)                                  | 0.109 / 0.087 (79.7 %)                                    | 0.334 / 0.310 (92.8 %)                                     | 0.461 / 0.440 (95.4 %)                                    |
| S100P    | YOLOv8s Seg | 0.386 / 0.363 (94.0 %)                                  | 0.180 / 0.149 (82.8 %)                                    | 0.432 / 0.410 (94.8 %)                                     | 0.564 / 0.547 (97.0 %)                                    |
| S100P    | YOLOv8m Seg | 0.431 / 0.407 (94.3 %)                                  | 0.228 / 0.191 (83.9 %)                                    | 0.486 / 0.467 (96.0 %)                                     | 0.608 / 0.596 (98.0 %)                                    |
| S100P    | YOLOv8l Seg | 0.453 / 0.426 (94.1 %)                                  | 0.258 / 0.220 (85.0 %)                                    | 0.502 / 0.483 (96.3 %)                                     | 0.626 / 0.607 (97.0 %)                                    |
| S100P    | YOLOv8x Seg | 0.465 / 0.435 (93.5 %)                                  | 0.268 / 0.214 (79.7 %)                                    | 0.520 / 0.496 (95.2 %)                                     | 0.641 / 0.622 (97.1 %)                                    |
##### mask
| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Seg | 0.262 / 0.226 (86.3 %)                                  | 0.062 / 0.044 (72.0 %)                                    | 0.283 / 0.250 (88.2 %)                                     | 0.444 / 0.394 (88.8 %)                                    |
| S100P    | YOLO11s Seg | 0.311 / 0.287 (92.2 %)                                  | 0.099 / 0.088 (88.9 %)                                    | 0.350 / 0.326 (93.3 %)                                     | 0.509 / 0.474 (93.2 %)                                    |
| S100P    | YOLO11m Seg | 0.347 / 0.315 (90.7 %)                                  | 0.136 / 0.122 (90.3 %)                                    | 0.396 / 0.362 (91.4 %)                                     | 0.549 / 0.493 (89.8 %)                                    |
| S100P    | YOLO11l Seg | 0.357 / 0.325 (91.1 %)                                  | 0.143 / 0.126 (88.1 %)                                    | 0.409 / 0.374 (91.4 %)                                     | 0.560 / 0.504 (90.1 %)                                    |
| S100P    | YOLO11x Seg | 0.366 / 0.331 (90.4 %)                                  | 0.149 / 0.129 (86.9 %)                                    | 0.420 / 0.379 (90.2 %)                                     | 0.572 / 0.520 (90.9 %)                                    |
| S100P    | YOLOv9c Seg | 0.352 / 0.319 (90.7 %)                                  | 0.132 / 0.116 (88.1 %)                                    | 0.404 / 0.367 (90.8 %)                                     | 0.547 / 0.497 (91.0 %)                                    |
| S100P    | YOLOv9e Seg | 0.371 / 0.340 (91.7 %)                                  | 0.155 / 0.136 (87.8 %)                                    | 0.425 / 0.386 (90.7 %)                                     | 0.571 / 0.525 (92.0 %)                                    |
| S100P    | YOLOv8n Seg | 0.246 / 0.221 (89.9 %)                                  | 0.059 / 0.048 (81.8 %)                                    | 0.265 / 0.243 (91.8 %)                                     | 0.409 / 0.364 (89.0 %)                                    |
| S100P    | YOLOv8s Seg | 0.305 / 0.282 (92.6 %)                                  | 0.096 / 0.086 (90.2 %)                                    | 0.343 / 0.316 (92.1 %)                                     | 0.496 / 0.457 (92.1 %)                                    |
| S100P    | YOLOv8m Seg | 0.337 / 0.312 (92.7 %)                                  | 0.121 / 0.110 (90.8 %)                                    | 0.386 / 0.358 (92.8 %)                                     | 0.533 / 0.494 (92.5 %)                                    |
| S100P    | YOLOv8l Seg | 0.351 / 0.326 (92.9 %)                                  | 0.137 / 0.126 (92.1 %)                                    | 0.398 / 0.371 (93.4 %)                                     | 0.550 / 0.509 (92.5 %)                                    |
| S100P    | YOLOv8x Seg | 0.358 / 0.331 (92.3 %)                                  | 0.139 / 0.119 (85.5 %)                                    | 0.409 / 0.379 (92.5 %)                                     | 0.562 / 0.514 (91.4 %)                                    |
#### Pose Estimation
| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100P    | YOLO11n Pose | 0.465 / 0.445 (95.7 %)                                  | 0.386 / 0.373 (96.5 %)                                     | 0.597 / 0.568 (95.1 %)                                    |
| S100P    | YOLO11s Pose | 0.559 / 0.533 (95.4 %)                                  | 0.495 / 0.467 (94.5 %)                                     | 0.672 / 0.649 (96.6 %)                                    |
| S100P    | YOLO11m Pose | 0.627 / 0.607 (96.8 %)                                  | 0.586 / 0.563 (96.2 %)                                     | 0.711 / 0.692 (97.3 %)                                    |
| S100P    | YOLO11l Pose | 0.636 / 0.617 (97.0 %)                                  | 0.592 / 0.570 (96.3 %)                                     | 0.726 / 0.704 (97.0 %)                                    |
| S100P    | YOLO11x Pose | 0.672 / 0.648 (96.5 %)                                  | 0.634 / 0.605 (95.5 %)                                     | 0.750 / 0.733 (97.8 %)                                    |
| S100P    | YOLOv8n Pose | 0.476 / 0.460 (96.7 %)                                  | 0.391 / 0.372 (95.0 %)                                     | 0.610 / 0.593 (97.2 %)                                    |
| S100P    | YOLOv8s Pose | 0.578 / 0.550 (95.2 %)                                  | 0.510 / 0.476 (93.4 %)                                     | 0.692 / 0.667 (96.4 %)                                    |
| S100P    | YOLOv8m Pose | 0.630 / 0.605 (96.0 %)                                  | 0.578 / 0.553 (95.7 %)                                     | 0.724 / 0.697 (96.3 %)                                    |
| S100P    | YOLOv8l Pose | 0.657 / 0.631 (96.1 %)                                  | 0.607 / 0.579 (95.3 %)                                     | 0.747 / 0.726 (97.2 %)                                    |
| S100P    | YOLOv8x Pose | 0.671 / 0.649 (96.7 %)                                  | 0.624 / 0.602 (96.4 %)                                     | 0.757 / 0.733 (96.8 %)                                    |
#### Image Classification
| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| S100P    | YOLO11n CLS | 0.700 / 0.566 (80.8 %)                  | 0.894 / 0.803 (89.8 %)                  |
| S100P    | YOLO11s CLS | 0.754 / 0.661 (87.7 %)                  | 0.927 / 0.872 (94.1 %)                  |
| S100P    | YOLO11m CLS | 0.773 / 0.706 (91.3 %)                  | 0.939 / 0.903 (96.1 %)                  |
| S100P    | YOLO11l CLS | 0.783 / 0.712 (90.8 %)                  | 0.942 / 0.905 (96.1 %)                  |
| S100P    | YOLO11x CLS | 0.795 / 0.734 (92.4 %)                  | 0.949 / 0.919 (96.8 %)                  |
| S100P    | YOLOv8n CLS | 0.689 / 0.577 (83.7 %)                  | 0.883 / 0.808 (91.5 %)                  |
| S100P    | YOLOv8s CLS | 0.737 / 0.631 (85.6 %)                  | 0.917 / 0.850 (92.8 %)                  |
| S100P    | YOLOv8m CLS | 0.768 / 0.703 (91.6 %)                  | 0.935 / 0.899 (96.2 %)                  |
| S100P    | YOLOv8l CLS | 0.783 / 0.723 (92.3 %)                  | 0.942 / 0.910 (96.6 %)                  |
| S100P    | YOLOv8x CLS | 0.790 / 0.742 (93.9 %)                  | 0.945 / 0.923 (97.6 %)                  |
## Performance
### RDK S100
#### Obeject Detection
| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO12n Detect  | 640×640        |        80 | 2.65 ms / 368.54 FPS (1 thread ) <br/> 4.43 ms / 443.33 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100     | YOLO12s Detect  | 640×640        |        80 | 4.48 ms / 220.08 FPS (1 thread ) <br/> 8.10 ms / 244.66 FPS (2 threads)  | 2.0 ms                         | 9.3 M       | 21.4 M     |
| S100     | YOLO12m Detect  | 640×640        |        80 | 9.27 ms / 107.09 FPS (1 thread ) <br/> 17.56 ms / 113.12 FPS (2 threads) | 2.0 ms                         | 20.2 M      | 67.5 M     |
| S100     | YOLO12l Detect  | 640×640        |        80 | 14.66 ms / 67.85 FPS (1 thread ) <br/> 28.30 ms / 70.28 FPS (2 threads)  | 2.0 ms                         | 26.4 M      | 88.9 M     |
| S100     | YOLO12x Detect  | 640×640        |        80 | 24.72 ms / 40.33 FPS (1 thread ) <br/> 48.27 ms / 41.26 FPS (2 threads)  | 2.0 ms                         | 59.1 M      | 199.0 M    |
| S100     | YOLO11n Detect  | 640×640        |        80 | 1.62 ms / 596.53 FPS (1 thread ) <br/> 2.39 ms / 813.87 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 6.5 M      |
| S100     | YOLO11s Detect  | 640×640        |        80 | 2.63 ms / 371.42 FPS (1 thread ) <br/> 4.39 ms / 448.18 FPS (2 threads)  | 2.0 ms                         | 9.4 M       | 21.5 M     |
| S100     | YOLO11m Detect  | 640×640        |        80 | 5.63 ms / 175.69 FPS (1 thread ) <br/> 10.35 ms / 191.62 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 68.0 M     |
| S100     | YOLO11l Detect  | 640×640        |        80 | 6.96 ms / 142.36 FPS (1 thread ) <br/> 13.02 ms / 152.41 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 86.9 M     |
| S100     | YOLO11x Detect  | 640×640        |        80 | 13.13 ms / 75.78 FPS (1 thread ) <br/> 25.24 ms / 78.82 FPS (2 threads)  | 2.0 ms                         | 56.9 M      | 194.9 M    |
| S100     | YOLOv10n Detect | 640×640        |        80 | 1.58 ms / 608.94 FPS (1 thread ) <br/> 2.32 ms / 837.04 FPS (2 threads)  | 2.0 ms                         | 2.3 M       | 6.7 M      |
| S100     | YOLOv10s Detect | 640×640        |        80 | 2.53 ms / 385.50 FPS (1 thread ) <br/> 4.18 ms / 471.09 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 21.6 M     |
| S100     | YOLOv10m Detect | 640×640        |        80 | 4.49 ms / 219.98 FPS (1 thread ) <br/> 8.11 ms / 244.17 FPS (2 threads)  | 2.0 ms                         | 15.4 M      | 59.1 M     |
| S100     | YOLOv10b Detect | 640×640        |        80 | 6.28 ms / 157.57 FPS (1 thread ) <br/> 11.65 ms / 170.32 FPS (2 threads) | 2.0 ms                         | 19.1 M      | 92.0 M     |
| S100     | YOLOv10l Detect | 640×640        |        80 | 7.95 ms / 124.70 FPS (1 thread ) <br/> 14.98 ms / 132.53 FPS (2 threads) | 2.0 ms                         | 24.4 M      | 120.3 M    |
| S100     | YOLOv10x Detect | 640×640        |        80 | 10.83 ms / 91.79 FPS (1 thread ) <br/> 20.66 ms / 96.17 FPS (2 threads)  | 2.0 ms                         | 29.5 M      | 160.4 M    |
| S100     | YOLOv9t Detect  | 640×640        |        80 | 1.77 ms / 546.03 FPS (1 thread ) <br/> 2.67 ms / 730.68 FPS (2 threads)  | 2.0 ms                         | 2.1 M       | 8.2 M      |
| S100     | YOLOv9s Detect  | 640×640        |        80 | 2.74 ms / 357.91 FPS (1 thread ) <br/> 4.62 ms / 425.97 FPS (2 threads)  | 2.0 ms                         | 7.2 M       | 26.9 M     |
| S100     | YOLOv9m Detect  | 640×640        |        80 | 5.52 ms / 179.23 FPS (1 thread ) <br/> 10.13 ms / 195.30 FPS (2 threads) | 2.0 ms                         | 20.1 M      | 76.8 M     |
| S100     | YOLOv9c Detect  | 640×640        |        80 | 6.98 ms / 142.00 FPS (1 thread ) <br/> 13.05 ms / 151.95 FPS (2 threads) | 2.0 ms                         | 25.3 M      | 102.7 M    |
| S100     | YOLOv9e Detect  | 640×640        |        80 | 17.75 ms / 56.15 FPS (1 thread ) <br/> 34.41 ms / 57.85 FPS (2 threads)  | 2.0 ms                         | 57.4 M      | 189.5 M    |
| S100     | YOLOv8n Detect  | 640×640        |        80 | 1.53 ms / 632.06 FPS (1 thread ) <br/> 2.24 ms / 868.87 FPS (2 threads)  | 2.0 ms                         | 3.2 M       | 8.7 M      |
| S100     | YOLOv8s Detect  | 640×640        |        80 | 2.63 ms / 371.16 FPS (1 thread ) <br/> 4.41 ms / 446.48 FPS (2 threads)  | 2.0 ms                         | 11.2 M      | 28.6 M     |
| S100     | YOLOv8m Detect  | 640×640        |        80 | 5.18 ms / 190.64 FPS (1 thread ) <br/> 9.45 ms / 209.80 FPS (2 threads)  | 2.0 ms                         | 25.9 M      | 78.9 M     |
| S100     | YOLOv8l Detect  | 640×640        |        80 | 9.97 ms / 99.68 FPS (1 thread ) <br/> 19.00 ms / 104.65 FPS (2 threads)  | 2.0 ms                         | 43.7 M      | 165.2 M    |
| S100     | YOLOv8x Detect  | 640×640        |        80 | 15.77 ms / 63.15 FPS (1 thread ) <br/> 30.53 ms / 65.20 FPS (2 threads)  | 2.0 ms                         | 68.2 M      | 257.8 M    |
| S100     | YOLOv5nu Detect | 640×640        |        80 | 1.42 ms / 674.92 FPS (1 thread ) <br/> 2.02 ms / 959.05 FPS (2 threads)  | 2.0 ms                         | 2.6 M       | 7.7 M      |
| S100     | YOLOv5su Detect | 640×640        |        80 | 2.31 ms / 420.83 FPS (1 thread ) <br/> 3.79 ms / 519.22 FPS (2 threads)  | 2.0 ms                         | 9.1 M       | 24.0 M     |
| S100     | YOLOv5mu Detect | 640×640        |        80 | 4.50 ms / 218.77 FPS (1 thread ) <br/> 8.11 ms / 244.06 FPS (2 threads)  | 2.0 ms                         | 25.1 M      | 64.2 M     |
| S100     | YOLOv5lu Detect | 640×640        |        80 | 8.96 ms / 110.78 FPS (1 thread ) <br/> 16.97 ms / 117.15 FPS (2 threads) | 2.0 ms                         | 53.2 M      | 135.0 M    |
| S100     | YOLOv5xu Detect | 640×640        |        80 | 15.97 ms / 62.32 FPS (1 thread ) <br/> 30.90 ms / 64.41 FPS (2 threads)  | 2.0 ms                         | 97.2 M      | 246.4 M    |
#### Instance Segmentation
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n Seg | 640×640        |        80 | 2.06 ms / 463.93 FPS (1 thread ) <br/> 3.17 ms / 613.76 FPS (2 threads)  | 5.0 ms                         | 2.9 M       | 10.4 M     |
| S100     | YOLO11s Seg | 640×640        |        80 | 3.34 ms / 291.16 FPS (1 thread ) <br/> 5.69 ms / 344.91 FPS (2 threads)  | 5.0 ms                         | 10.1 M      | 35.5 M     |
| S100     | YOLO11m Seg | 640×640        |        80 | 7.86 ms / 125.63 FPS (1 thread ) <br/> 14.67 ms / 135.16 FPS (2 threads) | 5.0 ms                         | 22.4 M      | 123.3 M    |
| S100     | YOLO11l Seg | 640×640        |        80 | 9.17 ms / 108.00 FPS (1 thread ) <br/> 17.29 ms / 114.67 FPS (2 threads) | 5.0 ms                         | 27.6 M      | 142.2 M    |
| S100     | YOLO11x Seg | 640×640        |        80 | 17.74 ms / 56.07 FPS (1 thread ) <br/> 34.33 ms / 57.96 FPS (2 threads)  | 5.0 ms                         | 62.1 M      | 319.0 M    |
| S100     | YOLOv9c Seg | 640×640        |        80 | 9.07 ms / 109.16 FPS (1 thread ) <br/> 17.12 ms / 115.80 FPS (2 threads) | 5.0 ms                         | 27.7 M      | 158.0 M    |
| S100     | YOLOv9e Seg | 640×640        |        80 | 20.15 ms / 49.38 FPS (1 thread ) <br/> 39.07 ms / 50.91 FPS (2 threads)  | 5.0 ms                         | 59.7 M      | 244.8 M    |
| S100     | YOLOv8n Seg | 640×640        |        80 | 1.93 ms / 495.35 FPS (1 thread ) <br/> 2.98 ms / 652.21 FPS (2 threads)  | 5.0 ms                         | 3.4 M       | 12.6 M     |
| S100     | YOLOv8s Seg | 640×640        |        80 | 3.37 ms / 288.70 FPS (1 thread ) <br/> 5.76 ms / 341.12 FPS (2 threads)  | 5.0 ms                         | 11.8 M      | 42.6 M     |
| S100     | YOLOv8m Seg | 640×640        |        80 | 6.65 ms / 148.28 FPS (1 thread ) <br/> 12.29 ms / 161.07 FPS (2 threads) | 5.0 ms                         | 27.3 M      | 100.2 M    |
| S100     | YOLOv8l Seg | 640×640        |        80 | 12.21 ms / 81.34 FPS (1 thread ) <br/> 23.32 ms / 85.17 FPS (2 threads)  | 5.0 ms                         | 46.0 M      | 220.5 M    |
| S100     | YOLOv8x Seg | 640×640        |        80 | 19.51 ms / 51.00 FPS (1 thread ) <br/> 37.80 ms / 52.62 FPS (2 threads)  | 5.0 ms                         | 71.8 M      | 344.1 M    |
#### Pose Estimation
| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n Pose | 640×640        |        80 | 1.69 ms / 568.00 FPS (1 thread ) <br/> 2.48 ms / 780.47 FPS (2 threads)  | 1.0 ms                         | 2.9 M       | 7.6 M      |
| S100     | YOLO11s Pose | 640×640        |        80 | 2.76 ms / 354.06 FPS (1 thread ) <br/> 4.62 ms / 424.92 FPS (2 threads)  | 1.0 ms                         | 9.9 M       | 23.2 M     |
| S100     | YOLO11m Pose | 640×640        |        80 | 5.89 ms / 167.50 FPS (1 thread ) <br/> 10.79 ms / 183.34 FPS (2 threads) | 1.0 ms                         | 20.9 M      | 71.7 M     |
| S100     | YOLO11l Pose | 640×640        |        80 | 7.23 ms / 136.91 FPS (1 thread ) <br/> 13.48 ms / 147.21 FPS (2 threads) | 1.0 ms                         | 26.2 M      | 90.7 M     |
| S100     | YOLO11x Pose | 640×640        |        80 | 13.61 ms / 73.02 FPS (1 thread ) <br/> 26.16 ms / 76.04 FPS (2 threads)  | 1.0 ms                         | 58.8 M      | 203.3 M    |
| S100     | YOLOv8n Pose | 640×640        |        80 | 1.62 ms / 587.59 FPS (1 thread ) <br/> 2.31 ms / 837.95 FPS (2 threads)  | 1.0 ms                         | 3.3 M       | 9.2 M      |
| S100     | YOLOv8s Pose | 640×640        |        80 | 2.83 ms / 344.35 FPS (1 thread ) <br/> 4.71 ms / 417.72 FPS (2 threads)  | 1.0 ms                         | 11.6 M      | 30.2 M     |
| S100     | YOLOv8m Pose | 640×640        |        80 | 5.47 ms / 180.46 FPS (1 thread ) <br/> 9.92 ms / 199.50 FPS (2 threads)  | 1.0 ms                         | 26.4 M      | 81.0 M     |
| S100     | YOLOv8l Pose | 640×640        |        80 | 10.31 ms / 96.20 FPS (1 thread ) <br/> 19.55 ms / 101.60 FPS (2 threads) | 1.0 ms                         | 44.4 M      | 168.6 M    |
| S100     | YOLOv8x Pose | 640×640        |        80 | 16.07 ms / 61.88 FPS (1 thread ) <br/> 31.01 ms / 64.14 FPS (2 threads)  | 1.0 ms                         | 69.4 M      | 263.2 M    |
#### Image Classification
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                                                                   | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|-------------------------------------------------------------------------------------------------------------------|--------------------------------|-------------|------------|
| S100     | YOLO11n CLS | 640×640        |        80 | 0.53 ms / 1827.92 FPS (1 thread ) <br/> 0.62 ms / 3115.65 FPS (2 threads) <br/> 0.70 ms / 4141.22 FPS (3 threads) | 0.5 ms                         | 2.8 M       | 4.2 M      |
| S100     | YOLO11s CLS | 640×640        |        80 | 0.68 ms / 1415.98 FPS (1 thread ) <br/> 0.76 ms / 2553.99 FPS (2 threads) <br/> 1.05 ms / 2767.63 FPS (3 threads) | 0.5 ms                         | 6.7 M       | 13.0 M     |
| S100     | YOLO11m CLS | 640×640        |        80 | 1.02 ms / 955.28 FPS (1 thread ) <br/> 1.35 ms / 1445.18 FPS (2 threads)                                          | 0.5 ms                         | 11.6 M      | 40.3 M     |
| S100     | YOLO11l CLS | 640×640        |        80 | 1.21 ms / 805.52 FPS (1 thread ) <br/> 1.73 ms / 1139.48 FPS (2 threads)                                          | 0.5 ms                         | 14.1 M      | 50.4 M     |
| S100     | YOLO11x CLS | 640×640        |        80 | 1.97 ms / 501.49 FPS (1 thread ) <br/> 3.23 ms / 612.29 FPS (2 threads)                                           | 0.5 ms                         | 29.6 M      | 111.3 M    |
| S100     | YOLOv8n CLS | 640×640        |        80 | 0.49 ms / 1928.23 FPS (1 thread ) <br/> 0.57 ms / 3399.86 FPS (2 threads) <br/> 0.66 ms / 4410.92 FPS (3 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| S100     | YOLOv8s CLS | 640×640        |        80 | 0.62 ms / 1562.83 FPS (1 thread ) <br/> 0.71 ms / 2712.53 FPS (2 threads) <br/> 0.89 ms / 3279.66 FPS (3 threads) | 0.5 ms                         | 6.4 M       | 13.5 M     |
| S100     | YOLOv8m CLS | 640×640        |        80 | 1.00 ms / 970.04 FPS (1 thread ) <br/> 1.31 ms / 1500.86 FPS (2 threads)                                          | 0.5 ms                         | 17.0 M      | 42.7 M     |
| S100     | YOLOv8l CLS | 640×640        |        80 | 1.98 ms / 497.58 FPS (1 thread ) <br/> 3.22 ms / 614.92 FPS (2 threads)                                           | 0.5 ms                         | 37.5 M      | 99.7 M     |
| S100     | YOLOv8x CLS | 640×640        |        80 | 2.77 ms / 357.03 FPS (1 thread ) <br/> 4.81 ms / 412.60 FPS (2 threads)                                           | 0.5 ms                         | 57.4 M      | 154.8 M    |
## Accuracy
#### Obeject Detection
| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO12n Detect  | 0.338 / 0.311 (92.0 %)                                  | 0.128 / 0.096 (74.9 %)                                    | 0.374 / 0.344 (91.8 %)                                     | 0.524 / 0.507 (96.6 %)                                    |
| S100     | YOLO12s Detect  | 0.403 / 0.380 (94.3 %)                                  | 0.201 / 0.156 (77.4 %)                                    | 0.450 / 0.431 (95.9 %)                                     | 0.602 / 0.573 (95.1 %)                                    |
| S100     | YOLO12m Detect  | 0.452 / 0.424 (93.8 %)                                  | 0.251 / 0.211 (84.2 %)                                    | 0.509 / 0.488 (95.9 %)                                     | 0.638 / 0.609 (95.4 %)                                    |
| S100     | YOLO12l Detect  | 0.463 / 0.431 (93.1 %)                                  | 0.268 / 0.220 (82.0 %)                                    | 0.522 / 0.494 (94.7 %)                                     | 0.646 / 0.629 (97.5 %)                                    |
| S100     | YOLO12x Detect  | 0.475 / 0.441 (92.8 %)                                  | 0.276 / 0.215 (78.0 %)                                    | 0.536 / 0.512 (95.5 %)                                     | 0.659 / 0.619 (94.0 %)                                    |
| S100     | YOLO11n Detect  | 0.327 / 0.309 (94.5 %)                                  | 0.130 / 0.108 (83.2 %)                                    | 0.357 / 0.338 (94.7 %)                                     | 0.511 / 0.497 (97.4 %)                                    |
| S100     | YOLO11s Detect  | 0.400 / 0.380 (95.2 %)                                  | 0.198 / 0.167 (84.5 %)                                    | 0.445 / 0.426 (95.9 %)                                     | 0.587 / 0.575 (97.9 %)                                    |
| S100     | YOLO11m Detect  | 0.444 / 0.417 (94.1 %)                                  | 0.247 / 0.211 (85.7 %)                                    | 0.497 / 0.479 (96.3 %)                                     | 0.627 / 0.590 (94.1 %)                                    |
| S100     | YOLO11l Detect  | 0.460 / 0.433 (94.1 %)                                  | 0.267 / 0.226 (84.9 %)                                    | 0.520 / 0.495 (95.3 %)                                     | 0.638 / 0.605 (94.9 %)                                    |
| S100     | YOLO11x Detect  | 0.474 / 0.445 (93.7 %)                                  | 0.283 / 0.231 (81.4 %)                                    | 0.529 / 0.506 (95.5 %)                                     | 0.652 / 0.623 (95.5 %)                                    |
| S100     | YOLOv10n Detect | 0.303 / 0.278 (91.7 %)                                  | 0.099 / 0.068 (68.6 %)                                    | 0.330 / 0.304 (92.1 %)                                     | 0.478 / 0.455 (95.3 %)                                    |
| S100     | YOLOv10s Detect | 0.386 / 0.354 (91.7 %)                                  | 0.175 / 0.122 (69.7 %)                                    | 0.434 / 0.405 (93.3 %)                                     | 0.574 / 0.529 (92.2 %)                                    |
| S100     | YOLOv10m Detect | 0.425 / 0.374 (88.0 %)                                  | 0.221 / 0.179 (81.0 %)                                    | 0.481 / 0.439 (91.3 %)                                     | 0.603 / 0.490 (81.3 %)                                    |
| S100     | YOLOv10b Detect | 0.443 / 0.380 (85.7 %)                                  | 0.242 / 0.193 (79.7 %)                                    | 0.498 / 0.434 (87.1 %)                                     | 0.618 / 0.469 (75.9 %)                                    |
| S100     | YOLOv10l Detect | 0.445 / 0.380 (85.4 %)                                  | 0.258 / 0.209 (81.3 %)                                    | 0.498 / 0.444 (89.1 %)                                     | 0.626 / 0.476 (76.0 %)                                    |
| S100     | YOLOv10x Detect | 0.459 / 0.413 (90.0 %)                                  | 0.258 / 0.214 (82.9 %)                                    | 0.518 / 0.480 (92.7 %)                                     | 0.639 / 0.539 (84.3 %)                                    |
| S100     | YOLOv9t Detect  | 0.313 / 0.300 (95.8 %)                                  | 0.113 / 0.105 (93.7 %)                                    | 0.338 / 0.325 (96.1 %)                                     | 0.483 / 0.458 (94.9 %)                                    |
| S100     | YOLOv9s Detect  | 0.400 / 0.383 (95.9 %)                                  | 0.191 / 0.160 (84.0 %)                                    | 0.444 / 0.435 (97.9 %)                                     | 0.583 / 0.556 (95.4 %)                                    |
| S100     | YOLOv9m Detect  | 0.449 / 0.434 (96.6 %)                                  | 0.253 / 0.228 (90.3 %)                                    | 0.504 / 0.492 (97.6 %)                                     | 0.617 / 0.593 (96.1 %)                                    |
| S100     | YOLOv9c Detect  | 0.461 / 0.445 (96.5 %)                                  | 0.269 / 0.246 (91.6 %)                                    | 0.512 / 0.500 (97.6 %)                                     | 0.640 / 0.610 (95.2 %)                                    |
| S100     | YOLOv9e Detect  | 0.481 / 0.460 (95.7 %)                                  | 0.298 / 0.266 (89.3 %)                                    | 0.538 / 0.516 (95.9 %)                                     | 0.662 / 0.626 (94.5 %)                                    |
| S100     | YOLOv8n Detect  | 0.309 / 0.291 (94.3 %)                                  | 0.113 / 0.101 (89.3 %)                                    | 0.338 / 0.320 (94.8 %)                                     | 0.473 / 0.448 (94.7 %)                                    |
| S100     | YOLOv8s Detect  | 0.391 / 0.373 (95.5 %)                                  | 0.195 / 0.168 (86.2 %)                                    | 0.437 / 0.421 (96.4 %)                                     | 0.566 / 0.556 (98.3 %)                                    |
| S100     | YOLOv8m Detect  | 0.441 / 0.419 (95.2 %)                                  | 0.249 / 0.213 (85.7 %)                                    | 0.494 / 0.477 (96.5 %)                                     | 0.618 / 0.602 (97.4 %)                                    |
| S100     | YOLOv8l Detect  | 0.461 / 0.441 (95.6 %)                                  | 0.271 / 0.241 (88.9 %)                                    | 0.516 / 0.499 (96.6 %)                                     | 0.651 / 0.625 (96.0 %)                                    |
| S100     | YOLOv8x Detect  | 0.474 / 0.449 (94.7 %)                                  | 0.280 / 0.250 (89.2 %)                                    | 0.527 / 0.505 (95.8 %)                                     | 0.658 / 0.628 (95.5 %)                                    |
| S100     | YOLOv5nu Detect | 0.278 / 0.261 (93.6 %)                                  | 0.093 / 0.081 (86.6 %)                                    | 0.309 / 0.287 (93.0 %)                                     | 0.417 / 0.400 (96.0 %)                                    |
| S100     | YOLOv5su Detect | 0.367 / 0.352 (95.9 %)                                  | 0.169 / 0.144 (85.3 %)                                    | 0.416 / 0.402 (96.7 %)                                     | 0.530 / 0.521 (98.3 %)                                    |
| S100     | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.195 (86.4 %)                                    | 0.477 / 0.465 (97.6 %)                                     | 0.603 / 0.594 (98.4 %)                                    |
| S100     | YOLOv5lu Detect | 0.458 / 0.437 (95.4 %)                                  | 0.260 / 0.226 (86.9 %)                                    | 0.516 / 0.499 (96.7 %)                                     | 0.641 / 0.628 (97.9 %)                                    |
| S100     | YOLOv5xu Detect | 0.466 / 0.445 (95.6 %)                                  | 0.281 / 0.238 (84.8 %)                                    | 0.523 / 0.506 (96.9 %)                                     | 0.645 / 0.634 (98.3 %)                                    |
#### Instance Segmentation
##### bbox
| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Seg | 0.322 / 0.295 (91.6 %)                                  | 0.113 / 0.084 (74.2 %)                                    | 0.352 / 0.322 (91.3 %)                                     | 0.502 / 0.487 (97.0 %)                                    |
| S100     | YOLO11s Seg | 0.394 / 0.369 (93.7 %)                                  | 0.184 / 0.149 (81.0 %)                                    | 0.442 / 0.419 (94.9 %)                                     | 0.582 / 0.571 (98.1 %)                                    |
| S100     | YOLO11m Seg | 0.443 / 0.414 (93.2 %)                                  | 0.246 / 0.206 (83.5 %)                                    | 0.497 / 0.473 (95.3 %)                                     | 0.627 / 0.590 (94.2 %)                                    |
| S100     | YOLO11l Seg | 0.460 / 0.428 (93.1 %)                                  | 0.267 / 0.217 (81.5 %)                                    | 0.520 / 0.490 (94.3 %)                                     | 0.638 / 0.604 (94.8 %)                                    |
| S100     | YOLO11x Seg | 0.474 / 0.440 (92.8 %)                                  | 0.283 / 0.223 (78.7 %)                                    | 0.529 / 0.501 (94.6 %)                                     | 0.652 / 0.621 (95.2 %)                                    |
| S100     | YOLOv9c Seg | 0.453 / 0.420 (92.6 %)                                  | 0.254 / 0.206 (81.2 %)                                    | 0.508 / 0.479 (94.2 %)                                     | 0.621 / 0.584 (93.9 %)                                    |
| S100     | YOLOv9e Seg | 0.481 / 0.449 (93.5 %)                                  | 0.292 / 0.246 (84.1 %)                                    | 0.537 / 0.506 (94.3 %)                                     | 0.650 / 0.620 (95.4 %)                                    |
| S100     | YOLOv8n Seg | 0.304 / 0.283 (93.1 %)                                  | 0.109 / 0.088 (80.3 %)                                    | 0.334 / 0.310 (92.8 %)                                     | 0.461 / 0.441 (95.5 %)                                    |
| S100     | YOLOv8s Seg | 0.386 / 0.363 (94.1 %)                                  | 0.180 / 0.153 (85.2 %)                                    | 0.432 / 0.405 (93.7 %)                                     | 0.564 / 0.550 (97.5 %)                                    |
| S100     | YOLOv8m Seg | 0.431 / 0.407 (94.4 %)                                  | 0.228 / 0.193 (84.7 %)                                    | 0.486 / 0.468 (96.2 %)                                     | 0.608 / 0.591 (97.2 %)                                    |
| S100     | YOLOv8l Seg | 0.453 / 0.425 (93.9 %)                                  | 0.258 / 0.214 (83.0 %)                                    | 0.502 / 0.484 (96.4 %)                                     | 0.626 / 0.592 (94.6 %)                                    |
| S100     | YOLOv8x Seg | 0.465 / 0.434 (93.4 %)                                  | 0.268 / 0.216 (80.6 %)                                    | 0.520 / 0.494 (95.0 %)                                     | 0.641 / 0.613 (95.6 %)                                    |
##### mask
| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Seg | 0.262 / 0.227 (86.7 %)                                  | 0.062 / 0.046 (75.3 %)                                    | 0.283 / 0.249 (88.0 %)                                     | 0.444 / 0.392 (88.3 %)                                    |
| S100     | YOLO11s Seg | 0.311 / 0.285 (91.7 %)                                  | 0.099 / 0.088 (89.3 %)                                    | 0.350 / 0.322 (91.9 %)                                     | 0.509 / 0.470 (92.3 %)                                    |
| S100     | YOLO11m Seg | 0.347 / 0.313 (90.3 %)                                  | 0.136 / 0.121 (89.0 %)                                    | 0.396 / 0.361 (91.2 %)                                     | 0.549 / 0.482 (87.8 %)                                    |
| S100     | YOLO11l Seg | 0.357 / 0.324 (90.7 %)                                  | 0.143 / 0.124 (86.9 %)                                    | 0.409 / 0.372 (90.8 %)                                     | 0.560 / 0.499 (89.1 %)                                    |
| S100     | YOLO11x Seg | 0.366 / 0.332 (90.6 %)                                  | 0.149 / 0.124 (83.2 %)                                    | 0.420 / 0.381 (90.8 %)                                     | 0.572 / 0.516 (90.3 %)                                    |
| S100     | YOLOv9c Seg | 0.352 / 0.317 (90.1 %)                                  | 0.132 / 0.116 (87.9 %)                                    | 0.404 / 0.366 (90.5 %)                                     | 0.547 / 0.485 (88.6 %)                                    |
| S100     | YOLOv9e Seg | 0.371 / 0.340 (91.6 %)                                  | 0.155 / 0.137 (88.2 %)                                    | 0.425 / 0.386 (90.8 %)                                     | 0.571 / 0.521 (91.3 %)                                    |
| S100     | YOLOv8n Seg | 0.246 / 0.220 (89.8 %)                                  | 0.059 / 0.049 (83.0 %)                                    | 0.265 / 0.242 (91.5 %)                                     | 0.409 / 0.365 (89.3 %)                                    |
| S100     | YOLOv8s Seg | 0.305 / 0.281 (92.3 %)                                  | 0.096 / 0.088 (92.3 %)                                    | 0.343 / 0.313 (91.2 %)                                     | 0.496 / 0.459 (92.5 %)                                    |
| S100     | YOLOv8m Seg | 0.337 / 0.311 (92.2 %)                                  | 0.121 / 0.112 (92.1 %)                                    | 0.386 / 0.358 (92.8 %)                                     | 0.533 / 0.484 (90.7 %)                                    |
| S100     | YOLOv8l Seg | 0.351 / 0.326 (92.8 %)                                  | 0.137 / 0.124 (90.8 %)                                    | 0.398 / 0.372 (93.4 %)                                     | 0.550 / 0.495 (90.1 %)                                    |
| S100     | YOLOv8x Seg | 0.358 / 0.330 (92.0 %)                                  | 0.139 / 0.120 (86.6 %)                                    | 0.409 / 0.377 (92.0 %)                                     | 0.562 / 0.508 (90.4 %)                                    |
#### Pose Estimation
| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| S100     | YOLO11n Pose | 0.465 / 0.451 (97.0 %)                                  | 0.386 / 0.375 (97.2 %)                                     | 0.597 / 0.576 (96.5 %)                                    |
| S100     | YOLO11s Pose | 0.559 / 0.531 (95.0 %)                                  | 0.495 / 0.465 (94.0 %)                                     | 0.672 / 0.647 (96.3 %)                                    |
| S100     | YOLO11m Pose | 0.627 / 0.601 (95.9 %)                                  | 0.586 / 0.559 (95.4 %)                                     | 0.711 / 0.690 (97.0 %)                                    |
| S100     | YOLO11l Pose | 0.636 / 0.615 (96.6 %)                                  | 0.592 / 0.571 (96.6 %)                                     | 0.726 / 0.698 (96.1 %)                                    |
| S100     | YOLO11x Pose | 0.672 / 0.651 (96.8 %)                                  | 0.634 / 0.607 (95.8 %)                                     | 0.750 / 0.734 (97.9 %)                                    |
| S100     | YOLOv8n Pose | 0.476 / 0.461 (96.9 %)                                  | 0.391 / 0.372 (95.1 %)                                     | 0.610 / 0.595 (97.6 %)                                    |
| S100     | YOLOv8s Pose | 0.578 / 0.548 (94.9 %)                                  | 0.510 / 0.475 (93.3 %)                                     | 0.692 / 0.667 (96.4 %)                                    |
| S100     | YOLOv8m Pose | 0.630 / 0.604 (95.9 %)                                  | 0.578 / 0.551 (95.4 %)                                     | 0.724 / 0.699 (96.5 %)                                    |
| S100     | YOLOv8l Pose | 0.657 / 0.632 (96.3 %)                                  | 0.607 / 0.578 (95.2 %)                                     | 0.747 / 0.728 (97.5 %)                                    |
| S100     | YOLOv8x Pose | 0.671 / 0.649 (96.7 %)                                  | 0.624 / 0.596 (95.5 %)                                     | 0.757 / 0.739 (97.6 %)                                    |
#### Image Classification
| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| S100     | YOLO11n CLS | 0.700 / 0.590 (84.3 %)                  | 0.894 / 0.820 (91.7 %)                  |
| S100     | YOLO11s CLS | 0.754 / 0.667 (88.6 %)                  | 0.927 / 0.875 (94.5 %)                  |
| S100     | YOLO11m CLS | 0.773 / 0.706 (91.3 %)                  | 0.939 / 0.902 (96.1 %)                  |
| S100     | YOLO11l CLS | 0.783 / 0.712 (90.9 %)                  | 0.942 / 0.906 (96.1 %)                  |
| S100     | YOLO11x CLS | 0.795 / 0.733 (92.2 %)                  | 0.949 / 0.918 (96.7 %)                  |
| S100     | YOLOv8n CLS | 0.689 / 0.570 (82.7 %)                  | 0.883 / 0.802 (90.8 %)                  |
| S100     | YOLOv8s CLS | 0.737 / 0.636 (86.3 %)                  | 0.917 / 0.852 (92.9 %)                  |
| S100     | YOLOv8m CLS | 0.768 / 0.702 (91.4 %)                  | 0.935 / 0.899 (96.2 %)                  |
| S100     | YOLOv8l CLS | 0.783 / 0.723 (92.3 %)                  | 0.942 / 0.909 (96.5 %)                  |
| S100     | YOLOv8x CLS | 0.790 / 0.742 (93.9 %)                  | 0.945 / 0.921 (97.5 %)                  |
## Performance
### RDK X5
#### Obeject Detection
| Device   | Model           | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-----------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO12n Detect  | 640×640        |        80 | 39.70 ms / 25.17 FPS (1 thread ) <br/> 73.19 ms / 27.24 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| X5       | YOLO12s Detect  | 640×640        |        80 | 63.74 ms / 15.68 FPS (1 thread ) <br/> 121.24 ms / 16.45 FPS (2 threads) | 5.0 ms                         | 9.3 M       | 21.4 M     |
| X5       | YOLO12m Detect  | 640×640        |        80 | 103.02 ms / 9.70 FPS (1 thread ) <br/> 199.58 ms / 9.99 FPS (2 threads)  | 5.0 ms                         | 20.2 M      | 67.5 M     |
| X5       | YOLO12l Detect  | 640×640        |        80 | 183.00 ms / 5.46 FPS (1 thread ) <br/> 359.03 ms / 5.56 FPS (2 threads)  | 5.0 ms                         | 26.4 M      | 88.9 M     |
| X5       | YOLO12x Detect  | 640×640        |        80 | 315.16 ms / 3.17 FPS (1 thread )                                         | 5.0 ms                         | 59.1 M      | 199.0 M    |
| X5       | YOLO11n Detect  | 640×640        |        80 | 8.25 ms / 121.05 FPS (1 thread ) <br/> 10.56 ms / 188.57 FPS (2 threads) | 5.0 ms                         | 2.6 M       | 6.5 M      |
| X5       | YOLO11s Detect  | 640×640        |        80 | 15.81 ms / 63.16 FPS (1 thread ) <br/> 25.74 ms / 77.43 FPS (2 threads)  | 5.0 ms                         | 9.4 M       | 21.5 M     |
| X5       | YOLO11m Detect  | 640×640        |        80 | 34.68 ms / 28.82 FPS (1 thread ) <br/> 63.30 ms / 31.51 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 68.0 M     |
| X5       | YOLO11l Detect  | 640×640        |        80 | 45.23 ms / 22.10 FPS (1 thread ) <br/> 84.30 ms / 23.66 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 86.9 M     |
| X5       | YOLO11x Detect  | 640×640        |        80 | 96.70 ms / 10.34 FPS (1 thread ) <br/> 186.76 ms / 10.68 FPS (2 threads) | 5.0 ms                         | 56.9 M      | 194.9 M    |
| X5       | YOLOv10n Detect | 640×640        |        80 | 8.75 ms / 114.19 FPS (1 thread ) <br/> 11.60 ms / 171.72 FPS (2 threads) | 5.0 ms                         | 2.3 M       | 6.7 M      |
| X5       | YOLOv10s Detect | 640×640        |        80 | 14.84 ms / 67.32 FPS (1 thread ) <br/> 23.85 ms / 83.58 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 21.6 M     |
| X5       | YOLOv10m Detect | 640×640        |        80 | 29.40 ms / 33.99 FPS (1 thread ) <br/> 52.83 ms / 37.75 FPS (2 threads)  | 5.0 ms                         | 15.4 M      | 59.1 M     |
| X5       | YOLOv10b Detect | 640×640        |        80 | 40.14 ms / 24.90 FPS (1 thread ) <br/> 74.20 ms / 26.88 FPS (2 threads)  | 5.0 ms                         | 19.1 M      | 92.0 M     |
| X5       | YOLOv10l Detect | 640×640        |        80 | 49.89 ms / 20.04 FPS (1 thread ) <br/> 93.66 ms / 21.30 FPS (2 threads)  | 5.0 ms                         | 24.4 M      | 120.3 M    |
| X5       | YOLOv10x Detect | 640×640        |        80 | 68.92 ms / 14.51 FPS (1 thread ) <br/> 131.54 ms / 15.16 FPS (2 threads) | 5.0 ms                         | 29.5 M      | 160.4 M    |
| X5       | YOLOv9t Detect  | 640×640        |        80 | 6.97 ms / 143.14 FPS (1 thread ) <br/> 7.96 ms / 250.11 FPS (2 threads)  | 5.0 ms                         | 2.1 M       | 8.2 M      |
| X5       | YOLOv9s Detect  | 640×640        |        80 | 13.00 ms / 76.81 FPS (1 thread ) <br/> 20.16 ms / 98.81 FPS (2 threads)  | 5.0 ms                         | 7.2 M       | 26.9 M     |
| X5       | YOLOv9m Detect  | 640×640        |        80 | 32.63 ms / 30.63 FPS (1 thread ) <br/> 59.31 ms / 33.62 FPS (2 threads)  | 5.0 ms                         | 20.1 M      | 76.8 M     |
| X5       | YOLOv9c Detect  | 640×640        |        80 | 40.46 ms / 24.71 FPS (1 thread ) <br/> 74.77 ms / 26.67 FPS (2 threads)  | 5.0 ms                         | 25.3 M      | 102.7 M    |
| X5       | YOLOv9e Detect  | 640×640        |        80 | 119.80 ms / 8.35 FPS (1 thread ) <br/> 233.08 ms / 8.56 FPS (2 threads)  | 5.0 ms                         | 57.4 M      | 189.5 M    |
| X5       | YOLOv8n Detect  | 640×640        |        80 | 7.00 ms / 142.60 FPS (1 thread ) <br/> 8.06 ms / 246.82 FPS (2 threads)  | 5.0 ms                         | 3.2 M       | 8.7 M      |
| X5       | YOLOv8s Detect  | 640×640        |        80 | 13.63 ms / 73.30 FPS (1 thread ) <br/> 21.38 ms / 93.20 FPS (2 threads)  | 5.0 ms                         | 11.2 M      | 28.6 M     |
| X5       | YOLOv8m Detect  | 640×640        |        80 | 30.74 ms / 32.51 FPS (1 thread ) <br/> 55.51 ms / 35.93 FPS (2 threads)  | 5.0 ms                         | 25.9 M      | 78.9 M     |
| X5       | YOLOv8l Detect  | 640×640        |        80 | 59.51 ms / 16.80 FPS (1 thread ) <br/> 112.80 ms / 17.68 FPS (2 threads) | 5.0 ms                         | 43.7 M      | 165.2 M    |
| X5       | YOLOv8x Detect  | 640×640        |        80 | 92.72 ms / 10.78 FPS (1 thread ) <br/> 178.95 ms / 11.15 FPS (2 threads) | 5.0 ms                         | 68.2 M      | 257.8 M    |
| X5       | YOLOv5nu Detect | 640×640        |        80 | 6.33 ms / 157.59 FPS (1 thread ) <br/> 6.80 ms / 291.89 FPS (2 threads)  | 5.0 ms                         | 2.6 M       | 7.7 M      |
| X5       | YOLOv5su Detect | 640×640        |        80 | 12.33 ms / 81.04 FPS (1 thread ) <br/> 18.88 ms / 105.56 FPS (2 threads) | 5.0 ms                         | 9.1 M       | 24.0 M     |
| X5       | YOLOv5mu Detect | 640×640        |        80 | 26.57 ms / 37.62 FPS (1 thread ) <br/> 47.20 ms / 42.24 FPS (2 threads)  | 5.0 ms                         | 25.1 M      | 64.2 M     |
| X5       | YOLOv5lu Detect | 640×640        |        80 | 52.83 ms / 18.92 FPS (1 thread ) <br/> 99.42 ms / 20.06 FPS (2 threads)  | 5.0 ms                         | 53.2 M      | 135.0 M    |
| X5       | YOLOv5xu Detect | 640×640        |        80 | 91.55 ms / 10.92 FPS (1 thread ) <br/> 176.49 ms / 11.30 FPS (2 threads) | 5.0 ms                         | 97.2 M      | 246.4 M    |
#### Instance Segmentation
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n Seg | 640×640        |        80 | 11.55 ms / 86.39 FPS (1 thread ) <br/> 12.83 ms / 155.10 FPS (2 threads) | 20.0 ms                        | 2.9 M       | 10.4 M     |
| X5       | YOLO11s Seg | 640×640        |        80 | 21.62 ms / 46.22 FPS (1 thread ) <br/> 33.12 ms / 60.20 FPS (2 threads)  | 20.0 ms                        | 10.1 M      | 35.5 M     |
| X5       | YOLO11m Seg | 640×640        |        80 | 50.43 ms / 19.82 FPS (1 thread ) <br/> 90.49 ms / 22.04 FPS (2 threads)  | 20.0 ms                        | 22.4 M      | 123.3 M    |
| X5       | YOLO11l Seg | 640×640        |        80 | 60.60 ms / 16.50 FPS (1 thread ) <br/> 110.99 ms / 17.97 FPS (2 threads) | 20.0 ms                        | 27.6 M      | 142.2 M    |
| X5       | YOLO11x Seg | 640×640        |        80 | 130.40 ms / 7.67 FPS (1 thread ) <br/> 249.71 ms / 7.99 FPS (2 threads)  | 20.0 ms                        | 62.1 M      | 319.0 M    |
| X5       | YOLOv9c Seg | 640×640        |        80 | 55.85 ms / 17.90 FPS (1 thread ) <br/> 101.47 ms / 19.65 FPS (2 threads) | 20.0 ms                        | 27.7 M      | 158.0 M    |
| X5       | YOLOv9e Seg | 640×640        |        80 | 135.34 ms / 7.39 FPS (1 thread ) <br/> 260.08 ms / 7.67 FPS (2 threads)  | 20.0 ms                        | 59.7 M      | 244.8 M    |
| X5       | YOLOv8n Seg | 640×640        |        80 | 10.40 ms / 96.02 FPS (1 thread ) <br/> 10.75 ms / 185.21 FPS (2 threads) | 20.0 ms                        | 3.4 M       | 12.6 M     |
| X5       | YOLOv8s Seg | 640×640        |        80 | 19.56 ms / 51.08 FPS (1 thread ) <br/> 28.99 ms / 68.76 FPS (2 threads)  | 20.0 ms                        | 11.8 M      | 42.6 M     |
| X5       | YOLOv8m Seg | 640×640        |        80 | 40.52 ms / 24.67 FPS (1 thread ) <br/> 70.70 ms / 28.21 FPS (2 threads)  | 20.0 ms                        | 27.3 M      | 100.2 M    |
| X5       | YOLOv8l Seg | 640×640        |        80 | 75.00 ms / 13.33 FPS (1 thread ) <br/> 139.61 ms / 14.29 FPS (2 threads) | 20.0 ms                        | 46.0 M      | 220.5 M    |
| X5       | YOLOv8x Seg | 640×640        |        80 | 115.94 ms / 8.62 FPS (1 thread ) <br/> 221.06 ms / 9.02 FPS (2 threads)  | 20.0 ms                        | 71.8 M      | 344.1 M    |
#### Pose Estimation
| Device   | Model        | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                          | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|--------------|----------------|-----------|--------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n Pose | 640×640        |        80 | 8.36 ms / 119.43 FPS (1 thread ) <br/> 10.97 ms / 181.61 FPS (2 threads) | 10.0 ms                        | 2.9 M       | 7.6 M      |
| X5       | YOLO11s Pose | 640×640        |        80 | 16.35 ms / 61.11 FPS (1 thread ) <br/> 26.99 ms / 73.85 FPS (2 threads)  | 10.0 ms                        | 9.9 M       | 23.2 M     |
| X5       | YOLO11m Pose | 640×640        |        80 | 35.74 ms / 27.97 FPS (1 thread ) <br/> 65.60 ms / 30.40 FPS (2 threads)  | 10.0 ms                        | 20.9 M      | 71.7 M     |
| X5       | YOLO11l Pose | 640×640        |        80 | 46.38 ms / 21.55 FPS (1 thread ) <br/> 86.82 ms / 22.97 FPS (2 threads)  | 10.0 ms                        | 26.2 M      | 90.7 M     |
| X5       | YOLO11x Pose | 640×640        |        80 | 98.88 ms / 10.11 FPS (1 thread ) <br/> 191.38 ms / 10.42 FPS (2 threads) | 10.0 ms                        | 58.8 M      | 203.3 M    |
| X5       | YOLOv8n Pose | 640×640        |        80 | 6.95 ms / 143.64 FPS (1 thread ) <br/> 8.23 ms / 241.76 FPS (2 threads)  | 10.0 ms                        | 3.3 M       | 9.2 M      |
| X5       | YOLOv8s Pose | 640×640        |        80 | 14.16 ms / 70.54 FPS (1 thread ) <br/> 22.62 ms / 88.09 FPS (2 threads)  | 10.0 ms                        | 11.6 M      | 30.2 M     |
| X5       | YOLOv8m Pose | 640×640        |        80 | 31.60 ms / 31.62 FPS (1 thread ) <br/> 57.34 ms / 34.78 FPS (2 threads)  | 10.0 ms                        | 26.4 M      | 81.0 M     |
| X5       | YOLOv8l Pose | 640×640        |        80 | 60.37 ms / 16.56 FPS (1 thread ) <br/> 114.73 ms / 17.38 FPS (2 threads) | 10.0 ms                        | 44.4 M      | 168.6 M    |
| X5       | YOLOv8x Pose | 640×640        |        80 | 94.15 ms / 10.62 FPS (1 thread ) <br/> 182.08 ms / 10.96 FPS (2 threads) | 10.0 ms                        | 69.4 M      | 263.2 M    |
#### Image Classification
| Device   | Model       | Size(Pixels)   |   Classes | BPU Task Latency  /<br>BPU Throughput (Threads)                           | CPU Latency<br>(Single Core)   | params(M)   | FLOPs(B)   |
|----------|-------------|----------------|-----------|---------------------------------------------------------------------------|--------------------------------|-------------|------------|
| X5       | YOLO11n CLS | 640×640        |        80 | 1.06 ms / 939.95 FPS (1 thread ) <br/> 1.61 ms / 1236.07 FPS (2 threads)  | 0.5 ms                         | 2.8 M       | 4.2 M      |
| X5       | YOLO11s CLS | 640×640        |        80 | 2.01 ms / 495.14 FPS (1 thread ) <br/> 3.49 ms / 569.44 FPS (2 threads)   | 0.5 ms                         | 6.7 M       | 13.0 M     |
| X5       | YOLO11m CLS | 640×640        |        80 | 3.82 ms / 261.13 FPS (1 thread ) <br/> 7.09 ms / 280.82 FPS (2 threads)   | 0.5 ms                         | 11.6 M      | 40.3 M     |
| X5       | YOLO11l CLS | 640×640        |        80 | 5.02 ms / 199.15 FPS (1 thread ) <br/> 9.49 ms / 210.12 FPS (2 threads)   | 0.5 ms                         | 14.1 M      | 50.4 M     |
| X5       | YOLO11x CLS | 640×640        |        80 | 10.04 ms / 99.49 FPS (1 thread ) <br/> 19.48 ms / 102.39 FPS (2 threads)  | 0.5 ms                         | 29.6 M      | 111.3 M    |
| X5       | YOLOv8n CLS | 640×640        |        80 | 0.74 ms / 1348.98 FPS (1 thread ) <br/> 0.98 ms / 2018.94 FPS (2 threads) | 0.5 ms                         | 2.7 M       | 4.3 M      |
| X5       | YOLOv8s CLS | 640×640        |        80 | 1.44 ms / 690.86 FPS (1 thread ) <br/> 2.36 ms / 842.52 FPS (2 threads)   | 0.5 ms                         | 6.4 M       | 13.5 M     |
| X5       | YOLOv8m CLS | 640×640        |        80 | 3.66 ms / 272.72 FPS (1 thread ) <br/> 6.78 ms / 294.01 FPS (2 threads)   | 0.5 ms                         | 17.0 M      | 42.7 M     |
| X5       | YOLOv8l CLS | 640×640        |        80 | 7.98 ms / 125.23 FPS (1 thread ) <br/> 15.38 ms / 129.63 FPS (2 threads)  | 0.5 ms                         | 37.5 M      | 99.7 M     |
| X5       | YOLOv8x CLS | 640×640        |        80 | 13.12 ms / 76.18 FPS (1 thread ) <br/> 25.64 ms / 77.78 FPS (2 threads)   | 0.5 ms                         | 57.4 M      | 154.8 M    |
## Accuracy
#### Obeject Detection
| Device   | Model           | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-----------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO12n Detect  | 0.338 / 0.313 (92.5 %)                                  | 0.128 / 0.095 (74.3 %)                                    | 0.374 / 0.343 (91.7 %)                                     | 0.524 / 0.511 (97.4 %)                                    |
| X5       | YOLO12s Detect  | 0.403 / 0.379 (94.0 %)                                  | 0.201 / 0.157 (78.1 %)                                    | 0.450 / 0.427 (95.0 %)                                     | 0.602 / 0.575 (95.5 %)                                    |
| X5       | YOLO12m Detect  | 0.452 / 0.424 (93.8 %)                                  | 0.251 / 0.208 (82.7 %)                                    | 0.509 / 0.489 (96.1 %)                                     | 0.638 / 0.617 (96.7 %)                                    |
| X5       | YOLO12l Detect  | 0.463 / 0.434 (93.8 %)                                  | 0.268 / 0.212 (78.9 %)                                    | 0.522 / 0.499 (95.6 %)                                     | 0.646 / 0.630 (97.6 %)                                    |
| X5       | YOLO12x Detect  | 0.475 / 0.443 (93.3 %)                                  | 0.276 / 0.227 (82.3 %)                                    | 0.536 / 0.513 (95.7 %)                                     | 0.659 / 0.632 (95.9 %)                                    |
| X5       | YOLO11n Detect  | 0.327 / 0.310 (95.1 %)                                  | 0.130 / 0.110 (84.8 %)                                    | 0.357 / 0.341 (95.4 %)                                     | 0.511 / 0.498 (97.5 %)                                    |
| X5       | YOLO11s Detect  | 0.400 / 0.381 (95.2 %)                                  | 0.198 / 0.165 (83.1 %)                                    | 0.445 / 0.426 (95.8 %)                                     | 0.587 / 0.577 (98.3 %)                                    |
| X5       | YOLO11m Detect  | 0.444 / 0.278 (62.7 %)                                  | 0.247 / 0.048 (19.3 %)                                    | 0.497 / 0.299 (60.2 %)                                     | 0.627 / 0.490 (78.2 %)                                    |
| X5       | YOLO11l Detect  | 0.460 / 0.435 (94.7 %)                                  | 0.267 / 0.224 (84.1 %)                                    | 0.520 / 0.499 (96.0 %)                                     | 0.638 / 0.611 (95.8 %)                                    |
| X5       | YOLO11x Detect  | 0.474 / 0.445 (93.9 %)                                  | 0.283 / 0.233 (82.3 %)                                    | 0.529 / 0.505 (95.3 %)                                     | 0.652 / 0.627 (96.2 %)                                    |
| X5       | YOLOv10n Detect | 0.303 / 0.280 (92.5 %)                                  | 0.099 / 0.079 (79.2 %)                                    | 0.330 / 0.302 (91.3 %)                                     | 0.478 / 0.457 (95.7 %)                                    |
| X5       | YOLOv10s Detect | 0.386 / 0.357 (92.4 %)                                  | 0.175 / 0.131 (74.7 %)                                    | 0.434 / 0.406 (93.6 %)                                     | 0.574 / 0.520 (90.6 %)                                    |
| X5       | YOLOv10m Detect | 0.425 / 0.379 (89.1 %)                                  | 0.221 / 0.181 (82.0 %)                                    | 0.481 / 0.439 (91.3 %)                                     | 0.603 / 0.502 (83.2 %)                                    |
| X5       | YOLOv10b Detect | 0.443 / 0.390 (88.1 %)                                  | 0.242 / 0.207 (85.6 %)                                    | 0.498 / 0.435 (87.2 %)                                     | 0.618 / 0.502 (81.2 %)                                    |
| X5       | YOLOv10l Detect | 0.445 / 0.379 (85.1 %)                                  | 0.258 / 0.211 (81.9 %)                                    | 0.498 / 0.440 (88.2 %)                                     | 0.626 / 0.476 (76.1 %)                                    |
| X5       | YOLOv10x Detect | 0.459 / 0.418 (91.3 %)                                  | 0.258 / 0.216 (83.8 %)                                    | 0.518 / 0.480 (92.8 %)                                     | 0.639 / 0.562 (88.0 %)                                    |
| X5       | YOLOv9t Detect  | 0.313 / 0.299 (95.6 %)                                  | 0.113 / 0.105 (93.1 %)                                    | 0.338 / 0.322 (95.5 %)                                     | 0.483 / 0.456 (94.5 %)                                    |
| X5       | YOLOv9s Detect  | 0.400 / 0.384 (96.2 %)                                  | 0.191 / 0.174 (90.9 %)                                    | 0.444 / 0.430 (96.8 %)                                     | 0.583 / 0.557 (95.6 %)                                    |
| X5       | YOLOv9m Detect  | 0.449 / 0.432 (96.3 %)                                  | 0.253 / 0.227 (89.6 %)                                    | 0.504 / 0.488 (96.8 %)                                     | 0.617 / 0.604 (97.9 %)                                    |
| X5       | YOLOv9c Detect  | 0.461 / 0.440 (95.5 %)                                  | 0.269 / 0.242 (90.1 %)                                    | 0.512 / 0.497 (96.9 %)                                     | 0.640 / 0.611 (95.4 %)                                    |
| X5       | YOLOv9e Detect  | 0.481 / 0.462 (96.1 %)                                  | 0.298 / 0.268 (90.1 %)                                    | 0.538 / 0.514 (95.5 %)                                     | 0.662 / 0.642 (97.0 %)                                    |
| X5       | YOLOv8n Detect  | 0.309 / 0.293 (94.7 %)                                  | 0.113 / 0.103 (90.9 %)                                    | 0.338 / 0.323 (95.4 %)                                     | 0.473 / 0.448 (94.7 %)                                    |
| X5       | YOLOv8s Detect  | 0.391 / 0.378 (96.7 %)                                  | 0.195 / 0.174 (89.3 %)                                    | 0.437 / 0.426 (97.5 %)                                     | 0.566 / 0.558 (98.6 %)                                    |
| X5       | YOLOv8m Detect  | 0.441 / 0.425 (96.4 %)                                  | 0.249 / 0.220 (88.6 %)                                    | 0.494 / 0.480 (97.0 %)                                     | 0.618 / 0.612 (99.0 %)                                    |
| X5       | YOLOv8l Detect  | 0.461 / 0.444 (96.2 %)                                  | 0.271 / 0.243 (89.6 %)                                    | 0.516 / 0.501 (97.1 %)                                     | 0.651 / 0.628 (96.4 %)                                    |
| X5       | YOLOv8x Detect  | 0.474 / 0.451 (95.1 %)                                  | 0.280 / 0.251 (89.7 %)                                    | 0.527 / 0.504 (95.6 %)                                     | 0.658 / 0.638 (97.0 %)                                    |
| X5       | YOLOv5nu Detect | 0.278 / 0.212 (76.0 %)                                  | 0.093 / 0.043 (46.2 %)                                    | 0.309 / 0.219 (71.0 %)                                     | 0.417 / 0.356 (85.5 %)                                    |
| X5       | YOLOv5su Detect | 0.367 / 0.354 (96.5 %)                                  | 0.169 / 0.148 (88.0 %)                                    | 0.416 / 0.402 (96.7 %)                                     | 0.530 / 0.523 (98.6 %)                                    |
| X5       | YOLOv5mu Detect | 0.425 / 0.406 (95.6 %)                                  | 0.226 / 0.195 (86.1 %)                                    | 0.477 / 0.461 (96.7 %)                                     | 0.603 / 0.594 (98.5 %)                                    |
| X5       | YOLOv5lu Detect | 0.458 / 0.440 (96.0 %)                                  | 0.260 / 0.226 (87.0 %)                                    | 0.516 / 0.503 (97.3 %)                                     | 0.641 / 0.627 (97.7 %)                                    |
| X5       | YOLOv5xu Detect | 0.466 / 0.448 (96.2 %)                                  | 0.281 / 0.241 (85.8 %)                                    | 0.523 / 0.512 (98.0 %)                                     | 0.645 / 0.639 (99.2 %)                                    |
#### Instance Segmentation
##### bbox
| Device   | Model       | Accuracy bbox-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy bbox-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Seg | 0.322 / 0.294 (91.4 %)                                  | 0.113 / 0.089 (78.8 %)                                    | 0.352 / 0.320 (90.9 %)                                     | 0.502 / 0.479 (95.3 %)                                    |
| X5       | YOLO11s Seg | 0.394 / 0.373 (94.7 %)                                  | 0.184 / 0.155 (84.2 %)                                    | 0.442 / 0.422 (95.5 %)                                     | 0.582 / 0.570 (97.9 %)                                    |
| X5       | YOLO11m Seg | 0.443 / 0.413 (93.2 %)                                  | 0.246 / 0.199 (80.8 %)                                    | 0.497 / 0.472 (95.0 %)                                     | 0.627 / 0.601 (95.9 %)                                    |
| X5       | YOLO11l Seg | 0.460 / 0.430 (93.6 %)                                  | 0.267 / 0.216 (80.9 %)                                    | 0.520 / 0.494 (95.2 %)                                     | 0.638 / 0.609 (95.5 %)                                    |
| X5       | YOLO11x Seg | 0.474 / 0.441 (92.9 %)                                  | 0.283 / 0.225 (79.3 %)                                    | 0.529 / 0.500 (94.4 %)                                     | 0.652 / 0.625 (95.9 %)                                    |
| X5       | YOLOv9c Seg | 0.453 / 0.422 (93.0 %)                                  | 0.254 / 0.205 (80.7 %)                                    | 0.508 / 0.482 (94.7 %)                                     | 0.621 / 0.605 (97.3 %)                                    |
| X5       | YOLOv9e Seg | 0.481 / 0.452 (94.2 %)                                  | 0.292 / 0.254 (86.7 %)                                    | 0.537 / 0.507 (94.4 %)                                     | 0.650 / 0.632 (97.1 %)                                    |
| X5       | YOLOv8n Seg | 0.304 / 0.286 (94.1 %)                                  | 0.109 / 0.091 (83.6 %)                                    | 0.334 / 0.314 (94.0 %)                                     | 0.461 / 0.446 (96.7 %)                                    |
| X5       | YOLOv8s Seg | 0.386 / 0.368 (95.2 %)                                  | 0.180 / 0.155 (86.4 %)                                    | 0.432 / 0.413 (95.4 %)                                     | 0.564 / 0.554 (98.3 %)                                    |
| X5       | YOLOv8m Seg | 0.431 / 0.410 (95.1 %)                                  | 0.228 / 0.197 (86.4 %)                                    | 0.486 / 0.469 (96.5 %)                                     | 0.608 / 0.596 (98.0 %)                                    |
| X5       | YOLOv8l Seg | 0.453 / 0.427 (94.2 %)                                  | 0.258 / 0.216 (83.5 %)                                    | 0.502 / 0.487 (96.9 %)                                     | 0.626 / 0.605 (96.6 %)                                    |
| X5       | YOLOv8x Seg | 0.465 / 0.438 (94.3 %)                                  | 0.268 / 0.219 (81.6 %)                                    | 0.520 / 0.499 (96.0 %)                                     | 0.641 / 0.626 (97.8 %)                                    |
##### mask
| Device   | Model       | Accuracy mask-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-small mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy mask-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|-------------|---------------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Seg | 0.262 / 0.224 (85.6 %)                                  | 0.062 / 0.049 (79.2 %)                                    | 0.283 / 0.245 (86.6 %)                                     | 0.444 / 0.384 (86.5 %)                                    |
| X5       | YOLO11s Seg | 0.311 / 0.288 (92.6 %)                                  | 0.099 / 0.092 (93.0 %)                                    | 0.350 / 0.324 (92.7 %)                                     | 0.509 / 0.470 (92.3 %)                                    |
| X5       | YOLO11m Seg | 0.347 / 0.314 (90.5 %)                                  | 0.136 / 0.115 (84.6 %)                                    | 0.396 / 0.361 (91.2 %)                                     | 0.549 / 0.492 (89.5 %)                                    |
| X5       | YOLO11l Seg | 0.357 / 0.325 (90.9 %)                                  | 0.143 / 0.125 (87.1 %)                                    | 0.409 / 0.373 (91.1 %)                                     | 0.560 / 0.504 (90.0 %)                                    |
| X5       | YOLO11x Seg | 0.366 / 0.332 (90.6 %)                                  | 0.149 / 0.125 (84.5 %)                                    | 0.420 / 0.379 (90.2 %)                                     | 0.572 / 0.520 (91.0 %)                                    |
| X5       | YOLOv9c Seg | 0.352 / 0.319 (90.5 %)                                  | 0.132 / 0.115 (87.0 %)                                    | 0.404 / 0.366 (90.6 %)                                     | 0.547 / 0.500 (91.6 %)                                    |
| X5       | YOLOv9e Seg | 0.371 / 0.342 (92.2 %)                                  | 0.155 / 0.142 (91.6 %)                                    | 0.425 / 0.386 (90.9 %)                                     | 0.571 / 0.527 (92.3 %)                                    |
| X5       | YOLOv8n Seg | 0.246 / 0.222 (90.4 %)                                  | 0.059 / 0.052 (87.6 %)                                    | 0.265 / 0.246 (92.8 %)                                     | 0.409 / 0.368 (89.9 %)                                    |
| X5       | YOLOv8s Seg | 0.305 / 0.284 (93.0 %)                                  | 0.096 / 0.089 (93.3 %)                                    | 0.343 / 0.318 (92.9 %)                                     | 0.496 / 0.462 (93.1 %)                                    |
| X5       | YOLOv8m Seg | 0.337 / 0.314 (93.2 %)                                  | 0.121 / 0.113 (93.6 %)                                    | 0.386 / 0.360 (93.4 %)                                     | 0.533 / 0.493 (92.5 %)                                    |
| X5       | YOLOv8l Seg | 0.351 / 0.327 (93.2 %)                                  | 0.137 / 0.124 (90.5 %)                                    | 0.398 / 0.374 (94.0 %)                                     | 0.550 / 0.506 (92.1 %)                                    |
| X5       | YOLOv8x Seg | 0.358 / 0.332 (92.6 %)                                  | 0.139 / 0.121 (87.1 %)                                    | 0.409 / 0.380 (92.7 %)                                     | 0.562 / 0.517 (91.9 %)                                    |
#### Pose Estimation
| Device   | Model        | Accuracy pose-all mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-medium mAP@.50:.95 <br/> FP32 / BPU Python   | Accuracy pose-large mAP@.50:.95 <br/> FP32 / BPU Python   |
|----------|--------------|---------------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| X5       | YOLO11n Pose | 0.465 / 0.453 (97.3 %)                                  | 0.386 / 0.379 (98.2 %)                                     | 0.597 / 0.577 (96.6 %)                                    |
| X5       | YOLO11s Pose | 0.559 / 0.532 (95.1 %)                                  | 0.495 / 0.468 (94.7 %)                                     | 0.672 / 0.644 (95.8 %)                                    |
| X5       | YOLO11m Pose | 0.627 / 0.609 (97.1 %)                                  | 0.586 / 0.565 (96.4 %)                                     | 0.711 / 0.693 (97.4 %)                                    |
| X5       | YOLO11l Pose | 0.636 / 0.619 (97.3 %)                                  | 0.592 / 0.569 (96.3 %)                                     | 0.726 / 0.710 (97.8 %)                                    |
| X5       | YOLO11x Pose | 0.672 / 0.650 (96.8 %)                                  | 0.634 / 0.609 (96.1 %)                                     | 0.750 / 0.733 (97.8 %)                                    |
| X5       | YOLOv8n Pose | 0.476 / 0.459 (96.4 %)                                  | 0.391 / 0.373 (95.3 %)                                     | 0.610 / 0.594 (97.4 %)                                    |
| X5       | YOLOv8s Pose | 0.578 / 0.551 (95.3 %)                                  | 0.510 / 0.478 (93.7 %)                                     | 0.692 / 0.667 (96.5 %)                                    |
| X5       | YOLOv8m Pose | 0.630 / 0.606 (96.2 %)                                  | 0.578 / 0.552 (95.6 %)                                     | 0.724 / 0.692 (95.5 %)                                    |
| X5       | YOLOv8l Pose | 0.657 / 0.632 (96.3 %)                                  | 0.607 / 0.582 (95.8 %)                                     | 0.747 / 0.725 (97.0 %)                                    |
| X5       | YOLOv8x Pose | 0.671 / 0.648 (96.6 %)                                  | 0.624 / 0.599 (96.0 %)                                     | 0.757 / 0.736 (97.2 %)                                    |
#### Image Classification
| Device   | Model       | Accuracy TOP1 <br/> FP32 / BPU Python   | Accuracy TOP5 <br/> FP32 / BPU Python   |
|----------|-------------|-----------------------------------------|-----------------------------------------|
| X5       | YOLO11n CLS | 0.700 / 0.585 (83.6 %)                  | 0.894 / 0.815 (91.2 %)                  |
| X5       | YOLO11s CLS | 0.754 / 0.663 (88.0 %)                  | 0.927 / 0.873 (94.2 %)                  |
| X5       | YOLO11m CLS | 0.773 / 0.708 (91.5 %)                  | 0.939 / 0.903 (96.1 %)                  |
| X5       | YOLO11l CLS | 0.783 / 0.714 (91.1 %)                  | 0.942 / 0.906 (96.1 %)                  |
| X5       | YOLO11x CLS | 0.795 / 0.733 (92.2 %)                  | 0.949 / 0.917 (96.6 %)                  |
| X5       | YOLOv8n CLS | 0.689 / 0.574 (83.2 %)                  | 0.883 / 0.806 (91.2 %)                  |
| X5       | YOLOv8s CLS | 0.737 / 0.635 (86.1 %)                  | 0.917 / 0.850 (92.8 %)                  |
| X5       | YOLOv8m CLS | 0.768 / 0.702 (91.5 %)                  | 0.935 / 0.899 (96.2 %)                  |
| X5       | YOLOv8l CLS | 0.783 / 0.727 (92.9 %)                  | 0.942 / 0.912 (96.9 %)                  |
| X5       | YOLOv8x CLS | 0.790 / 0.741 (93.8 %)                  | 0.945 / 0.921 (97.5 %)                  |
