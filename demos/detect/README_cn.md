# 目标检测
目标检测(Object Detection)的任务是找出图像中所有感兴趣的目标(物体), 确定它们的类别(id, score)和位置(xyxy).
- [目标检测](#目标检测)
  - [模型数据(简明)](#模型数据简明)
    - [RDK X5 \& RDK X5 Module](#rdk-x5--rdk-x5-module)
    - [RDK X3 \& RDK X3 Module](#rdk-x3--rdk-x3-module)
  - [模型数据(详细)](#模型数据详细)


## 模型数据(简明)

### RDK X5 & RDK X5 Module
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | 参数量 | BPU吞吐量 | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|
| fcos_efficientnetb0 | 512×512 | 80 | - | 323.0 FPS | 9 ms |
| fcos_efficientnetb2 | 768×768 | 80 | - | 70.9 FPS | 16 ms |
| fcos_efficientnetb3 | 896×896 | 80 | - | 38.7 FPS | 20 ms |
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640×640 | 80 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640×640 | 80 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640×640 | 80 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640×640 | 80 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 13.1 FPS | 12 ms |
| YOLOv8n | 640×640 | 80 | 3.2 M | 263.6 FPS | 5 ms |
| YOLOv8s | 640×640 | 80 | 11.2 M | 94.9 FPS | 5 ms |
| YOLOv8m | 640×640 | 80 | 25.9 M | 35.7 FPS | 5 ms |
| YOLOv8l | 640×640 | 80 | 43.7 M | 17.9 FPS | 5 ms |
| YOLOv8x | 640×640 | 80 | 68.2 M | 11.2 FPS | 5 ms |
| YOLOv10n | 640×640 | 80 | 6.7 G | 132.7 FPS | 4.5 ms | 
| YOLOv10s | 640×640 | 80 | 21.6 G | 71.0 FPS | 4.5 ms |  
| YOLOv10m | 640×640 | 80 | 59.1 G | 34.5 FPS | 4.5 ms |  
| YOLOv10b | 640×640 | 80 | 92.0 G | 25.4 FPS | 4.5 ms |  
| YOLOv10l | 640×640 | 80 | 120.3 G | 20.0 FPS | 4.5 ms |  
| YOLOv10x | 640×640 | 80 | 160.4 G | 14.5 FPS | 4.5 ms |  

### RDK X3 & RDK X3 Module
目标检测 Detection (COCO)
| 模型(公版) | 尺寸(像素) | 类别数 | 参数量 | BPU吞吐量 | 后处理时间(Python) |
|---------|---------|-------|---------|---------|----------|
| fcos | 512×512 | 80 | - | 173.9 FPS | 5 ms |
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 38.2 FPS | 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 3.9 FPS | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 37.2 FPS | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 20.9 FPS | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 3.6 FPS | 13 ms |
| YOLOv8n | 640×640 | 80 | 3.2 M | 34.1 FPS | 6 ms |
| YOLOv10n | 640×640 | 80 | 6.7 G | 18.1 FPS | 5 ms | 

## 模型数据(详细)
模型详细数据, 包括各个线程下的BPU帧延迟, BPU吞吐量, 后处理时间, mAP等数据, 在对应子文件夹的README.md中.
