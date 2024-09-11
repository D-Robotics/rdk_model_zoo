# Object detection
The task of Object Detection is to find all objects (objects) of interest in an image, determine their category (id, score) and location (xyxy).

## Model Data (concise)


### RDK X5 & RDK X5 Module
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
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
| YOLOv8s | 640×640 | 80 | 11.2 M | 194.9 FPS | 5 ms |
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
Object Detection (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| fcos | 512×512 | 80 | - | 173.9 FPS | 5 ms |
| YOLOv5s_v2.0 | 640×640 | 80 | 7.5 M | 38.2 FPS | 13 ms |
| YOLOv5x_v2.0 | 640×640 | 80 | 89.0 M | 3.9 FPS | 13 ms |
| YOLOv5n_v7.0 | 640×640 | 80 | 1.9 M | 37.2 FPS | 13 ms |
| YOLOv5s_v7.0 | 640×640 | 80 | 7.2 M | 20.9 FPS | 13 ms |
| YOLOv5x_v7.0 | 640×640 | 80 | 86.7 M | 3.6 FPS | 13 ms |
| YOLOv8n | 640×640 | 80 | 3.2 M | 34.1 FPS | 6 ms |
| YOLOv10n | 640×640 | 80 | 6.7 G | 18.1 FPS | 5 ms | 

## Model Data (detailed)
The detailed data of the model, including BPU frame delay, BPU throughput, post-processing time, mAP and other data under each thread, are in the README.md of the corresponding subfolder.



