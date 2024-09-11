# Instance segmentation
Instance segmentation is a higher-level task that combines object detection and semantic segmentation.

Object detection: distinguish different instances and locate the object with box.
Semantic segmentation: distinguish different classes and label them with masks.
Instance segmentation: distinguish different instances and label them with masks.

## Performance Data (in brief)
### RDK X5 & RDK X5 Module
### Instance Segmentation (COCO)
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv8n-seg | 640×640 | 80 | 3.4 M  | 175.3 FPS | 6 ms |
| YOLOv8s-seg | 640×640 | 80 | 11.8 M | 67.7 FPS | 6 ms |
| YOLOv8m-seg | 640×640 | 80 | 27.3 M | 27.0 FPS | 6 ms |
| YOLOv8l-seg | 640×640 | 80 | 46.0 M | 14.4 FPS | 6 ms |
| YOLOv8x-seg | 640×640 | 80 | 71.8 M | 8.9 FPS | 6 ms |


### RDK X3 & RDK X3 Module
Instance Segmentation (COCO) Instance Segmentation
| model (public) | size (pixels) | number of classes | number of parameters | BPU throughput | post-processing time (Python) |
|---------|---------|-------|---------|---------|----------|
| YOLOv8n-seg | 640×640 | 80 | 3.4 M | 27.3 FPS | 6 ms |

## Model Data (detailed)
The detailed data of the model, including BPU frame delay, BPU throughput, post-processing time, mAP and other data under each thread, are in the README.md of the corresponding subfolder.