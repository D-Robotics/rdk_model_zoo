# Semantic Segmentation

Semantic segmentation is a pixel-level classification task that assigns a class label to each pixel in an image without distinguishing between different instances of the same class.

- **Object Detection**: Distinguishes different instances and locates objects with bounding boxes.
- **Semantic Segmentation**: Distinguishes different classes and labels each pixel with a class mask (no instance separation).
- **Instance Segmentation**: A higher-level task combining object detection and semantic segmentation; distinguishes different instances and labels them with masks.

## Performance Data (Brief)
### RDK X5 & RDK X5 Module

#### Semantic Segmentation (VOC)

| Model         | Size (Pixels) | Classes | Parameters | BPU Throughput      | Post-processing Time (Python) |
| ------------- | ------------- | ------- | ---------- | ------------------------------------------------------------ | ----------------------------- |
| UNet-resnet50 | 512×512       | 20      | 43.93 M    | 13.23 | 267.08 ms                     |

### RDK X3 & RDK X3 Module

#### Semantic Segmentation (VOC)

| Model | Size (pixels) | Classes | Parameters | BPU Throughput | Post-processing Time (Python) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| UNet-resnet50 | 512×512 | 20 | 43.93 M | 5 FPS | 362 ms |

## Model Data (Detailed)
Detailed model data, including BPU frame latency, BPU throughput, post-processing time, and mIoU under various thread configurations, are available in the `README.md` of the corresponding subfolder.
