#!/user/bin/env python3

# YOLO26 Inference Script for RDK X5
# 2026-01-14 Version

import os
import cv2
import numpy as np
from time import time
import argparse
import logging 

# hobot_dnn
try:
    from hobot_dnn import pyeasy_dnn as dnn
except ImportError:
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26n_cut_bayese_640x640_nv12.bin', help="Path to YOLO26 *.bin Model.") 
    parser.add_argument('--test-img', type=str, default='bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='result_yolo26.jpg', help='Path to Save Result Image.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1. Load Model
    t0 = time()
    models = dnn.load(opt.model_path)
    logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 2. Get Model Info
    input_shape = models[0].inputs[0].properties.shape # NCHW or NHWC
    # Auto detect model input size
    if input_shape[3] == 3: # NHWC
        m_h, m_w = input_shape[1], input_shape[2]
    else: # NCHW
        m_h, m_w = input_shape[2], input_shape[3]
    
    # 3. Pre-process (BGR -> NV12)
    t0 = time()
    img = cv2.imread(opt.test_img)
    if img is None: return
    orig_h, orig_w = img.shape[:2]
    scale = min(m_h / orig_h, m_w / orig_w)
    
    # Resize & Pad (Left-top align)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    input_tensor = cv2.copyMakeBorder(img_resized, 0, m_h - new_h, 0, m_w - new_w, cv2.BORDER_CONSTANT, value=127)
    
    # Convert to NV12
    yuv = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2YUV_I420).flatten()
    nv12 = np.zeros((m_h * m_w * 3 // 2,), dtype=np.uint8)
    y_size = m_h * m_w
    nv12[:y_size] = yuv[:y_size]
    nv12[y_size::2] = yuv[y_size : y_size + y_size//4] # U
    nv12[y_size+1::2] = yuv[y_size + y_size//4 : ] # V
    logger.debug(f"\033[1;31mPre-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 4. Forward
    t0 = time()
    outputs = models[0].forward(nv12)
    logger.debug(f"\033[1;31mForward time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 5. Post-process (YOLO26 Optimized)
    t0 = time()
    strides = [8, 16, 32]
    conf_raw = -np.log(1/opt.score_thres - 1)
    detections = []

    # Map outputs by shape (assuming NHWC)
    # output properties shape is logical (NCHW), but buffer is physical (NHWC)
    output_items = []
    for i, out in enumerate(outputs):
        output_items.append({'data': out.buffer, 'shape': models[0].outputs[i].properties.shape})

    for stride in strides:
        grid_size = m_h // stride # 80, 40, 20
        
        # Match Box(4) and Cls(80)
        box_data, cls_data = None, None
        for item in output_items:
            _, h, w, c = item['shape'] # logical NCHW
            # Due to physical NHWC, we use size to verify
            if h == grid_size and w == grid_size:
                if c == 4: box_data = item['data'].reshape(-1, 4)
                elif c == 80: cls_data = item['data'].reshape(-1, 80)
        
        if box_data is None: continue

        # Filtering
        max_scores = np.max(cls_data, axis=1)
        valid_mask = max_scores >= conf_raw
        if not np.any(valid_mask): continue

        # Decode
        v_box = box_data[valid_mask]
        v_score = 1 / (1 + np.exp(-max_scores[valid_mask]))
        v_id = np.argmax(cls_data[valid_mask], axis=1)
        
        # Grid Generation
        gv, gu = np.indices((grid_size, grid_size))
        grid = np.stack((gu, gv), axis=-1).reshape(-1, 2)[valid_mask] + 0.5

        # ltrb to xyxy: (grid - dist)*stride, (grid + dist)*stride
        xyxy = np.hstack([(grid - v_box[:, :2]), (grid + v_box[:, 2:])]) * stride
        
        for box, s, cid in zip(xyxy, v_score, v_id):
            detections.append([*box, s, cid])

    # NMS & Render
    final_res = []
    if detections:
        dets = np.array(detections)
        # xyxy to xywh for NMS
        xywh = dets[:, :4].copy()
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), dets[:, 4].tolist(), opt.score_thres, opt.nms_thres)
        
        for i in indices.flatten():
            d = dets[i]
            x1, y1, x2, y2 = (d[:4] / scale).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            final_res.append((int(d[5]), d[4], x1, y1, x2, y2))

    logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")

    # 6. Draw
    logger.info(f"\033[1;32mDraw Results ({len(final_res)}): \033[0m")
    for cid, s, x1, y1, x2, y2 in final_res:
        name = coco_names[cid] if cid < len(coco_names) else str(cid)
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {s:.2f}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name}:{s:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(opt.img_save_path, img)
    logger.info(f"Saved to {opt.img_save_path}")

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

if __name__ == "__main__":
    main()
