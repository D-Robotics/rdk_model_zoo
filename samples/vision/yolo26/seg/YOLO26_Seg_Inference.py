#!/user/bin/env python

# YOLO26 Instance Segmentation Inference Script for RDK X5
# 2026-01-15 Clean Version

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
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
    except ImportError:
        print("Error: hobot_dnn not found. Please install it on RDK board.")
        exit(1)

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26-Seg")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26n_seg_bayese_640x640_nv12.bin', help="Path to *.bin Model.") 
    parser.add_argument('--test-img', type=str, default='bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='result_seg.jpg', help='Path to Save Result Image.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1. Init
    engine = YOLO26SegEngine(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )

    # 2. Load Image
    img = cv2.imread(opt.test_img)
    if img is None: return
    
    # 3. Preprocess
    input_tensor, scale_info = engine.preprocess(img)

    # 4. Inference
    outputs = engine.forward(input_tensor)

    # 5. Postprocess
    results = engine.post_process(outputs, scale_info)

    # 6. Draw & Save
    engine.draw_and_save(img, results, opt.img_save_path)


class YOLO26SegEngine:
    def __init__(self, model_path, score_thres=0.25, nms_thres=0.7):
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.conf_thres_raw = -np.log(1/score_thres - 1)
        
        t0 = time()
        self.models = dnn.load(model_path)
        logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")
        
        input_shape = self.models[0].inputs[0].properties.shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]
        
        self.strides = [8, 16, 32]
        self.proto_h, self.input_proto_w = 160, 160 # Standard YOLOv8/26 Seg Proto size

    def preprocess(self, img):
        t0 = time()
        orig_h, orig_w = img.shape[:2]
        scale = min(self.input_h / orig_h, self.input_w / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        input_tensor = cv2.copyMakeBorder(img_resized, 0, self.input_h - new_h, 0, self.input_w - new_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        yuv = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.zeros((self.input_h * self.input_w * 3 // 2,), dtype=np.uint8)
        y_size = self.input_h * self.input_w
        nv12[:y_size] = yuv[:y_size]
        nv12[y_size::2] = yuv[y_size : y_size + y_size//4]
        nv12[y_size+1::2] = yuv[y_size + y_size//4 : ]
        
        logger.debug(f"\033[1;31mPre-process time = {(time() - t0)*1000:.2f} ms\033[0m")
        return nv12, (scale, 0, 0)

    def forward(self, input_tensor):
        t0 = time()
        outputs = self.models[0].forward(input_tensor)
        logger.debug(f"\033[1;31mForward time = {(time() - t0)*1000:.2f} ms\033[0m")
        return outputs

    def post_process(self, outputs, scale_info):
        t0 = time()
        scale, _, _ = scale_info
        
        # 1. Organize outputs by Stride and find Proto
        features = {} 
        protos = None
        
        for i, out in enumerate(outputs):
            shape = self.models[0].outputs[i].properties.shape
            if len(shape) != 4: continue
            
            h, w, c = shape[1], shape[2], shape[3] # NHWC
            
            if h == 160 and w == 160 and c == 32:
                protos = out.buffer.reshape(160, 160, 32)
                continue
                
            stride = self.input_h // h
            if stride not in features: features[stride] = {}
            
            if c == 4: features[stride]['box'] = out.buffer.reshape(-1, 4)
            elif c == 80: features[stride]['cls'] = out.buffer.reshape(-1, 80)
            elif c == 32: features[stride]['mc'] = out.buffer.reshape(-1, 32)

        if protos is None:
            logger.error("Proto head not found in outputs!")
            return []

        # 2. Decode Candidates
        detections = []
        for stride, feats in features.items():
            if 'box' not in feats or 'cls' not in feats or 'mc' not in feats: continue

            box_data, cls_data, mc_data = feats['box'], feats['cls'], feats['mc']
            grid_size = self.input_h // stride

            scores = np.max(cls_data, axis=1)
            valid_mask = scores >= self.conf_thres_raw
            if not np.any(valid_mask): continue

            v_scores = sigmoid(scores[valid_mask])
            v_ids = np.argmax(cls_data[valid_mask], axis=1)
            v_box = box_data[valid_mask]
            v_mc = mc_data[valid_mask]
            
            gv, gu = np.indices((grid_size, grid_size))
            grid_y, grid_x = gv.reshape(-1)[valid_mask], gu.reshape(-1)[valid_mask]
            
            x1 = (grid_x + 0.5 - v_box[:, 0]) * stride
            y1 = (grid_y + 0.5 - v_box[:, 1]) * stride
            x2 = (grid_x + 0.5 + v_box[:, 2]) * stride
            y2 = (grid_y + 0.5 + v_box[:, 3]) * stride
            xyxy = np.stack([x1, y1, x2, y2], axis=-1)

            for box, score, cid, mc in zip(xyxy, v_scores, v_ids, v_mc):
                detections.append({'box': box, 'score': score, 'id': cid, 'mc': mc})

        # 3. NMS
        final_res = []
        if detections:
            boxes_np = np.array([d['box'] for d in detections])
            scores_np = np.array([d['score'] for d in detections])
            xywh = boxes_np.copy()
            xywh[:, 2:] -= xywh[:, :2]
            indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores_np.tolist(), self.score_thres, self.nms_thres)
            
            # Mask Crop Scales
            # Proto is 160x160 for 640x640 input -> 1/4 scale
            mask_h, mask_w = 160, 160
            scale_p = mask_h / self.input_h # 0.25

            if len(indices) > 0:
                for i in indices.flatten():
                    det = detections[i]
                    box = det['box']
                    
                    # Compute Mask
                    # mask = mc (1, 32) @ protos (160, 160, 32).T -> (160, 160)
                    mc = det['mc']
                    full_mask = sigmoid(np.dot(protos, mc))
                    
                    # Crop mask to Box (in proto scale)
                    px1, py1, px2, py2 = (box * scale_p).astype(int)
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(mask_w, px2), min(mask_h, py2)
                    
                    if px2 > px1 and py2 > py1:
                        # Extract the box area from the mask
                        cropped_mask = full_mask[py1:py2, px1:px2]
                        
                        # Morphology Open to remove noise (optional but recommended)
                        # Need to threshold temporarily for morphology, or use on float? 
                        # Morphology usually works on binary. Let's do it on float carefully or skip if complex.
                        # Actually, better to do morphology AFTER resizing or on binary.
                        # Reference script does it after resize. Let's follow that.
                        
                        mask_result = cropped_mask
                    else:
                        mask_result = np.zeros((0,0), dtype=np.float32)

                    final_res.append({
                        'box': (box / scale).astype(int),
                        'score': det['score'],
                        'id': det['id'],
                        'mask': mask_result,
                        'mask_box': [px1, py1, px2, py2]
                    })
        
        logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")
        return final_res

    def draw_and_save(self, img, results, save_path):
        logger.info(f"\033[1;32mDraw Results ({len(results)}): \033[0m")
        
        overlay = img.copy()
        img_h, img_w = img.shape[:2]
        
        # Kernel for morphology
        morph_kernel = np.ones((3,3), np.uint8)
        
        for res in results:
            box, score, cid, mask = res['box'], res['score'], res['id'], res['mask']
            
            x1 = max(0, min(box[0], img_w))
            y1 = max(0, min(box[1], img_h))
            x2 = max(0, min(box[2], img_w))
            y2 = max(0, min(box[3], img_h))
            
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue

            name = coco_names[cid] if cid < len(coco_names) else str(cid)
            logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {score:.2f}")
            
            color = rdk_colors[cid % 20]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name}:{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if mask.size > 0:
                roi = overlay[y1:y2, x1:x2]
                
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    # 1. Resize using LANCZOS4 for high quality
                    prob_mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    
                    # 2. Thresholding
                    binary_mask = (prob_mask > 0.5).astype(np.uint8)
                    
                    # 3. Morphology Open to clean up edges/noise
                    # This smooths the binary mask boundaries
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, morph_kernel)
                    
                    roi[binary_mask == 1] = color

        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.imwrite(save_path, img)
        logger.info(f"Saved to {save_path}")

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
rdk_colors = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

if __name__ == "__main__":
    main()
