#!/user/bin/env python

# YOLO26 Pose Inference Script for RDK X5 (BPU Optimized)
# Clean Version: Strict NHWC Layout

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
logger = logging.getLogger("YOLO26-Pose")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26n_pose_bayese_640x640_nv12.bin', help="Path to YOLO26 Pose *.bin Model.") 
    parser.add_argument('--test-img', type=str, default='bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='result_pose.jpg', help='Path to Save Result Image.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='Keypoint confidence threshold.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1. Instantiate Inference Engine
    engine = YOLO26PoseEngine(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres,
        kpt_conf_thres=opt.kpt_conf_thres
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


class YOLO26PoseEngine:
    def __init__(self, model_path, score_thres=0.25, nms_thres=0.7, kpt_conf_thres=0.5):
        self.score_thres = score_thres
        self.nms_thres = nms_thres
        self.kpt_conf_thres = kpt_conf_thres
        self.conf_thres_raw = -np.log(1/score_thres - 1)
        
        # Load Model
        t0 = time()
        self.models = dnn.load(model_path)
        logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")
        
        # Get Input Info
        input_shape = self.models[0].inputs[0].properties.shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]
        
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

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
        
        # Organize outputs by Stride
        features = {} # Key: stride, Value: dict of data
        
        for i, out in enumerate(outputs):
            shape = self.models[0].outputs[i].properties.shape
            # Strict NHWC: Index 1=H, 2=W, 3=C
            h, w, c = shape[1], shape[2], shape[3]
            stride = self.input_h // h
            
            if stride not in features: features[stride] = {}
            
            if c == 4:
                features[stride]['box'] = out.buffer.reshape(-1, 4)
            elif c == 80 or c == 1:
                features[stride]['cls'] = out.buffer.reshape(-1, c)
            elif c == 51:
                features[stride]['kpt'] = out.buffer.reshape(-1, 17, 3)

        detections = []
        for stride, feats in features.items():
            if 'box' not in feats or 'cls' not in feats or 'kpt' not in feats:
                continue

            box_data = feats['box']
            cls_data = feats['cls']
            kpt_data = feats['kpt']
            grid_size = self.input_h // stride

            # 1. Filter
            scores = cls_data[:, 0]
            valid_mask = scores >= self.conf_thres_raw
            if not np.any(valid_mask): continue

            # 2. Decode
            v_scores = sigmoid(scores[valid_mask])
            v_box = box_data[valid_mask]
            v_kpts = kpt_data[valid_mask]
            
            gv, gu = np.indices((grid_size, grid_size))
            grid_y = gv.reshape(-1)[valid_mask]
            grid_x = gu.reshape(-1)[valid_mask]
            
            # Box: (grid + 0.5 +/- dist) * stride
            x1 = (grid_x + 0.5 - v_box[:, 0]) * stride
            y1 = (grid_y + 0.5 - v_box[:, 1]) * stride
            x2 = (grid_x + 0.5 + v_box[:, 2]) * stride
            y2 = (grid_y + 0.5 + v_box[:, 3]) * stride
            xyxy = np.stack([x1, y1, x2, y2], axis=-1)
            
            # Kpts: (raw + grid + 0.5) * stride
            kpt_x = (v_kpts[:, :, 0] + grid_x[:, None] + 0.5) * stride
            kpt_y = (v_kpts[:, :, 1] + grid_y[:, None] + 0.5) * stride
            kpt_conf = sigmoid(v_kpts[:, :, 2])
            decoded_kpts = np.stack([kpt_x, kpt_y, kpt_conf], axis=-1)

            for box, score, kpts in zip(xyxy, v_scores, decoded_kpts):
                detections.append({'box': box, 'score': score, 'kpts': kpts})

        # 3. NMS
        final_res = []
        if detections:
            boxes_np = np.array([d['box'] for d in detections])
            scores_np = np.array([d['score'] for d in detections])
            xywh = boxes_np.copy()
            xywh[:, 2:] -= xywh[:, :2]
            
            indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores_np.tolist(), self.score_thres, self.nms_thres)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    d = detections[i]
                    final_res.append({
                        'box': (d['box'] / scale).astype(int),
                        'score': d['score'],
                        'kpts': d['kpts'] / [scale, scale, 1.0]
                    })
        
        logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")
        return final_res

    def draw_and_save(self, img, results, save_path):
        logger.info(f"\033[1;32mDraw Results ({len(results)}): \033[0m")
        for res in results:
            box, score, kpts = res['box'], res['score'], res['kpts']
            logger.info(f"({box[0]}, {box[1]}, {box[2]}, {box[3]}) -> person: {score:.2f}")
            
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, f"person:{score:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            for x, y, conf in kpts:
                if conf < self.kpt_conf_thres: continue
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
            for sk in self.skeleton:
                p1, p2 = kpts[sk[0]-1], kpts[sk[1]-1]
                if p1[2] < self.kpt_conf_thres or p2[2] < self.kpt_conf_thres: continue
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)

        cv2.imwrite(save_path, img)
        logger.info(f"Saved to {save_path}")

if __name__ == "__main__":
    main()