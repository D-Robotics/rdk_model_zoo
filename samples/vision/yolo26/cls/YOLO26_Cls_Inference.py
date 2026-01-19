#!/user/bin/env python

# YOLO26 Classification Inference Script for RDK X5 (BPU Optimized)
# Input: 224x224 NV12
# Output: Top-5 Classes

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
logger = logging.getLogger("YOLO26-Cls")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='yolo26n_cls_bayese_224x224_nv12.bin', help="Path to *.bin Model.") 
    parser.add_argument('--test-img', type=str, default='bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--topk', type=int, default=5, help='Top K results to show.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 1. Init
    engine = YOLO26ClsEngine(opt.model_path)
    
    # Load Labels (Optional)
    labels = {}
    label_path = os.path.join(os.path.dirname(__file__), 'imagenet1000_clsidx_to_labels.txt')
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Try eval if it's a python dict string
                if content.startswith('{') and content.endswith('}'):
                    labels = eval(content)
                else:
                    # Assume line by line
                    for i, line in enumerate(content.split('\n')):
                        labels[i] = line.strip()
            logger.info(f"Loaded {len(labels)} labels from {label_path}")
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}")

    # 2. Load Image
    img = cv2.imread(opt.test_img)
    if img is None: 
        logger.error(f"Failed to load image: {opt.test_img}")
        return

    # 3. Preprocess
    input_tensor = engine.preprocess(img)

    # 4. Inference
    outputs = engine.forward(input_tensor)

    # 5. Postprocess
    results = engine.post_process(outputs, topk=opt.topk)

    # 6. Show Results
    logger.info(f"\033[1;32mTop-{opt.topk} Results:\033[0m")
    for i, (cid, score) in enumerate(results):
        name = labels.get(cid, str(cid))
        logger.info(f"Rank {i+1}: Class {cid} ({name}) | Score: {score:.4f}")

class YOLO26ClsEngine:
    def __init__(self, model_path):
        t0 = time()
        self.models = dnn.load(model_path)
        logger.debug(f"\033[1;31mLoad model time = {(time() - t0)*1000:.2f} ms\033[0m")
        
        # Input Shape (Batch, H, W, C) or (Batch, C, H, W)
        # Usually index 2,3 for H,W in properties (logical NCHW)
        input_shape = self.models[0].inputs[0].properties.shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]
        logger.info(f"Model Input Shape: {self.input_h}x{self.input_w}")

    def preprocess(self, img):
        t0 = time()
        # Classification usually just Resizes (no padding needed if aspect ratio is close, 
        # or CenterCrop. Here we use simple Resize to 224x224 for simplicity and speed)
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        
        # BGR -> NV12
        yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV_I420).flatten()
        nv12 = np.zeros((self.input_h * self.input_w * 3 // 2,), dtype=np.uint8)
        y_size = self.input_h * self.input_w
        nv12[:y_size] = yuv[:y_size]
        nv12[y_size::2] = yuv[y_size : y_size + y_size//4]
        nv12[y_size+1::2] = yuv[y_size + y_size//4 : ]
        
        logger.debug(f"\033[1;31mPre-process time = {(time() - t0)*1000:.2f} ms\033[0m")
        return nv12

    def forward(self, input_tensor):
        t0 = time()
        outputs = self.models[0].forward(input_tensor)
        logger.debug(f"\033[1;31mForward time = {(time() - t0)*1000:.2f} ms\033[0m")
        return outputs

    def post_process(self, outputs, topk=5):
        t0 = time()
        
        # Cls model has 1 output: (1, 1000)
        # buffer size should be 1000 * 4 (float32)
        logits = outputs[0].buffer.reshape(-1) # Flatten to (1000,)
        
        # Softmax
        probs = softmax(logits)
        
        # TopK
        top_indices = np.argsort(probs)[::-1][:topk]
        top_scores = probs[top_indices]
        
        results = list(zip(top_indices, top_scores))
        
        logger.debug(f"\033[1;31mPost-process time = {(time() - t0)*1000:.2f} ms\033[0m")
        return results

if __name__ == "__main__":
    main()
