#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.

import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API

from time import time
import argparse
import logging 

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/yolov8n_instance_seg_bayese_640x640_nv12_modified.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../resource/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='jupyter_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化
    model = YOLOv8_Seg(opt.model_path, opt.conf_thres, opt.iou_thres)
    # 读图
    img = cv2.imread(opt.test_img)
    # 准备输入数据
    input_tensor = model.bgr2nv12(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    ids, scores, bboxes, corpes, mces, protos = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    zeros = np.zeros((160,160,3), dtype=np.uint8)
    for class_id, score, bbox, corp, mc in zip(ids, scores, bboxes, corpes, mces):
        # detect
        x1, y1, x2, y2 = bbox
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
        # mask
        x1_corp, y1_corp, x2_corp, y2_corp = corp
        mask = (np.sum(mc[np.newaxis, np.newaxis, :]*protos[y1_corp:y2_corp,x1_corp:x2_corp,:], axis=2) > 0.0).astype(np.int32)
        zeros[y1_corp:y2_corp,x1_corp:x2_corp, :][mask == 1] = rdk_colors[class_id%20]
    # 保存结果
    img_h, img_w = img.shape[0:2]
    zeros = cv2.resize(zeros, (img_w, img_h),cv2.INTER_LANCZOS4)
    add_result = np.clip(img + 0.3*zeros, 0, 255).astype(np.uint8)
    cv2.imwrite(opt.img_save_path, np.hstack((img, zeros, add_result)))
    logger.info("\033[1;32m" + f"saved in path: \"./{opt.img_save_path}\"" + "\033[0m")

class BaseModel:
    def __init__(
        self,
        model_file: str
        ) -> None:
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_file))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]

    def resizer(self, img: np.ndarray)->np.ndarray:
        img_h, img_w = img.shape[0:2]
        self.y_scale, self.x_scale = img_h/self.model_input_height, img_w/self.model_input_weight
        return cv2.resize(img, (self.model_input_height, self.model_input_weight), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存
    
    def preprocess(self, img: np.ndarray)->np.array:
        """
        Preprocesses an input image to prepare it for model inference.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            np.array: The preprocessed image tensor in NCHW format ready for model input.

        Procedure:
            1. Resizes the image to a specified dimension (`input_image_size`) using nearest neighbor interpolation.
            2. Converts the image color space from BGR to RGB.
            3. Transposes the dimensions of the image tensor to channel-first order (CHW).
            4. Adds a batch dimension, thus conforming to the NCHW format expected by many models.
            Note: Normalization to [0, 1] is assumed to be handled elsewhere based on configuration.
        """
        begin_time = time()

        input_tensor = self.resizer(img)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        # input_tensor = np.array(input_tensor) / 255.0  # yaml文件中已经配置前处理
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # NCHW

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return input_tensor

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to the NV12 format.

        NV12 is a common video encoding format where the Y component (luminance) is full resolution,
        and the UV components (chrominance) are half-resolution and interleaved. This function first
        converts the BGR image to YUV 4:2:0 planar format, then rearranges the UV components to fit
        the NV12 format.

        Parameters:
        bgr_img (np.ndarray): The input BGR image array.

        Returns:
        np.ndarray: The converted NV12 format image array.
        """
        begin_time = time()
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed

        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12


    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs


    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

class YOLOv8_Seg(BaseModel):
    def __init__(self, 
                model_file: str, 
                conf: float, 
                iou: float
                ):
        super().__init__(model_file)
        # 将反量化系数准备好, 只需要准备一次
        # prepare the quantize scale, just need to generate once
        self.s_mces_scale = self.quantize_model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_mces_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.l_mces_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_mces_scale.shape=}, {self.m_mces_scale.shape=}, {self.l_mces_scale.shape=}")

        self.s_bboxes_scale = self.quantize_model[0].outputs[3].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[4].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[5].properties.scale_data[np.newaxis, :]
        logger.info(f"{self.s_bboxes_scale.shape=}, {self.m_bboxes_scale.shape=}, {self.l_bboxes_scale.shape=}")

        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)
        logger.info(f"{self.s_anchor.shape = }, {self.m_anchor.shape = }, {self.l_anchor.shape = }")

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.conf = conf
        self.iou = iou
        self.conf_inverse = -np.log(1/conf - 1)
        logger.info("iou threshol = %.2f, conf threshol = %.2f"%(iou, conf))
        logger.info("sigmoid_inverse threshol = %.2f"%self.conf_inverse)
    
        # mask 切片 corp 缩放系数
        self.x_scale_corp, self.y_scale_corp = 160/self.model_input_height, 160/self.model_input_weight
    

    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        begin_time = time()

        # reshape
        s_mces = outputs[0].reshape(-1, 32)
        m_mces = outputs[1].reshape(-1, 32)
        l_mces = outputs[2].reshape(-1, 32)
        s_bboxes = outputs[3].reshape(-1, 64)
        m_bboxes = outputs[4].reshape(-1, 64)
        l_bboxes = outputs[5].reshape(-1, 64)
        s_clses = outputs[6].reshape(-1, 80)
        m_clses = outputs[7].reshape(-1, 80)
        l_clses = outputs[8].reshape(-1, 80)
        protos = outputs[9][0,:,:,:]

        # classify: 利用numpy向量化操作完成阈值筛选 (优化版 2.0)=
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy) transpose
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

        s_mces_float32 = (s_mces[s_valid_indices,:].astype(np.float32) * self.s_mces_scale)
        m_mces_float32 = (m_mces[m_valid_indices,:].astype(np.float32) * self.m_mces_scale)
        l_mces_float32 = (l_mces[l_valid_indices,:].astype(np.float32) * self.l_mces_scale)

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        mces = np.concatenate((s_mces_float32, m_mces_float32, l_mces_float32), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)

        # 还原到原始的尺度
        bboxes = (dbboxes[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)
        scores = scores[indices]
        ids = ids[indices]
        corpes = (dbboxes[indices] * np.array([self.x_scale_corp, self.y_scale_corp, self.x_scale_corp, self.y_scale_corp])).astype(np.int32)
        mces = mces[indices]

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return ids, scores, bboxes, corpes, mces, protos


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

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img: np.array, 
                   bbox: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()