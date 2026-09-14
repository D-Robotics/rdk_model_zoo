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
    parser.add_argument('--model-path', type=str, default='models/fcos_512x512_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../resource/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='jupyter_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold.')
    parser.add_argument('--is-stride', type=bool, default=True, help='True: X5, False: X3')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), 
                        default=[8, 16, 32, 64, 128],
                        help='--anchors 8,16,32,64,128')
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化
    model = FCOS(opt.model_path, opt.conf_thres, opt.iou_thres, opt.classes_num, opt.strides, opt.is_stride)
    # 读图
    img = cv2.imread(opt.test_img)
    # 准备输入数据
    input_tensor = model.bgr2nv12(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    ids, scores, bboxes = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    for class_id, score, bbox in zip(ids, scores, bboxes):
        x1, y1, x2, y2 = bbox
        logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
    # 保存结果
    cv2.imwrite(opt.img_save_path, img)
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
        self.trans = [_ for _ in range(15)]

        # 反量化系数(如果有)
        self.scale_datas = [ output.properties.scale_data for output in self.quantize_model[0].outputs]
    
    def trans_outputs(self, order_we_want, len_outputs):
        for i in range(15):
            # 寻找输出shape的第几个符合order_we_want
            for j in range(15):
                h,w,c = self.quantize_model[0].outputs[j].properties.shape[1:]
                if h==order_we_want[i][0] and w==order_we_want[i][1] and c==order_we_want[i][2]:
                    self.trans[i] = j
                    break
        logger.info(f"trans: {self.trans}")

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


    # def c2numpy(self, outputs) -> list[np.array]:
    #     begin_time = time()
    #     outputs = [dnnTensor.buffer for dnnTensor in outputs]
    #     logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
    #     return outputs

    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        outputs = [outputs[self.trans[i]].buffer for i in range(len(outputs))]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

class FCOS(BaseModel):
    def __init__(self, 
                model_file: str, 
                conf: float, 
                iou: float,
                nc: int,
                strides: list,
                is_stride: bool,
                ):
        super().__init__(model_file)
        # 配置项目
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.strides = np.array(strides) 
        self.nl = len(strides)
        model_h, model_w = self.model_input_height, self.model_input_weight
        self.is_stride = is_stride

        # strides的grid网格, 只需要生成一次
        self.grids = []
        for stride in strides:
            h, w = model_h//stride, model_w//stride
            yv, xv = np.meshgrid(np.arange(h), np.arange(w))
            self.grids.append(((np.stack((yv, xv), 2) + 0.5) * stride).reshape(-1, 2))

        for stride, grid in zip(strides, self.grids):
            logger.info(f"stride {stride}: {grid.shape=}")    

        # 调整输出的顺序
        order_we_want = []
        for stride in strides:
            order_we_want.append([model_h//stride, model_w//stride, nc])
        for stride in strides:
            order_we_want.append([model_h//stride, model_w//stride, 4])
        for stride in strides:
            order_we_want.append([model_h//stride, model_w//stride, 1])
        self.trans_outputs(order_we_want, 15)

        # 准备反量化系数(如果有), 只需要准备一次
        # 反量化系数
        self.clses_scales, self.bboxes_scales, self.centers_scales = [], [], []
        for i in range(self.nl):
            if len(self.scale_datas[self.trans[i]])!=0:
                self.clses_scales.append(self.scale_datas[self.trans[i]][np.newaxis, :])
            else:
                self.clses_scales.append(None)

            if len(self.scale_datas[self.trans[i+5]])!=0:
                self.bboxes_scales.append(self.scale_datas[self.trans[i+5]][np.newaxis, :])
            else:
                self.bboxes_scales.append(None)

            if len(self.scale_datas[self.trans[i+10]])!=0:
                self.centers_scales.append(self.scale_datas[self.trans[i+10]][np.newaxis, :])
            else:
                self.centers_scales.append(None)

        for stride, clses_scale, bboxes_scale, centers_scale in zip(strides, self.clses_scales, self.bboxes_scales, self.centers_scales):
            if clses_scale is not None and bboxes_scale is not None and centers_scale is not None:
                logger.info(f"stride {stride}: {clses_scale.shape=}  {bboxes_scale.shape=}  {centers_scale.shape=}")
            else:
                logger.info(f"stride {stride}: Needn't Dequantized")

    # def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        clses = [outputs[i].reshape(-1, self.nc) for i in range(self.nl)]
        bboxes = [outputs[i+5].reshape(-1, 4) for i in range(self.nl)]
        centers = [outputs[i+10].reshape(-1, 1) for i in range(self.nl)]

        # classify: 利用numpy向量化操作完成阈值筛选 (优化版 2.0)
        scores, ids, indices = [], [], []
        for cls, center, clses_scale, centers_scale in zip(clses, centers, self.clses_scales, self.centers_scales):
            cls = cls if clses_scale is None else cls.astype(np.float32)*clses_scale
            center = center if centers_scale is None else center.astype(np.float32)*centers_scale
            raw_max_scores = np.max(cls, axis=1)
            max_scores = np.sqrt(1 / ((1 + np.exp(-center[:,0]))*(1 + np.exp(-raw_max_scores))))
            valid_indices = np.flatnonzero(max_scores >= self.conf)
            ids.append(np.argmax(cls[valid_indices, :], axis=1))
            scores.append(max_scores[valid_indices])
            indices.append(valid_indices)

        # 特征解码
        xyxys = []
        for indic, grid, stride, bbox, bboxes_scale in zip(indices, self.grids, self.strides, bboxes, self.bboxes_scales):
            grid_indices = grid[indic, :]
            bbox = bbox[indic, :] if bboxes_scale is None else bbox[indic, :].astype(np.float32)*bboxes_scale
            bbox = bbox*stride if self.is_stride else bbox
            x1y1 = grid_indices - bbox[:, 0:2]
            x2y2 = grid_indices + bbox[:, 2:4]
            xyxys.append(np.hstack([x1y1, x2y2]))

        # 大中小特征层阈值筛选结果拼接
        xyxy = np.concatenate([_ for _ in xyxys], axis=0)
        scores = np.concatenate([_ for _ in scores], axis=0)
        ids = np.concatenate([_ for _ in ids], axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(xyxy, scores, self.conf, self.iou)

        # 还原到原始的img尺度
        bboxes = (xyxy[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return ids[indices], scores[indices], bboxes


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
                   bbox,#: tuple[int, int, int, int],
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