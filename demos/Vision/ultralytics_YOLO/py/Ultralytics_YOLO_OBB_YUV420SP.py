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

import os
import cv2
import numpy as np
# scipy
try:
    from scipy.special import softmax
except:
    print("scipy is  not installed, installing.")
    os.system("pip install scipy")
    from scipy.special import softmax

# hobot_dnn
try:
    from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
except:
    print("Your python environment is not ready, please use system python3 to run this program.")
    exit()


from time import time
import argparse
import logging 
import math

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")


def rotate_point(px, py, cx, cy, angle):
    """
    旋转一个点(px, py)围绕中心(cx, cy)按指定的角度(弧度制)
    :param px: 点x坐标
    :param py: 点y坐标
    :param cx: 中心x坐标
    :param cy: 中心y坐标
    :param angle: 旋转角度，弧度制
    :return: 旋转后的新坐标(x', y')
    """
    # 将点平移到原点
    translated_x = px - cx
    translated_y = py - cy
    # 应用旋转
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_x = translated_x * cos_angle - translated_y * sin_angle
    rotated_y = translated_x * sin_angle + translated_y * cos_angle
    # 平移回原来的位置
    final_x = rotated_x + cx
    final_y = rotated_y + cy
    return final_x, final_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_bin_models/obb/yolo11n_obb_detect_bayese_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../resource/datasets/DOTA/asset/P0009.png', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='py_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=15, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), 
                        default=[8, 16 ,32],
                        help='--strides 8, 16, 32')
    opt = parser.parse_args()
    logger.info(opt)

    # quick demo
    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. downloading ...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_obb_detect_bayese_640x640_nv12.bin")
        opt.model_path = 'yolo11n_obb_detect_bayese_640x640_nv12.bin'

    # 实例化
    model = Ultralytics_YOLO_Detect_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,   # default: 80
        nms_thres=opt.nms_thres,       # default: 0.7
        score_thres=opt.score_thres,   # default: 0.25
        reg=opt.reg,                   # default: 16
        strides=opt.strides            # default: [8, 16, 32]
        )
    # 读图
    img = cv2.imread(opt.test_img)
    if img is None:
        raise ValueError(f"Load image failed: {opt.test_img}")
        exit()
    # 准备输入数据
    input_tensor = model.preprocess_yuv420sp(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    results = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    for class_id, score, x, y, w, h, t in results:
        print(class_id, score, x, y, w, h, t)
        rect_points = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        rotated_points = []
        for point in rect_points:
            rotated_points.append(rotate_point(point[0], point[1], 0, 0, t))
        rotated_points = [(point[0] + x, point[1] + y) for point in rotated_points]
        for i in range(len(rotated_points)):
            pt1 = (int(rotated_points[i][0]), int(rotated_points[i][1]))
            pt2 = (int(rotated_points[(i+1)%len(rotated_points)][0]), int(rotated_points[(i+1)%len(rotated_points)][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(opt.img_save_path, img)
    logger.info("\033[1;32m" + f"saved in path: \"./{opt.img_save_path}\"" + "\033[0m")


class Ultralytics_YOLO_Detect_Bayese_YUV420SP():
    def __init__(self, model_path, classes_num, nms_thres, score_thres, reg, strides):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        # init
        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        self.strides = strides
        logger.info(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)
        logger.info(f"{self.input_H = }, {self.input_W = }")
        logger.info(f"{self.strides = }")

        # DFL求期望的系数, 只需要生成一次
        # DFL calculates the expected coefficients, which only needs to be generated once.
        self.weights_static = np.array([i for i in range(reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        # anchors, 只需要生成一次
        self.grids = []
        for stride in self.strides:
            assert self.input_H % stride == 0, f"{stride=}, {self.input_H=}: input_H % stride != 0"
            assert self.input_W % stride == 0, f"{stride=}, {self.input_W=}: input_W % stride != 0"
            grid_H, grid_W = self.input_H // stride, self.input_W // stride
            self.grids.append(np.stack([np.tile(np.linspace(0.5, grid_H-0.5, grid_H), reps=grid_H), 
                            np.repeat(np.arange(0.5, grid_W+0.5, 1), grid_W)], axis=0).transpose(1,0))
            logger.info(f"{self.grids[-1].shape = }")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            # 利用resize的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存节约一次
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            # 利用 letter box 的方式进行前处理, 准备nv12的输入数据
            begin_time = time()
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")
            
            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift
            
            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift
            
            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
            input_tensor = self.bgr2nv12(input_tensor)
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
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

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        # reshape
        clses = [outputs[0].reshape(-1, self.CLASSES_NUM), outputs[3].reshape(-1, self.CLASSES_NUM), outputs[6].reshape(-1, self.CLASSES_NUM)]
        bboxes = [outputs[1].reshape(-1, self.REG * 4), outputs[4].reshape(-1, self.REG * 4), outputs[7].reshape(-1, self.REG * 4)]
        logit_angles = [outputs[2].reshape(-1), outputs[5].reshape(-1), outputs[8].reshape(-1)]
        
        dbboxes, ids, scores, rotate_bboxes = [], [], [], []
        for cls, bbox, stride, grid, logit_angle in zip(clses, bboxes, self.strides, self.grids, logit_angles):    
            # score 筛选
            max_scores = np.max(cls, axis=1)
            bbox_selected = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            ids.append(np.argmax(cls[bbox_selected, : ], axis=1))
            # 3个Classify分类分支：Sigmoid计算 
            scores.append(1 / (1 + np.exp(-max_scores[bbox_selected])))
            # dist2bbox (ltrb2xyxy)
            ltrb_selected = np.sum(softmax(bbox[bbox_selected,:].reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
            grid_selected = grid[bbox_selected, :]
            x1y1 = grid_selected - ltrb_selected[:, 0:2]
            x2y2 = grid_selected + ltrb_selected[:, 2:4]
            center_xy = (x1y1 + x2y2) / 2
            wh = (x2y2 - x1y1)
            angle = (1 / (1 + np.exp(-logit_angle[bbox_selected])) - 0.25)* math.pi  # [, pi]
            angle = np.clip(angle, 0, math.pi)
            dbboxes.append(np.hstack([center_xy * stride, wh * stride, angle[:, np.newaxis]]))
            # dbboxes.append(np.hstack([x1y1 * stride, x2y2 * stride, angle[:, np.newaxis]]))

        dbboxes = np.concatenate((dbboxes), axis=0).astype(np.float32)
        scores = np.concatenate((scores), axis=0)
        ids = np.concatenate((ids), axis=0)

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            if not np.any(id_indices):  # 没有该类别的框，跳过
                continue
            cv2_bboxs, cv2_scores = [], []
            for cnt, index in enumerate(id_indices):
                # print(f"{index = }")
                # print(f"{dbboxes[index, 0].shape = }")
                if not index:
                    continue
                cv2_bboxs.append(((float(dbboxes[cnt, 0]), float(dbboxes[cnt, 1])), 
                                  (float(dbboxes[cnt, 2]), float(dbboxes[cnt, 3])), 
                                  float(dbboxes[cnt, 4])))
                cv2_scores.append(float(scores[cnt]))
            # print(f"{len(cv2_bboxs) = }, {type(cv2_bboxs) = }")
            # print(f"{len(cv2_scores) = }, {type(cv2_scores) = }")
            indices = cv2.dnn.NMSBoxesRotated(cv2_bboxs, cv2_scores, self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x,y,w,h,t = dbboxes[id_indices,:][indic]
                x = int((x - self.x_shift) / self.x_scale)
                y = int((y - self.y_shift) / self.y_scale)
                w = int(w / self.x_scale)
                h = int(h / self.y_scale)

                results.append((i, scores[id_indices][indic], x, y, w, h, t))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

dotav1_names = [
    'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field',
    'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool'
    ]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img, bbox, score, class_id) -> None:
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
    label = f"{dotav1_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()