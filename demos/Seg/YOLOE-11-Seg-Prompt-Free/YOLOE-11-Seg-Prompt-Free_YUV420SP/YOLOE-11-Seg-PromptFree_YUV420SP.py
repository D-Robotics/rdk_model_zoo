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
    parser.add_argument('--model-path', type=str, default='ptq_models/yoloe-11s-seg-pf_bayese_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../../datasets/coco/assets/bus.jpg', help='Path to Load Test Image.')

    parser.add_argument('--img-save-path', type=str, default='py_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=4585, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--mc', type=int, default=32, help='Mask Coefficients')
    parser.add_argument('--is-open', type=bool, default=False, help='Ture: morphologyEx')
    parser.add_argument('--is-point', type=bool, default=False, help='Ture: Draw edge points')
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化
    model = YOLO11_Seg(opt)
    # 读图
    img = cv2.imread(opt.test_img)
    # 准备输入数据
    input_tensor = model.preprocess_yuv420sp(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    results = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    # 绘制
    draw_img = img.copy()
    zeros = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    for class_id, score, x1, y1, x2, y2, mask in results:
        # Detect
        print("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(draw_img, (x1, y1, x2, y2), score, class_id)
        # Instance Segment
        if mask.size == 0:
            continue
        mask = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_LANCZOS4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, model.kernel_for_morphologyEx, 1) if opt.is_open else mask      
        zeros[y1:y2,x1:x2, :][mask == 1] = rdk_colors[(class_id-1)%20]
        # points
        if not opt.is_point:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 手动连接轮廓
            contour = np.vstack((contours[0], np.array([contours[0][0]])))
            for i in range(1, len(contours)):
                contour = np.vstack((contour, contours[i], np.array([contours[i][0]])))
            # 轮廓投射回原来的图像大小
            merged_points = contour[:,0,:]
            merged_points[:,0] = merged_points[:,0] + x1
            merged_points[:,1] = merged_points[:,1] + y1
            points = np.array([[[int(x), int(y)] for x, y in merged_points]], dtype=np.int32)
            # 绘制轮廓
            cv2.polylines(draw_img, points, isClosed=True, color=rdk_colors[(class_id-1)%20], thickness=4)
    
    # 可视化, 这里采用直接相加的方式，实际应用中对Mask一般不需要Resize这些操作
    add_result = np.clip(draw_img + 0.3*zeros, 0, 255).astype(np.uint8)
    # 保存结果
    cv2.imwrite(opt.img_save_path, np.hstack((draw_img, zeros, add_result)))
    logger.info("\033[1;32m" + f"saved in path: \"./{opt.img_save_path}\"" + "\033[0m")


class YOLO11_Seg():
    def __init__(self, opt):
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(opt.model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(opt.model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")


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
        self.SCORE_THRESHOLD = opt.score_thres
        self.NMS_THRESHOLD = opt.nms_thres
        self.CONF_THRES_RAW = -np.log(1/self.SCORE_THRESHOLD - 1)
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f"%(self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f"%self.CONF_THRES_RAW)

        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        logger.info(f"{self.input_H = }, {self.input_W = }")

        self.Mask_H, self.Mask_W = 160, 160
        self.x_scale_corp = self.Mask_W / self.input_W
        self.y_scale_corp = self.Mask_H / self.input_H
        logger.info(f"{self.Mask_H = }   {self.Mask_W = }")
        logger.info(f"{self.x_scale_corp = }, {self.y_scale_corp = }")

        self.REG = opt.reg
        logger.info(f"{self.REG = }")

        self.CLASSES_NUM = opt.classes_num
        logger.info(f"{self.CLASSES_NUM = }")

        self.MCES_NUM = opt.mc
        logger.info(f"{self.MCES_NUM = }")

        self.IS_OPEN = opt.is_open  # 是否对Mask进行形态学开运算
        self.kernel_for_morphologyEx = np.ones((5,5), np.uint8) 
        logger.info(f"{self.IS_OPEN = }   {self.kernel_for_morphologyEx = }")

        self.IS_POINT = opt.is_point # 是否绘制边缘点
        logger.info(f"{self.IS_POINT = }")

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
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        s_mces = outputs[2].reshape(-1, self.MCES_NUM)

        m_clses = outputs[3].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[4].reshape(-1, self.REG * 4)
        m_mces = outputs[5].reshape(-1, self.MCES_NUM)

        l_clses = outputs[6].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[7].reshape(-1, self.REG * 4)
        l_mces = outputs[8].reshape(-1, self.MCES_NUM)

        protos = outputs[9]


        # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.CONF_THRES_RAW)  # 得到大于阈值分数的索引，此时为小数字
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 3个Classify分类分支：Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # 3个Bounding Box分支：反量化
        s_bboxes_float32 = s_bboxes[s_valid_indices,:]
        m_bboxes_float32 = m_bboxes[m_valid_indices,:]
        l_bboxes_float32 = l_bboxes[l_valid_indices,:]

        # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
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

        # 三个Mask Coefficients分支的索引
        s_mces_float32 = s_mces[s_valid_indices,:]
        m_mces_float32 = m_mces[m_valid_indices,:]
        l_mces_float32 = l_mces[l_valid_indices,:]

        # Mask Proto的反量化
        protos_float32 = protos.astype(np.float32)[0]

        # 大中小特征层阈值筛选结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        mces = np.concatenate((s_mces_float32, m_mces_float32, l_mces_float32), axis=0)

        # xyxy 2 xyhw
        xy = (dbboxes[:,2:4] + dbboxes[:,0:2])/2.0
        hw = (dbboxes[:,2:4] - dbboxes[:,0:2])
        xyhw = np.hstack([xy, hw])

        # 分类别nms
        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids==i
            indices = cv2.dnn.NMSBoxes(xyhw[id_indices,:], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices,:][indic]
                # mask
                x1_corp = int(x1 * self.x_scale_corp)
                y1_corp = int(y1 * self.y_scale_corp)
                x2_corp = int(x2 * self.x_scale_corp)
                y2_corp = int(y2 * self.y_scale_corp)
                # bbox
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)    
                # clip
                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h       
                # mask
                mc = mces[id_indices][indic]
                mask = (np.sum(mc[np.newaxis, np.newaxis, :]*protos_float32[y1_corp:y2_corp,x1_corp:x2_corp,:], axis=2) > 0.5).astype(np.uint8)
                # append
                results.append((i, scores[id_indices][indic], x1, y1, x2, y2, mask))

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return results

coco_names = ["3D CG rendering", "3D glasses", "abacus", "abalone", "monastery", "belly", "academy", "accessory", "accident", "accordion", "acorn", "acrylic paint", "act", "action", "action film", "activity", "actor", "adaptation", "add", "adhesive tape", "adjust", "adult", "adventure", "advertisement", "antenna", "aerobics", "spray can", "afro", "agriculture", "aid", "air conditioner", "air conditioning", "air sock", "aircraft cabin", "aircraft model", "air field", "air line", "airliner", "airman", "plane", "airplane window", "airport", "airport runway", "airport terminal", "airship", "airshow", "aisle", "alarm", "alarm clock", "mollymawk", "album", "album cover", "alcohol", "alcove", "algae", "alley", "almond", "aloe vera", "alp", "alpaca", "alphabet", "german shepherd", "altar", "amber", "ambulance", "bald eagle", "American shorthair", "amethyst", "amphitheater", "amplifier", "amusement park", "amusement ride", "anchor", "ancient", "anemone", "angel", "angle", "animal", "animal sculpture", "animal shelter", "animation", "animation film", "animator", "anime", "ankle", "anklet", "anniversary", "trench coat", "ant", "antelope", "antique", "antler", "anvil", "apartment", "ape", "app", "app icon", "appear", "appearance", "appetizer", "applause", "apple", "apple juice", "apple pie", "apple tree", "applesauce", "appliance", "appointment", "approach", "apricot", "apron", "aqua", "aquarium", "aquarium fish", "aqueduct", "arcade", "arcade machine", "arch", "arch bridge", "archaelogical excavation", "archery", "archipelago", "architect", "architecture", "archive", "archway", "area", "arena", "argument", "arm", "armadillo", "armband", "armchair", "armoire", "armor", "army", "army base", "army tank", "array", "arrest", "arrow", "art", "art exhibition", "art gallery", "art print", "art school", "art studio", "art vector illustration", "artichoke", "article", "artifact", "artist", "artists loft", "ash", "ashtray", "asia temple", "asparagus", "asphalt road", "assemble", "assembly", "assembly line", "association", "astronaut", "astronomer", "athlete", "athletic", "atlas", "atm", "atmosphere", "atrium", "attach", "fighter jet", "attend", "attraction", "atv", "eggplant", "auction", "audi", "audio", "auditorium", "aurora", "author", "auto factory", "auto mechanic", "auto part", "auto show", "auto showroom", "car battery", "automobile make", "automobile model", "motor vehicle", "autumn", "autumn forest", "autumn leave", "autumn park", "autumn tree", "avatar", "avenue", "aviator sunglasses", "avocado", "award", "award ceremony", "award winner", "shed", "ax", "azalea", "baboon", "baby", "baby bottle", "baby carriage", "baby clothe", "baby elephant", "baby food", "baby seat", "baby shower", "back", "backdrop", "backlight", "backpack", "backyard", "bacon", "badge", "badger", "badlands", "badminton", "badminton racket", "bag", "bagel", "bagpipe", "baguette", "bait", "baked goods", "baker", "bakery", "baking", "baking sheet", "balance", "balance car", "balcony", "ball", "ball pit", "ballerina", "ballet", "ballet dancer", "ballet skirt", "balloon", "balloon arch", "baseball player", "ballroom", "bamboo", "bamboo forest", "banana", "banana bread", "banana leaf", "banana tree", "band", "band aid", "bandage", "headscarf", "bandeau", "bangs", "bracelet", "balustrade", "banjo", "bank", "bank card", "bank vault", "banknote", "banner", "banquet", "banquet hall", "banyan tree", "baozi", "baptism", "bar", "bar code", "bar stool", "barbecue", "barbecue grill", "barbell", "barber", "barber shop", "barbie", "barge", "barista", "bark", "barley", "barn", "barn owl", "barn door", "barrel", "barricade", "barrier", "handcart", "bartender", "baseball", "baseball base", "baseball bat", "baseball hat", "baseball stadium", "baseball game", "baseball glove", "baseball pitcher", "baseball team", "baseball uniform", "basement", "basil", "basin", "basket", "basket container", "basketball", "basketball backboard", "basketball coach", "basketball court", "basketball game", "basketball hoop", "basketball player", "basketball stadium", "basketball team", "bass", "bass guitar", "bass horn", "bassist", "bat", "bath", "bath heater", "bath mat", "bath towel", "swimwear", "bathrobe", "bathroom", "bathroom accessory", "bathroom cabinet", "bathroom door", "bathroom mirror", "bathroom sink", "toilet paper", "bathroom window", "batman", "wand", "batter", "battery", "battle", "battle rope", "battleship", "bay", "bay bridge", "bay window", "bayberry", "bazaar", "beach", "beach ball", "beach chair", "beach house", "beach hut", "beach towel", "beach volleyball", "lighthouse", "bead", "beagle", "beak", "beaker", "beam", "bean", "bean bag chair", "beanbag", "bear", "bear cub", "beard", "beast", "beat", "beautiful", "beauty", "beauty salon", "beaver", "bed", "bedcover", "bed frame", "bedroom", "bedding", "bedpan", "bedroom window", "bedside lamp", "bee", "beech tree", "beef", "beekeeper", "beeper", "beer", "beer bottle", "beer can", "beer garden", "beer glass", "beer hall", "beet", "beetle", "beige", "clock", "bell pepper", "bell tower", "belt", "belt buckle", "bench", "bend", "bengal tiger", "bento", "beret", "berry", "berth", "beverage", "bib", "bibimbap", "bible", "bichon", "bicycle", "bicycle helmet", "bicycle wheel", "biker", "bidet", "big ben", "bike lane", "bike path", "bike racing", "bike ride", "bikini", "bikini top", "bill", "billard", "billboard", "billiard table", "bin", "binder", "binocular", "biology laboratory", "biplane", "birch", "birch tree", "bird", "bird bath", "bird feeder", "bird house", "bird nest", "birdbath", "bird cage", "birth", "birthday", "birthday cake", "birthday candle", "birthday card", "birthday party", "biscuit", "bishop", "bison", "bit", "bite", "black", "black sheep", "blackberry", "blackbird", "blackboard", "blacksmith", "blade", "blanket", "sports coat", "bleacher", "blender", "blessing", "blind", "eye mask", "flasher", "snowstorm", "block", "blog", "blood", "bloom", "blossom", "blouse", "blow", "hair drier", "blowfish", "blue", "blue artist", "blue jay", "blue sky", "blueberry", "bluebird", "pig", "board", "board eraser", "board game", "boardwalk", "boat", "boat deck", "boat house", "paddle", "boat ride", "bobfloat", "bobcat", "body", "bodyboard", "bodybuilder", "boiled egg", "boiler", "bolo tie", "bolt", "bomb", "bomber", "bonasa umbellu", "bone", "bonfire", "bonnet", "bonsai", "book", "book cover", "bookcase", "folder", "bookmark", "bookshelf", "bookstore", "boom microphone", "boost", "boot", "border", "Border collie", "botanical garden", "bottle", "bottle cap", "bottle opener", "bottle screw", "bougainvillea", "boulder", "bouquet", "boutique", "boutique hotel", "bow", "bow tie", "bow window", "bowl", "bowling", "bowling alley", "bowling ball", "bowling equipment", "box", "box girder bridge", "box turtle", "boxer", "underdrawers", "boxing", "boxing glove", "boxing ring", "boy", "brace", "bracket", "braid", "brain", "brake", "brake light", "branch", "brand", "brandy", "brass", "brass plaque", "bread", "breadbox", "break", "breakfast", "seawall", "chest", "brewery", "brick", "brick building", "wall", "brickwork", "wedding dress", "bride", "groom", "bridesmaid", "bridge", "bridle", "briefcase", "bright", "brim", "broach", "broadcasting", "broccoli", "bronze", "bronze medal", "bronze sculpture", "bronze statue", "brooch", "creek", "broom", "broth", "brown", "brown bear", "brownie", "brunch", "brunette", "brush", "coyote", "brussels sprout", "bubble", "bubble gum", "bubble tea", "bucket cabinet", "shield", "bud", "buddha", "buffalo", "buffet", "bug", "build", "builder", "building", "building block", "building facade", "building material", "lamp", "bull", "bulldog", "bullet", "bullet train", "bulletin board", "bulletproof vest", "bullfighting", "megaphone", "bullring", "bumblebee", "bumper", "roll", "bundle", "bungee", "bunk bed", "bunker", "bunny", "buoy", "bureau", "burial chamber", "burn", "burrito", "bus", "bus driver", "bus interior", "bus station", "bus stop", "bus window", "bush", "business", "business card", "business executive", "business suit", "business team", "business woman", "businessman", "bust", "butcher", "butchers shop", "butte", "butter", "cream", "butterfly", "butterfly house", "button", "buttonwood", "buy", "taxi", "cabana", "cabbage", "cabin", "cabin car", "cabinet", "cabinetry", "cable", "cable car", "cactus", "cafe", "canteen", "cage", "cake", "cake stand", "calculator", "caldron", "calendar", "calf", "call", "phone box", "calligraphy", "calm", "camcorder", "camel", "camera", "camera lens", "camouflage", "camp", "camper", "campfire", "camping", "campsite", "campus", "can", "can opener", "canal", "canary", "cancer", "candle", "candle holder", "candy", "candy bar", "candy cane", "candy store", "cane", "jar", "cannon", "canopy", "canopy bed", "cantaloupe", "cantilever bridge", "canvas", "canyon", "cap", "cape", "cape cod", "cappuccino", "capsule", "captain", "capture", "car", "car dealership", "car door", "car interior", "car logo", "car mirror", "parking lot", "car seat", "car show", "car wash", "car window", "caramel", "card", "card game", "cardboard", "cardboard box", "cardigan", "cardinal", "cargo", "cargo aircraft", "cargo ship", "caribbean", "carnation", "carnival", "carnivore", "carousel", "carp", "carpenter", "carpet", "slipper", "house finch", "coach", "dalmatian", "aircraft carrier", "carrot", "carrot cake", "carry", "cart", "carton", "cartoon", "cartoon character", "cartoon illustration", "cartoon style", "carve", "case", "cash", "cashew", "casino", "casserole", "cassette", "cassette deck", "plaster bandage", "casting", "castle", "cat", "cat bed", "cat food", "cat furniture", "cat tree", "catacomb", "catamaran", "catamount", "catch", "catcher", "caterpillar", "catfish", "cathedral", "cattle", "catwalk", "catwalk show", "cauliflower", "cave", "caviar", "CD", "CD player", "cedar", "ceiling", "ceiling fan", "celebrate", "celebration", "celebrity", "celery", "cello", "smartphone", "cement", "graveyard", "centerpiece", "centipede", "ceramic", "ceramic tile", "cereal", "ceremony", "certificate", "chain", "chain saw", "chair", "chairlift", "daybed", "chalet", "chalice", "chalk", "chamber", "chameleon", "champagne", "champagne flute", "champion", "championship", "chandelier", "changing table", "channel", "chap", "chapel", "character sculpture", "charcoal", "charge", "charger", "chariot", "charity", "charity event", "charm", "graph", "chase", "chassis", "check", "checkbook", "chessboard", "checklist", "cheer", "cheerlead", "cheese", "cheeseburger", "cheesecake", "cheetah", "chef", "chemical compound", "chemist", "chemistry", "chemistry lab", "cheongsam", "cherry", "cherry blossom", "cherry tomato", "cherry tree", "chess", "chestnut", "chicken", "chicken breast", "chicken coop", "chicken salad", "chicken wing", "garbanzo", "chiffonier", "chihuahua", "child", "child actor", "childs room", "chile", "chili dog", "chimney", "chimpanzee", "chinaware", "chinese cabbage", "chinese garden", "chinese knot", "chinese rose", "chinese tower", "chip", "chipmunk", "chisel", "chocolate", "chocolate bar", "chocolate cake", "chocolate chip", "chocolate chip cookie", "chocolate milk", "chocolate mousse", "truffle", "choir", "kitchen knife", "cutting board", "chopstick", "christmas", "christmas ball", "christmas card", "christmas decoration", "christmas dinner", "christmas eve", "christmas hat", "christmas light", "christmas market", "christmas ornament", "christmas tree", "chrysanthemum", "church", "church tower", "cider", "cigar", "cigar box", "cigarette", "cigarette case", "waistband", "cinema", "photographer", "cinnamon", "circle", "circuit", "circuit board", "circus", "water tank", "citrus fruit", "city", "city bus", "city hall", "city nightview", "city park", "city skyline", "city square", "city street", "city wall", "city view", "clam", "clarinet", "clasp", "class", "classic", "classroom", "clavicle", "claw", "clay", "pottery", "clean", "clean room", "cleaner", "cleaning product", "clear", "cleat", "clementine", "client", "cliff", "climb", "climb mountain", "climber", "clinic", "clip", "clip art", "clipboard", "clipper", "clivia", "cloak", "clogs", "close-up", "closet", "cloth", "clothe", "clothing", "clothespin", "clothesline", "clothing store", "cloud", "cloud forest", "cloudy", "clover", "joker", "clown fish", "club", "clutch", "clutch bag", "coal", "coast", "coat", "coatrack", "cob", "cock", "cockatoo", "cocker", "cockpit", "roach", "cocktail", "cocktail dress", "cocktail shaker", "cocktail table", "cocoa", "coconut", "coconut tree", "coffee", "coffee bean", "coffee cup", "coffee machine", "coffee shop", "coffeepot", "coffin", "cognac", "spiral", "coin", "coke", "colander", "cold", "slaw", "collaboration", "collage", "collection", "college student", "sheepdog", "crash", "color", "coloring book", "coloring material", "pony", "pillar", "comb", "combination lock", "comic", "comedy", "comedy film", "comet", "comfort", "comfort food", "comic book", "comic book character", "comic strip", "commander", "commentator", "community", "commuter", "company", "compass", "compete", "contest", "competitor", "composer", "composition", "compost", "computer", "computer box", "computer chair", "computer desk", "keyboard", "computer monitor", "computer room", "computer screen", "computer tower", "concept car", "concert", "concert hall", "conch", "concrete", "condiment", "condom", "condominium", "conductor", "cone", "meeting", "conference center", "conference hall", "meeting room", "confetti", "conflict", "confluence", "connect", "connector", "conservatory", "constellation", "construction site", "construction worker", "contain", "container", "container ship", "continent", "profile", "contract", "control", "control tower", "convenience store", "convention", "conversation", "converter", "convertible", "transporter", "cook", "cooking", "cooking spray", "cooker", "cool", "cooler", "copper", "copy", "coral", "coral reef", "rope", "corded phone", "liquor", "corgi", "cork", "corkboard", "cormorant", "corn", "corn field", "cornbread", "corner", "trumpet", "cornice", "cornmeal", "corral", "corridor", "corset", "cosmetic", "cosmetics brush", "cosmetics mirror", "cosplay", "costume", "costumer film designer", "infant bed", "cottage", "cotton", "cotton candy", "couch", "countdown", "counter", "counter top", "country artist", "country house", "country lane", "country pop artist", "countryside", "coupe", "couple", "couple photo", "courgette", "course", "court", "courthouse", "courtyard", "cousin", "coverall", "cow", "cowbell", "cowboy", "cowboy boot", "cowboy hat", "crab", "crabmeat", "crack", "cradle", "craft", "craftsman", "cranberry", "crane", "crape", "crapper", "crate", "crater lake", "lobster", "crayon", "cream cheese", "cream pitcher", "create", "creature", "credit card", "crescent", "croissant", "crest", "crew", "cricket", "cricket ball", "cricket team", "cricketer", "crochet", "crock pot", "crocodile", "crop", "crop top", "cross", "crossbar", "crossroad", "crosstalk", "crosswalk", "crouton", "crow", "crowbar", "crowd", "crowded", "crown", "crt screen", "crucifix", "cruise", "cruise ship", "cruiser", "crumb", "crush", "crutch", "crystal", "cub", "cube", "cucumber", "cue", "cuff", "cufflink", "cuisine", "farmland", "cup", "cupcake", "cupid", "curb", "curl", "hair roller", "currant", "currency", "curry", "curtain", "curve", "pad", "customer", "cut", "cutlery", "cycle", "cycling", "cyclone", "cylinder", "cymbal", "cypress", "cypress tree", "dachshund", "daffodil", "dagger", "dahlia", "daikon", "dairy", "daisy", "dam", "damage", "damp", "dance", "dance floor", "dance room", "dancer", "dandelion", "dark", "darkness", "dart", "dartboard", "dashboard", "date", "daughter", "dawn", "day bed", "daylight", "deadbolt", "death", "debate", "debris", "decanter", "deck", "decker bus", "decor", "decorate", "decorative picture", "deer", "defender", "deity", "delicatessen", "deliver", "demolition", "monster", "demonstration", "den", "denim jacket", "dentist", "department store", "depression", "derby", "dermopathy", "desert", "desert road", "design", "designer", "table", "table lamp", "desktop", "desktop computer", "dessert", "destruction", "detective", "detergent", "dew", "dial", "diamond", "diaper", "diaper bag", "journal", "die", "diet", "excavator", "number", "digital clock", "dill", "dinner", "rowboat", "dining room", "dinner party", "dinning table", "dinosaur", "dip", "diploma", "direct", "director", "dirt", "dirt bike", "dirt field", "dirt road", "dirt track", "disaster", "disciple", "disco", "disco ball", "discotheque", "disease", "plate", "dish antenna", "dish washer", "dishrag", "dishes", "dishsoap", "Disneyland", "dispenser", "display", "display window", "trench", "dive", "diver", "diving board", "paper cup", "dj", "doberman", "dock", "doctor", "document", "documentary", "dog", "dog bed", "dog breed", "dog collar", "dog food", "dog house", "doll", "dollar", "dollhouse", "dolly", "dolphin", "dome", "domicile", "domino", "donkey", "donut", "doodle", "door", "door handle", "doormat", "doorplate", "doorway", "dormitory", "dough", "downtown", "dozer", "drag", "dragon", "dragonfly", "drain", "drama", "drama film", "draw", "drawer", "drawing", "drawing pin", "pigtail", "dress", "dress hat", "dress shirt", "dress shoe", "dress suit", "dresser", "dressing room", "dribble", "drift", "driftwood", "drill", "drink", "drinking water", "drive", "driver", "driveway", "drone", "drop", "droplight", "dropper", "drought", "medicine", "pharmacy", "drum", "drummer", "drumstick", "dry", "duchess", "duck", "duckbill", "duckling", "duct tape", "dude", "duet", "duffel", "canoe", "dumbbell", "dumpling", "dune", "dunk", "durian", "dusk", "dust", "garbage truck", "dustpan", "duvet", "DVD", "dye", "eagle", "ear", "earmuff", "earphone", "earplug", "earring", "earthquake", "easel", "easter", "easter bunny", "easter egg", "eat", "restaurant", "eclair", "eclipse", "ecosystem", "edit", "education", "educator", "eel", "egg", "egg roll", "egg tart", "eggbeater", "egret", "Eiffel tower", "elastic band", "senior", "electric chair", "electric drill", "electrician", "electricity", "electron", "electronic", "elephant", "elevation map", "elevator", "elevator car", "elevator door", "elevator lobby", "elevator shaft", "embankment", "embassy", "embellishment", "ember", "emblem", "embroidery", "emerald", "emergency", "emergency service", "emergency vehicle", "emotion", "Empire State Building", "enamel", "enclosure", "side table", "energy", "engagement", "engagement ring", "engine", "engine room", "engineer", "engineering", "english shorthair", "ensemble", "enter", "entertainer", "entertainment", "entertainment center", "entrance", "entrance hall", "envelope", "equestrian", "equipment", "eraser", "erhu", "erosion", "escalator", "escargot", "espresso", "estate", "estuary", "eucalyptus tree", "evening", "evening dress", "evening light", "evening sky", "evening sun", "event", "evergreen", "ewe", "excavation", "exercise", "exhaust hood", "exhibition", "exit", "explorer", "explosion", "extension cord", "extinguisher", "extractor", "extrude", "eye", "eye shadow", "eyebrow", "eyeliner", "fabric", "fabric store", "facade", "face", "face close-up", "face powder", "face towel", "facial tissue holder", "facility", "factory", "factory workshop", "fair", "fairground", "fairy", "falcon", "fall", "family", "family car", "family photo", "family room", "fan", "fang", "farm", "farmer", "farmer market", "farmhouse", "fashion", "fashion accessory", "fashion designer", "fashion girl", "fashion illustration", "fashion look", "fashion model", "fashion show", "fast food", "fastfood restaurant", "father", "faucet", "fault", "fauna", "fawn", "fax", "feast", "feather", "fedora", "feed", "feedbag", "feeding", "feeding chair", "feline", "mountain lion", "fence", "fender", "fern", "ferret", "ferris wheel", "ferry", "fertilizer", "festival", "fiber", "fiction", "fiction book", "field", "field road", "fig", "fight", "figure skater", "figurine", "file", "file photo", "file cabinet", "fill", "film camera", "film director", "film format", "film premiere", "film producer", "filming", "filter", "fin", "hand", "finish line", "fir", "fir tree", "fire", "fire alarm", "fire department", "fire truck", "fire escape", "fire hose", "fire pit", "fire station", "firecracker", "fireman", "fireplace", "firework", "firework display", "first-aid kit", "fish", "fish boat", "fish market", "fish pond", "fishbowl", "fisherman", "fishing", "fishing boat", "fishing net", "fishing pole", "fishing village", "fitness", "fitness course", "five", "fixture", "fjord", "flag", "flag pole", "flake", "flame", "flamingo", "flannel", "flap", "flare", "flash", "flask", "flat", "flatfish", "flavor", "flea", "flea market", "fleet", "flight", "flight attendant", "flip", "flip-flop", "flipchart", "float", "flock", "flood", "floor", "floor fan", "floor mat", "floor plan", "floor window", "floral arrangement", "florist", "floss", "flour", "flow", "flower", "flower basket", "flower bed", "flower box", "flower field", "flower girl", "flower market", "fluid", "flush", "flute", "fly", "fly fishing", "flyer", "horse", "foam", "fog", "foggy", "foie gra", "foil", "folding chair", "leaf", "folk artist", "folk dance", "folk rock artist", "fondant", "hotpot", "font", "food", "food coloring", "food court", "food processor", "food stand", "food truck", "foosball", "foot", "foot bridge", "football", "football coach", "football college game", "football match", "football field", "football game", "football helmet", "football player", "football stadium", "football team", "path", "footprint", "footrest", "footstall", "footwear", "forbidden city", "ford", "forehead", "forest", "forest fire", "forest floor", "forest path", "forest road", "forge", "fork", "forklift", "form", "formal garden", "formation", "formula 1", "fort", "fortification", "forward", "fossil", "foundation", "fountain", "fountain pen", "fox", "frame", "freckle", "highway", "lorry", "French", "French bulldog", "French fries", "French toast", "freshener", "fridge", "fried chicken", "fried egg", "fried rice", "friendship", "frisbee", "frog", "frost", "frosting", "frosty", "frozen", "fruit", "fruit cake", "fruit dish", "fruit market", "fruit salad", "fruit stand", "fruit tree", "fruits shop", "fry", "frying pan", "fudge", "fuel", "fume hood", "fun", "funeral", "fungi", "funnel", "fur", "fur coat", "furniture", "futon", "gadget", "muzzle", "galaxy", "gallery", "game", "game board", "game controller", "ham", "gang", "garage", "garage door", "garage kit", "garbage", "garden", "garden asparagus", "garden hose", "garden spider", "gardener", "gardening", "garfield", "gargoyle", "wreath", "garlic", "garment", "gas", "gas station", "gas stove", "gasmask", "collect", "gathering", "gauge", "gazebo", "gear", "gecko", "geisha", "gel", "general store", "generator", "geranium", "ghost", "gift", "gift bag", "gift basket", "gift box", "gift card", "gift shop", "gift wrap", "gig", "gin", "ginger", "gingerbread", "gingerbread house", "ginkgo tree", "giraffe", "girl", "give", "glacier", "gladiator", "glass bead", "glass bottle", "glass bowl", "glass box", "glass building", "glass door", "glass floor", "glass house", "glass jar", "glass plate", "glass table", "glass vase", "glass wall", "glass window", "glasses", "glaze", "glider", "earth", "glove", "glow", "glue pudding", "go", "go for", "goal", "goalkeeper", "goat", "goat cheese", "gobi", "goggles", "gold", "gold medal", "Golden Gate Bridge", "golden retriever", "goldfish", "golf", "golf cap", "golf cart", "golf club", "golf course", "golfer", "goose", "gorilla", "gothic", "gourd", "government", "government agency", "gown", "graduate", "graduation", "grain", "grampus", "grand prix", "grandfather", "grandmother", "grandparent", "granite", "granola", "grape", "grapefruit", "wine", "grass", "grasshopper", "grassland", "grassy", "grater", "grave", "gravel", "gravestone", "gravy", "gravy boat", "gray", "graze", "grazing", "green", "greenery", "greet", "greeting", "greeting card", "greyhound", "grid", "griddle", "grill", "grille", "grilled eel", "grind", "grinder", "grits", "grocery bag", "grotto", "ground squirrel", "group", "group photo", "grove", "grow", "guacamole", "guard", "guard dog", "guest house", "guest room", "guide", "guinea pig", "guitar", "guitarist", "gulf", "gull", "gun", "gundam", "gurdwara", "guzheng", "gym", "gymnast", "habitat", "hacker", "hail", "hair", "hair color", "hair spray", "hairbrush", "haircut", "hairgrip", "hairnet", "hairpin", "hairstyle", "half", "hall", "halloween", "halloween costume", "halloween pumpkin", "halter top", "hamburg", "hamburger", "hami melon", "hammer", "hammock", "hamper", "hamster", "hand dryer", "hand glass", "hand towel", "handbag", "handball", "handcuff", "handgun", "handkerchief", "handle", "handsaw", "handshake", "handstand", "handwriting", "hanfu", "hang", "hangar", "hanger", "happiness", "harbor", "harbor seal", "hard rock artist", "hardback book", "safety helmet", "hardware", "hardware store", "hardwood", "hardwood floor", "mouth organ", "pipe organ", "harpsichord", "harvest", "harvester", "hassock", "hat", "hatbox", "hautboy", "hawthorn", "hay", "hayfield", "hazelnut", "head", "head coach", "headlight", "headboard", "headdress", "headland", "headquarter", "hearing", "heart", "heart shape", "heat", "heater", "heather", "hedge", "hedgehog", "heel", "helicopter", "heliport", "helmet", "help", "hen", "henna", "herb", "herd", "hermit crab", "hero", "heron", "hibiscus", "hibiscus flower", "hide", "high bar", "high heel", "highland", "highlight", "hike", "hiker", "hiking boot", "hiking equipment", "hill", "hill country", "hill station", "hillside", "hindu temple", "hinge", "hip", "hip hop artist", "hippo", "historian", "historic", "history", "hockey", "hockey arena", "hockey game", "hockey player", "hockey stick", "hoe", "hole", "vacation", "holly", "holothurian", "home", "home appliance", "home base", "home decor", "home interior", "home office", "home theater", "homework", "hummus", "honey", "beehive", "honeymoon", "hood", "hoodie", "hook", "jump", "horizon", "hornbill", "horned cow", "hornet", "horror", "horror film", "horse blanket", "horse cart", "horse farm", "horse ride", "horseback", "horseshoe", "hose", "hospital", "hospital bed", "hospital room", "host", "inn", "hot", "hot air balloon", "hot dog", "hot sauce", "hot spring", "hotel", "hotel lobby", "hotel room", "hotplate", "hourglass", "house", "house exterior", "houseplant", "hoverboard", "howler", "huddle", "hug", "hula hoop", "person", "humidifier", "hummingbird", "humpback whale", "hunt", "hunting lodge", "hurdle", "hurricane", "husky", "hut", "hyaena", "hybrid", "hydrangea", "hydrant", "seaplane", "ice", "ice bag", "polar bear", "ice cave", "icecream", "ice cream cone", "ice cream parlor", "ice cube", "ice floe", "ice hockey player", "ice hockey team", "lollipop", "ice maker", "rink", "ice sculpture", "ice shelf", "skate", "ice skating", "iceberg", "icicle", "icing", "icon", "id photo", "identity card", "igloo", "light", "iguana", "illuminate", "illustration", "image", "impala", "incense", "independence day", "individual", "indoor", "indoor rower", "induction cooker", "industrial area", "industry", "infantry", "inflatable boat", "information desk", "infrastructure", "ingredient", "inhalator", "injection", "injury", "ink", "inking pad", "inlet", "inscription", "insect", "install", "instrument", "insulated cup", "interaction", "interior design", "website", "intersection", "interview", "invertebrate", "invitation", "ipad", "iphone", "ipod", "iris", "iron", "ironing board", "irrigation system", "island", "islet", "isopod", "ivory", "ivy", "izakaya", "jack", "jackcrab", "jacket", "jacuzzi", "jade", "jaguar", "jail cell", "jam", "japanese garden", "jasmine", "jaw", "jay", "jazz", "jazz artist", "jazz fusion artist", "jeans", "jeep", "jelly", "jelly bean", "jellyfish", "jet", "motorboat", "jewel", "jewellery", "jewelry shop", "jigsaw puzzle", "rickshaw", "jockey", "jockey cap", "jog", "joint", "journalist", "joystick", "judge", "jug", "juggle", "juice", "juicer", "jujube", "jump rope", "jumpsuit", "jungle", "junkyard", "kale", "kaleidoscope", "kangaroo", "karaoke", "karate", "karting", "kasbah", "kayak", "kebab", "key", "keycard", "khaki", "kick", "kilt", "kimono", "kindergarden classroom", "kindergarten", "king", "king crab", "kiss", "kit", "kitchen", "kitchen cabinet", "kitchen counter", "kitchen floor", "kitchen hood", "kitchen island", "kitchen sink", "kitchen table", "kitchen utensil", "kitchen window", "kitchenware", "kite", "kiwi", "knee pad", "kneel", "knife", "rider", "knit", "knitting needle", "knob", "knocker", "knot", "koala", "koi", "ktv", "laboratory", "lab coat", "label", "labrador", "maze", "lace", "lace dress", "ladder", "ladle", "ladybird", "lagoon", "lake", "lake district", "lake house", "lakeshore", "lamb", "lamb chop", "lamp post", "lamp shade", "spear", "land", "land vehicle", "landfill", "landing", "landing deck", "landmark", "landscape", "landslide", "lanyard", "lantern", "lap", "laptop", "laptop keyboard", "larva", "lasagne", "laser", "lash", "lasso", "latch", "latex", "latte", "laugh", "launch", "launch event", "launch party", "laundromat", "laundry", "laundry basket", "laundry room", "lava", "lavender", "lawn", "lawn wedding", "lawyer", "lay", "lead", "lead singer", "lead to", "leader", "leak", "lean", "learn", "leash", "leather", "leather jacket", "leather shoe", "speech", "lecture hall", "lecture room", "ledge", "leftover", "leg", "legend", "legging", "legislative chamber", "lego", "legume", "lemon", "lemon juice", "lemonade", "lemur", "lens", "lens flare", "lentil", "leopard", "leotard", "tights", "leprechaun", "lesson", "letter", "mailbox", "letter logo", "lettering", "lettuce", "level", "library", "license", "license plate", "lichen", "lick", "lid", "lie", "life belt", "life jacket", "lifeboat", "lifeguard", "lift", "light fixture", "light show", "light switch", "lighting", "lightning", "lightning rod", "lilac", "lily", "limb", "lime", "limestone", "limo", "line", "line art", "line up", "linen", "liner", "lion", "lip balm", "lipstick", "liquid", "liquor store", "list", "litchi", "live", "livestock", "living room", "living space", "lizard", "load", "loading dock", "loafer", "hallway", "locate", "lock", "lock chamber", "locker", "loft", "log", "log cabin", "logo", "loki", "long hair", "longboard", "loom", "loop", "lose", "lottery", "lotus", "love", "loveseat", "luggage", "lumber", "lumberjack", "lunch", "lunch box", "lush", "luxury", "luxury yacht", "mac", "macadamia", "macaque", "macaroni", "macaw", "machete", "machine", "machine gun", "magazine", "magic", "magician", "magnet", "magnifying glass", "magnolia", "magpie", "mahjong", "mahout", "maid", "chain mail", "mail slot", "make", "makeover", "makeup artist", "makeup tool", "mallard", "mallard duck", "mallet", "mammal", "mammoth", "man", "management", "manager", "manatee", "mandala", "mandarin orange", "mandarine", "mane", "manga", "manger", "mango", "mangosteen", "mangrove", "manhattan", "manhole", "manhole cover", "manicure", "mannequin", "manor house", "mansion", "mantid", "mantle", "manufactured home", "manufacturing", "manuscript", "map", "maple", "maple leaf", "maple syrup", "maraca", "marathon", "marble", "march", "marching band", "mare", "marigold", "marine", "marine invertebrate", "marine mammal", "puppet", "mark", "market", "market square", "market stall", "marriage", "martial", "martial artist", "martial arts gym", "martini", "martini glass", "mascara", "mascot", "mashed potato", "masher", "mask", "massage", "mast", "mat", "matador", "match", "matchbox", "material", "mattress", "mausoleum", "maxi dress", "meal", "measuring cup", "measuring tape", "meat", "meatball", "mechanic", "mechanical fan", "medal", "media", "medical equipment", "medical image", "medical staff", "medicine cabinet", "medieval", "medina", "meditation", "meerkat", "meet", "melon", "monument", "menu", "mermaid", "net", "mess", "messenger bag", "metal", "metal artist", "metal detector", "meter", "mezzanine", "microphone", "microscope", "microwave", "midnight", "milestone", "military uniform", "milk", "milk can", "milk tea", "milkshake", "mill", "mine", "miner", "mineral", "mineral water", "miniskirt", "miniature", "minibus", "minister", "minivan", "mint", "mint candy", "mirror", "miss", "missile", "mission", "mistletoe", "mix", "mixer", "mixing bowl", "mixture", "moat", "mobility scooter", "model", "model car", "modern", "modern tower", "moisture", "mold", "molding", "mole", "monarch", "money", "monitor", "monk", "monkey", "monkey wrench", "monochrome", "monocycle", "monster truck", "moon", "moon cake", "moonlight", "moor", "moose", "swab", "moped", "morning", "morning fog", "morning light", "morning sun", "mortar", "mosaic", "mosque", "mosquito", "moss", "motel", "moth", "mother", "motherboard", "motif", "sport", "motor", "motorbike", "motorcycle", "motorcycle helmet", "motorcycle racer", "motorcyclist", "motorsport", "mound", "mountain", "mountain bike", "mountain biker", "mountain biking", "mountain gorilla", "mountain lake", "mountain landscape", "mountain pass", "mountain path", "mountain range", "mountain river", "mountain snowy", "mountain stream", "mountain view", "mountain village", "mountaineer", "mountaineering bag", "mouse", "mousepad", "mousetrap", "mouth", "mouthwash", "move", "movie poster", "movie ticket", "mower", "mp3 player", "mr", "mud", "muffin", "mug", "mulberry", "mulch", "mule", "municipality", "mural", "muscle", "muscle car", "museum", "mushroom", "music", "music festival", "music stool", "music studio", "music video performer", "musical keyboard", "musician", "mussel", "mustard", "mythology", "nacho", "nail polish", "nailfile", "nanny", "napkin", "narrow", "national flag", "nativity scene", "natural history museum", "nature", "nature reserve", "navigation", "navratri", "navy", "nebula", "neck", "neckband", "necklace", "neckline", "nectar", "nectarine", "needle", "neighbor", "neighbourhood", "neon", "neon light", "nerve", "nest", "new year", "newborn", "newfoundland", "newlywed", "news", "news conference", "newsstand", "night", "night market", "night sky", "night view", "nightclub", "nightstand", "noodle", "nose", "noseband", "note", "notebook", "notepad", "notepaper", "notice", "number icon", "nun", "nurse", "nursery", "nursing home", "nut", "nutcracker", "oak", "oak tree", "oar", "oasis", "oast house", "oatmeal", "oats", "obelisk", "observation tower", "observatory", "obstacle course", "sea", "octopus", "offer", "office", "office building", "office chair", "office cubicle", "office desk", "office supply", "office window", "officer", "official", "oil", "oil lamp", "oil painting", "oilrig", "okra", "old photo", "olive", "olive oil", "olive tree", "omelet", "onion", "onion ring", "opal", "open", "opening", "opening ceremony", "opera", "opera house", "operate", "operating room", "operation", "optical shop", "orangutan", "orange", "orange juice", "orange tree", "orangery", "orbit", "orchard", "orchestra pit", "orchid", "order", "organization", "origami", "ornament", "osprey", "ostrich", "otter", "out", "outcrop", "outdoor", "outhouse", "electric outlet", "outline", "oval", "oven", "overall", "overcoat", "overpass", "owl", "oyster", "teething ring", "pack", "package", "paddock", "police van", "padlock", "paella", "pagoda", "pain", "paint brush", "painter", "paisley bandanna", "palace", "palette", "paling", "pall", "palm tree", "pan", "pancake", "panda", "panel", "panorama", "pansy", "pant", "pantry", "pants", "pantyhose", "papaya", "paper", "paper bag", "paper cutter", "paper lantern", "paper plate", "paper towel", "paperback book", "paperweight", "parachute", "parade", "paradise", "parrot", "paramedic", "paraquet", "parasail", "paratrooper", "parchment", "parish", "park", "park bench", "parking", "parking garage", "parking meter", "parking sign", "parliament", "parsley", "participant", "partner", "partridge", "party", "party hat", "pass", "passage", "passbook", "passenger", "passenger ship", "passenger train", "passion fruit", "passport", "pasta", "paste", "pastry", "pasture", "patch", "patient", "pattern", "pavement", "pavilion", "paw", "pay", "payphone", "pea", "peace", "peach", "peacock", "peak", "peanut", "peanut butter", "pear", "pearl", "pebble", "pecan", "pedestrian", "pedestrian bridge", "pedestrian street", "peel", "peeler", "pegboard", "pegleg", "pelican", "pen", "penalty kick", "pencil", "pencil case", "pencil sharpener", "pencil skirt", "pendant", "pendulum", "penguin", "peninsula", "pennant", "penny", "piggy bank", "peony", "pepper", "pepper grinder", "peppercorn", "pepperoni", "perch", "perform", "performance", "performance arena", "perfume", "pergola", "persian cat", "persimmon", "personal care", "personal flotation device", "pest", "pet", "pet shop", "pet store", "petal", "petunia", "church bench", "pheasant", "phenomenon", "philosopher", "phone", "phonebook", "record player", "photo", "photo booth", "photo frame", "photography", "physicist", "physics laboratory", "pianist", "piano", "plectrum", "pick up", "pickle", "picnic", "picnic area", "picnic basket", "picnic table", "picture", "picture frame", "pie", "pigeon", "pilgrim", "tablet", "pillow", "pilot", "pilot boat", "pin", "pine", "pine cone", "pine forest", "pine nut", "pineapple", "table tennis table", "table tennis", "pink", "pint", "pipa", "pipe", "pipe bowl", "pirate", "pirate flag", "pirate ship", "pistachio", "ski slope", "pocket bread", "pitaya", "pitbull", "pitch", "pitcher", "pitcher plant", "pitchfork", "pizza", "pizza cutter", "pizza pan", "pizzeria", "placard", "place", "place mat", "plaid", "plain", "plan", "planet", "planet earth", "plank", "plant", "plantation", "planting", "plaque", "plaster", "plastic", "plasticine", "plateau", "platform", "platinum", "platter", "play", "play badminton", "play baseball", "play basketball", "play billiard", "play football", "play pong", "play tennis", "play volleyball", "player", "playground", "playhouse", "playing card", "playing chess", "playing golf", "playing mahjong", "playingfield", "playpen", "playroom", "plaza", "plier", "plot", "plow", "plug", "plug hat", "plum", "plumber", "plumbing fixture", "plume", "plywood", "pocket", "pocket watch", "pocketknife", "pod", "podium", "poetry", "poinsettia", "point", "pointer", "poker card", "poker chip", "poker table", "pole", "polecat", "police", "police car", "police dog", "police station", "politician", "polka dot", "pollen", "pollution", "polo", "polo neck", "polo shirt", "pomegranate", "pomeranian", "poncho", "pond", "ponytail", "poodle", "pool", "pop", "pop artist", "popcorn", "pope", "poppy", "porcelain", "porch", "pork", "porridge", "portable battery", "portal", "portfolio", "porthole", "portrait", "portrait session", "pose", "possum", "post", "post office", "stamp", "postcard", "poster", "poster page", "pot", "potato", "potato chip", "potato salad", "potholder", "potty", "pouch", "poultry", "pound", "pour", "powder", "power line", "power plugs and sockets", "power see", "power station", "practice", "Prague Castle", "prayer", "preacher", "premiere", "prescription", "show", "presentation", "president", "press room", "pressure cooker", "pretzel", "prince", "princess", "print", "printed page", "printer", "printing", "prison", "produce", "product", "profession", "professional", "professor", "project picture", "projection screen", "projector", "prom", "promenade", "propeller", "prophet", "proposal", "protective suit", "protest", "protester", "publication", "publicity portrait", "ice hockey", "pudding", "puddle", "puff", "puffin", "pug", "pull", "pulpit", "pulse", "pump", "pumpkin", "pumpkin pie", "pumpkin seed", "punch bag", "punch", "student", "purple", "push", "putt", "puzzle", "tower", "pyramid", "python", "qr code", "quail", "quarry", "quarter", "quartz", "queen", "quesadilla", "queue", "quiche", "quilt", "quilting", "quote", "rabbit", "raccoon", "race", "race track", "raceway", "race car", "racket", "radar", "radiator", "radio", "raft", "rag doll", "rail", "railcar", "railroad", "railroad bridge", "railway line", "railway station", "rain", "rain boot", "rainbow", "rainbow trout", "raincoat", "rainforest", "rainy", "raisin", "rake", "ram", "ramp", "rapeseed", "rapid", "rapper", "raspberry", "rat", "ratchet", "raven", "ravine", "ray", "razor", "razor blade", "read", "reading", "reamer", "rear", "rear light", "rear view", "rearview mirror", "receipt", "receive", "reception", "recipe", "record", "record producer", "recorder", "recording studio", "recreation room", "recreational vehicle", "rectangle", "recycling", "recycling bin", "red", "red carpet", "red flag", "red panda", "red wine", "redwood", "reed", "reef", "reel", "referee", "reflect", "reflection", "reflector", "register", "rein", "reindeer", "relax", "release", "relief", "religion", "religious", "relish", "remain", "remodel", "remote", "remove", "repair", "repair shop", "reptile", "rescue", "rescuer", "research", "researcher", "reservoir", "residence", "residential neighborhood", "resin", "resort", "resort town", "restaurant kitchen", "restaurant patio", "restroom", "retail", "retriever", "retro", "reveal", "rhinoceros", "rhododendron", "rib", "ribbon", "rice", "rice cooker", "rice field", "ride", "ridge", "riding", "rifle", "rim", "ring", "riot", "ripple", "rise", "rise building", "river", "river bank", "river boat", "river valley", "riverbed", "road", "road sign", "road trip", "roadside", "roast chicken", "robe", "robin", "robot", "stone", "rock arch", "rock artist", "rock band", "rock climber", "rock climbing", "rock concert", "rock face", "rock formation", "rocker", "rocket", "rocking chair", "rocky", "rodent", "rodeo", "rodeo arena", "roe", "roe deer", "roller", "coaster", "roller skate", "roller skates", "rolling pin", "romance", "romantic", "roof", "roof garden", "room", "room divider", "root", "root beer", "rope bridge", "rosary", "rose", "rosemary", "rosy cloud", "rottweiler", "round table", "router", "row", "rowan", "royal", "rubber stamp", "rubble", "rubik's cube", "ruby", "ruffle", "rugby", "rugby ball", "rugby player", "ruins", "ruler", "rum", "run", "runner", "running shoe", "rural", "rust", "rustic", "rye", "sack", "saddle", "saddlebag", "safari", "safe", "safety vest", "sage", "sail", "sailboat", "sailing", "sailor", "squirrel monkey", "sake", "salad", "salad bowl", "salamander", "salami", "sale", "salmon", "salon", "salsa", "salt", "salt and pepper shakers", "salt lake", "salt marsh", "salt shaker", "salute", "samoyed", "samurai", "sand", "sand bar", "sand box", "sand castle", "sand sculpture", "sandal", "sandwich", "sanitary napkin", "santa claus", "sapphire", "sardine", "sari", "sashimi", "satay", "satchel", "satellite", "satin", "sauce", "saucer", "sauna", "sausage", "savanna", "saw", "sawbuck", "sax", "saxophonist", "scaffold", "scale", "scale model", "scallop", "scar", "strawman", "scarf", "scene", "scenery", "schnauzer", "school", "school bus", "school uniform", "schoolhouse", "schooner", "science", "science fiction film", "science museum", "scientist", "scissors", "wall lamp", "scone", "scoop", "scooter", "score", "scoreboard", "scorpion", "scout", "scrambled egg", "scrap", "scraper", "scratch", "screen", "screen door", "screenshot", "screw", "screwdriver", "scroll", "scrub", "scrubbing brush", "sculptor", "sculpture", "sea cave", "sea ice", "sea lion", "sea turtle", "sea urchin", "seabass", "seabed", "seabird", "seafood", "seahorse", "seal", "sea view", "seashell", "seaside resort", "season", "seat", "seat belt", "seaweed", "secretary", "security", "sedan", "see", "seed", "seesaw", "segway", "selfie", "sell", "seminar", "sense", "sensor", "server", "server room", "service", "set", "sewing machine", "shadow", "shake", "shaker", "shampoo", "shape", "share", "shark", "sharpener", "sharpie", "shaver", "shaving cream", "shawl", "shear", "shears", "sheep", "sheet", "sheet music", "shelf", "shell", "shellfish", "shelter", "shelve", "shepherd", "sherbert", "shiba inu", "shine", "shipping", "shipping container", "shipwreck", "shipyard", "shirt", "shirtless", "shoal", "shoe", "shoe box", "shoe shop", "shoe tree", "shoot", "shooting basketball guard", "shop window", "shopfront", "shopper", "shopping", "shopping bag", "shopping basket", "shopping cart", "mall", "shopping street", "shore", "shoreline", "short", "short hair", "shorts", "shot glass", "shotgun", "shoulder", "shoulder bag", "shovel", "showcase", "shower", "shower cap", "shower curtain", "shower door", "shower head", "shredder", "shrew", "shrimp", "shrine", "shrub", "shutter", "siamese", "siberia", "sibling", "side", "side cabinet", "side dish", "sidecar", "sideline", "siding", "sign", "signage", "signal", "signature", "silk", "silk stocking", "silo", "silver", "silver medal", "silverware", "sing", "singe", "singer", "sink", "sip", "sit", "sitting", "skate park", "skateboard", "skateboarder", "skater", "skating rink", "skeleton", "sketch", "skewer", "ski", "ski boot", "ski equipment", "ski jacket", "ski lift", "ski pole", "ski resort", "snowboard", "skier", "skiing shoes", "skin", "skull", "skullcap", "sky", "sky tower", "skylight", "skyline", "skyscraper", "slalom", "slate", "sleigh", "sleep", "sleeping bag", "sleepwear", "sleeve", "slice", "slide", "slider", "sling", "slope", "slot", "slot machine", "sloth", "slow cooker", "slug", "slum", "smell", "smile", "smoke", "snack", "snail", "snake", "snapper", "snapshot", "snorkel", "snout", "snow", "snow leopard", "snow mountain", "snowball", "snowboarder", "snowfield", "snowflake", "snowman", "snowmobile", "snowplow", "snowshoe", "snowy", "soap", "soap bubble", "soap dispenser", "soccer goalkeeper", "socialite", "sock", "socket", "soda", "softball", "software", "solar battery", "soldier", "solo", "solution", "sombrero", "song", "sound", "soup", "soup bowl", "soupspoon", "sour cream", "souvenir", "soybean milk", "spa", "space", "space shuttle", "space station", "spacecraft", "spaghetti", "span", "wrench", "spark", "sparkle", "sparkler", "sparkling wine", "sparrow", "spatula", "speaker", "spectator", "speech bubble", "speed limit", "speed limit sign", "speedboat", "speedometer", "sphere", "spice", "spice rack", "spider", "spider web", "spike", "spin", "spinach", "spire", "splash", "sponge", "spoon", "sport association", "sport equipment", "sport team", "sports ball", "sports equipment", "sports meet", "sportswear", "dot", "spray", "spread", "spring", "spring roll", "sprinkle", "sprinkler", "sprout", "spruce", "spruce forest", "squad", "square", "squash", "squat", "squeeze", "squid", "squirrel", "water gun", "stab", "stable", "stack", "stadium", "staff", "stage", "stage light", "stagecoach", "stain", "stainless steel", "stair", "stairs", "stairwell", "stall", "stallion", "stand", "standing", "staple", "stapler", "star", "stare", "starfish", "starfruit", "starling", "state park", "state school", "station", "stationary bicycle", "stationery", "statue", "steak", "steak knife", "steam", "steam engine", "steam locomotive", "steam train", "steamed bread", "steel", "steering wheel", "stem", "stencil", "step stool", "stereo", "stethoscope", "stew", "stick", "stick insect", "sticker", "still life", "stilt", "stingray", "stir", "stirrer", "stirrup", "sew", "stock", "stocking", "stomach", "stone building", "stone carving", "stone house", "stone mill", "stool", "stop", "stop at", "stop light", "stop sign", "stop watch", "traffic light", "storage box", "storage room", "tank", "store", "storefront", "stork", "storm", "storm cloud", "stormy", "stove", "poker", "straddle", "strainer", "strait", "strap", "straw", "straw hat", "strawberry", "stream", "street art", "street artist", "street corner", "street dog", "street food", "street light", "street market", "street photography", "street scene", "street sign", "street vendor", "stretch", "stretcher", "strike", "striker", "string", "string cheese", "strip", "stripe", "stroll", "structure", "studio", "studio shot", "stuff", "stuffed animal", "stuffed toy", "stuffing", "stump", "stunning", "stunt", "stupa", "style", "stylus", "submarine", "submarine sandwich", "submarine water", "suburb", "subway", "subway station", "subwoofer", "succulent", "suede", "sugar", "sugar bowl", "sugar cane", "sugar cube", "suit", "suite", "summer", "summer evening", "summit", "sun", "sun hat", "sunbathe", "sunday", "sundial", "sunflower", "sunflower field", "sunflower seed", "sunglasses", "sunny", "sunrise", "sunset", "sunshade", "sunshine", "super bowl", "sports car", "superhero", "supermarket", "supermarket shelf", "supermodel", "supporter", "surf", "surface", "surfboard", "surfer", "surgeon", "surgery", "surround", "sushi", "sushi bar", "suspenders", "suspension", "suspension bridge", "suv", "swallow", "swallowtail butterfly", "swamp", "swan", "swan boat", "sweat pant", "sweatband", "sweater", "sweatshirt", "sweet", "sweet potato", "swim", "swim cap", "swimmer", "swimming hole", "swimming pool", "swing", "swing bridge", "swinge", "swirl", "switch", "swivel chair", "sword", "swordfish", "symbol", "symmetry", "synagogue", "syringe", "syrup", "system", "t shirt", "t-shirt", "tabasco sauce", "tabby", "table tennis racket", "table top", "tablecloth", "tablet computer", "tableware", "tachometer", "tackle", "taco", "tae kwon do", "tai chi", "tail", "tailor", "take", "takeoff", "talk", "tambourine", "tan", "tangerine", "tape", "tapestry", "tarmac", "taro", "tarp", "tart", "tassel", "taste", "tatami", "tattoo", "tattoo artist", "tavern", "tea", "tea bag", "tea party", "tea plantation", "tea pot", "tea set", "teach", "teacher", "teacup", "teal", "team photo", "team presentation", "tear", "technician", "technology", "teddy", "tee", "teenager", "telegraph pole", "zoom lens", "telescope", "television", "television camera", "television room", "television studio", "temperature", "temple", "tempura", "tennis", "tennis court", "tennis match", "tennis net", "tennis player", "tennis racket", "tent", "tequila", "terminal", "terrace", "terrain", "terrarium", "territory", "test", "test match", "test tube", "text", "text message", "textile", "texture", "thanksgiving", "thanksgiving dinner", "theater", "theatre actor", "therapy", "thermometer", "thermos", "thermos bottle", "thermostat", "thicket", "thimble", "thing", "thinking", "thistle", "throne", "throne room", "throw", "throw pillow", "thunder", "thunderstorm", "thyme", "tiara", "tick", "ticket", "ticket booth", "tide pool", "tie", "tiger", "tight", "tile", "tile flooring", "tile roof", "tile wall", "tin", "tinfoil", "tinsel", "tiramisu", "tire", "tissue", "toast", "toaster", "tobacco", "tobacco pipe", "toddler", "toe", "tofu", "toilet bowl", "toilet seat", "toiletry", "tokyo tower", "tomato", "tomato sauce", "tomato soup", "tomb", "tong", "tongs", "tool", "toolbox", "toothbrush", "toothpaste", "toothpick", "topiary garden", "topping", "torch", "tornado", "tortilla", "tortoise", "tote bag", "totem pole", "totoro", "toucan", "touch", "touchdown", "tour", "tour bus", "tour guide", "tourist", "tourist attraction", "tournament", "tow truck", "towel", "towel bar", "tower block", "tower bridge", "town", "town square", "toy", "toy car", "toy gun", "toyshop", "track", "tractor", "trade", "tradition", "traditional", "traffic", "traffic cone", "traffic congestion", "traffic jam", "traffic sign", "trail", "trailer", "trailer truck", "train", "train bridge", "train car", "train interior", "train track", "train window", "trainer", "training", "training bench", "training ground", "trolley", "trampoline", "transformer", "transparency", "travel", "tray", "treadmill", "treat", "tree", "tree branch", "tree farm", "tree frog", "tree house", "tree root", "tree trunk", "trial", "triangle", "triathlon", "tribe", "tributary", "trick", "tricycle", "trim", "trio", "tripod", "trombone", "troop", "trophy", "trophy cup", "tropic", "trout", "truck", "truck driver", "tub", "tube", "tugboat", "tulip", "tuna", "tundra", "tunnel", "turbine", "turkey", "turn", "turnip", "turquoise", "turret", "turtle", "tusk", "tv actor", "tv cabinet", "tv drama", "tv genre", "tv personality", "tv show", "tv sitcom", "tv tower", "twig", "twilight", "twin", "twine", "twist", "type", "type on", "typewriter", "ukulele", "ultraman", "umbrella", "underclothes", "underwater", "unicorn", "uniform", "universe", "university", "up", "urban", "urinal", "urn", "use", "utensil", "utility room", "vacuum", "valley", "valve", "vampire", "van", "vanilla", "vanity", "variety", "vase", "vault", "vector cartoon illustration", "vector icon", "vegetable", "vegetable garden", "vegetable market", "vegetation", "vehicle", "veil", "vein", "velvet", "vending machine", "vendor", "vent", "vespa", "vessel", "vest", "vet", "veteran", "veterinarians office", "viaduct", "video", "video camera", "video game", "videotape", "view mirror", "vigil", "villa", "village", "vine", "vinegar", "vineyard", "violence", "violet", "violin", "violinist", "violist", "vision", "visor", "vodka", "volcano", "volleyball", "volleyball court", "volleyball player", "volunteer", "voyage", "vulture", "waffle", "waffle iron", "wagon", "wagon wheel", "waist", "waiter", "waiting hall", "waiting room", "walk", "walking", "walking cane", "wall clock", "wallpaper", "walnut", "walrus", "war", "warehouse", "warm", "warning sign", "warrior", "warship", "warthog", "wash", "washer", "washing", "washing machine", "wasp", "waste", "waste container", "watch", "water", "water bird", "water buffalo", "water cooler", "water drop", "water feature", "water heater", "water level", "water lily", "water park", "water pipe", "water purifier", "water ski", "water sport", "water surface", "water tower", "watercolor", "watercolor illustration", "watercolor painting", "waterfall", "watering can", "watermark overlay stamp", "watermelon", "waterproof jacket", "waterway", "wave", "wax", "weapon", "wear", "weather", "vane", "web", "webcam", "wedding", "wedding ring", "wedding bouquet", "wedding cake", "wedding couple", "wedding invitation", "wedding party", "wedding photo", "wedding photographer", "wedding photography", "wedding reception", "wedge", "weed", "weight", "weight scale", "welder", "well", "western food", "western restaurant", "wet", "wet bar", "wet suit", "wetland", "wetsuit", "whale", "whale shark", "wheat", "wheat field", "wheel", "wheelchair", "wheelie", "whipped cream", "whisk", "whisker", "whiskey", "whistle", "white", "white house", "white wine", "whiteboard", "wicket", "wide", "wield", "wig", "Wii", "Wii controller", "wild", "wildebeest", "wildfire", "wildflower", "wildlife", "willow", "wind", "wind chime", "wind farm", "wind turbine", "windmill", "window", "window box", "window display", "window frame", "window screen", "window seat", "window sill", "wiper", "windshield", "windy", "wine bottle", "wine cooler", "wine cabinet", "wine cellar", "wine glass", "wine rack", "wine tasting", "winery", "wing", "winter", "winter melon", "winter morning", "winter scene", "winter sport", "winter storm", "wire", "wisteria", "witch", "witch hat", "wok", "wolf", "woman", "wood", "wood duck", "wood floor", "wood wall", "wood-burning stove", "wooden spoon", "woodland", "woodpecker", "woodworking plane", "wool", "job", "work card", "workbench", "worker", "workplace", "workshop", "world", "worm", "worship", "wound", "wrap", "wrap dress", "wrapping paper", "wrestle", "wrestler", "wrinkle", "wristband", "write", "writer", "writing", "writing brush", "writing desk", "yacht", "yak", "yard", "yellow", "yoga", "yoga mat", "yoghurt", "yoke", "yolk", "youth", "youth hostel", "yurt", "zebra", "zebra crossing", "zen garden", "zip", "zipper", "zombie", "zongzi", "zoo", ]

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
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()