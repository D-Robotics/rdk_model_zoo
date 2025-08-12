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

# 注意: 此程序在RDK板端运行
# Attention: This program runs on RDK board.

from Ultralytics_YOLO_Pose_YUV420SP import *

import json

import argparse
import logging 
import os


# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_bin_models/pose/yolov8n-pose_detect_bayes-e_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--classes-num', type=int, default=1, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--nkpt', type=int, default=17, help='num of keypoints.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='confidence threshold.')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), 
                        default=[8, 16 ,32],
                        help='--strides 8, 16, 32')
    parser.add_argument('--image-path', type=str, default="../../../resource/datasets/COCO2017/val2017", help='COCO2017 val source image path.')
    parser.add_argument('--result-image-dump', type=bool, default=False, help='dump image result or not')
    parser.add_argument('--result-image-path', type=str, default="coco2017_image_result", help='COCO2017 val image result saving path.')
    parser.add_argument('--json-path', type=str, default="yolov8n-pose_detect_bayes-e_640x640_nv12_py_coco2017_val_pridect.json", help='convert to json save path.')
    parser.add_argument('--max-num', type=int, default=50000, help='max num of images which will be precessed.')
    opt = parser.parse_args()
    logger.info(opt)
    begin_time = time()
    run(opt)
    print("\033[1;31m" + f"Total time = {(time() - begin_time)/60:.2f} min" + "\033[0m")

def run(opt):
    # id -> coco_id
    coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]


    # 实例化
    model = Ultralytics_YOLO_Pose_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,   # default: 1
        nms_thres=opt.nms_thres,       # default: 0.7
        score_thres=opt.score_thres,   # default: 0.25
        reg=opt.reg,                   # default: 16
        strides=opt.strides,           # default: [8, 16, 32]
        nkpt=opt.nkpt                  # default: 17
        )
    kpt_conf_inverse = -np.log(1/opt.kpt_conf_thres - 1)

    # 创建 result* 目录存储结果
    if opt.result_image_dump:
        for i in range(9999):
            result_image_path = f"{opt.result_image_path}_{i}"
            if not os.path.exists(result_image_path):
                os.makedirs(result_image_path)
                logger.info("\033[1;32m" + f"Created directory Successfully: \"{result_image_path}\"" + "\033[0m")
                break
    
    # 生成pycocotools的预测结果json文件
    pridect_json = []

    # 板端直接推理所有验证集，并生成标签和渲染结果 (串行程序)
    img_names = os.listdir(opt.image_path)
    img_num = len(img_names)
    control = opt.max_num
    for cnt, img_name in enumerate(img_names, 1):
        # 流程控制
        if control < 1:
            break
        control -= 1

        logger.info("\033[1;32m" + f"[{cnt}/{img_num}] Processing image: \"{img_name}\"" + "\033[0m")
        # 端到端推理
        img = cv2.imread(os.path.join(opt.image_path, img_name))
        input_tensor = model.preprocess_yuv420sp(img)
        outputs = model.c2numpy(model.forward(input_tensor))
        results = model.postProcess(outputs)
        # 渲染结果并保存
        if opt.result_image_dump:
            for class_id, score, x1, y1, x2, y2, kpts in results:
                logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
                draw_detection(img, (x1, y1, x2, y2), score, class_id)
                for j in range(opt.nkpt):
                    x, y = int(kpts[j][0]), int(kpts[j][1])
                    if kpts[j][2] < kpt_conf_inverse:
                        continue
                    cv2.circle(img, (x,y), 5, (0, 0, 255), -1)
                    cv2.circle(img, (x,y), 2, (0, 255, 255), -1)
                    cv2.putText(img, "%d"%j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3, cv2.LINE_AA)
                    cv2.putText(img, "%d"%j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                save_path = os.path.join(result_image_path, img_name[:-4] + "_result.jpg")
            cv2.imwrite(save_path, img)
            logger.info("\033[1;32m" + f"result image saved: \"./{save_path}\"" + "\033[0m")
        id_cnt = 0
        for class_id, score, x1, y1, x2, y2, kpts in results:
            # logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
            kpts_json = []
            for j in range(opt.nkpt):
                x, y = int(kpts[j][0]), int(kpts[j][1])
                if kpts[j][2] < kpt_conf_inverse:
                    vis = 2
                else:
                    vis = 1
                kpts_json+=[x, y, vis]

            pridect_json.append({
                    'image_id': int(img_name[:-4]),
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(coco_id[class_id]),
                    'id': id_cnt,
                    "score": float(score),
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    'keypoints': kpts_json
                })
            id_cnt += 1

    # 保存标签
    with open(opt.json_path, 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)
    
    logger.info("\033[1;32m" + f"result label saved: \"{opt.json_path}\"" + "\033[0m")
    

if __name__ == "__main__":
    main()