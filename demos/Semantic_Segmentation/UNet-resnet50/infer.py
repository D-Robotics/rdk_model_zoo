import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from time import time
import argparse

class UNetInference:
    def __init__(self, model_path: str, num_classes: int = 21):
        """
        加载模型并初始化参数（参考YOLOE风格）
        """
        self.num_classes = num_classes
        
        # 加载BPU模型
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_path)
            print(f"\033[1;32m[UNet] 模型加载成功，耗时: {1000*(time()-begin_time):.2f}ms\033[0m")
        except Exception as e:
            print(f"\033[1;31m[UNet] 模型加载失败: {e}\033[0m")
            exit(1)
        
        # 打印输入信息
        print("\033[1;32m-> 输入Tensors:\033[0m")
        for i, inp in enumerate(self.quantize_model[0].inputs):
            print(f"  input[{i}]: name={inp.name}, shape={inp.properties.shape}, type={inp.properties.dtype}")
            self.input_h = inp.properties.shape[2]
            self.input_w = inp.properties.shape[3]
        
        # 打印输出信息
        print("\033[1;32m-> 输出Tensors:\033[0m")
        for i, out in enumerate(self.quantize_model[0].outputs):
            print(f"  output[{i}]: name={out.name}, shape={out.properties.shape}, type={out.properties.dtype}")
            self.out_c = out.properties.shape[1]  # 21类
            self.out_h = out.properties.shape[2]
            self.out_w = out.properties.shape[3]
        
        print(f"[UNet] 输入尺寸: {self.input_w}x{self.input_h}, 输出类别: {self.out_c}")

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        BGR -> NV12 转换（完全参考YOLOE实现）
        """
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        
        # BGR -> I420 (YUV420P)
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        
        # 分离Y和UV
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        
        # UV打包（NV12格式：YYYYYYYY UVUVUVUV）
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        
        # 组装NV12
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        
        return nv12

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        预处理：Resize -> BGR2NV12（参考YOLOE preprocess_yuv420sp）
        """
        self.orig_h, self.orig_w = img.shape[:2]
        
        # Resize到模型输入尺寸（直接resize，不做letterbox）
        img_resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        
        # 转NV12
        begin_time = time()
        nv12 = self.bgr2nv12(img_resized)
        print(f"\033[1;31m[UNet] 预处理耗时: {1000*(time()-begin_time):.2f}ms\033[0m")
        
        return nv12

    def forward(self, input_tensor: np.ndarray):
        """
        执行推理（参考YOLOE forward）
        """
        begin_time = time()
        outputs = self.quantize_model[0].forward(input_tensor)
        print(f"\033[1;31m[UNet] 推理耗时: {1000*(time()-begin_time):.2f}ms\033[0m")
        return outputs

    def postprocess(self, outputs):
        """
        后处理：获取分割掩膜（UNet特定：argmax）
        """
        begin_time = time()
        
        # 获取输出buffer（numpy数组）
        # UNet输出形状: [1, 21, 512, 512]
        output = outputs[0].buffer  # 直接获取numpy数组
        
        # Reshape到逻辑形状（去除batch维度，得到[21, 512, 512]）
        output = output.reshape(self.out_c, self.out_h, self.out_w)
        
        # Argmax获取每个像素的类别（0-20）
        mask = np.argmax(output, axis=0).astype(np.uint8)
        
        # Resize回原图尺寸（最近邻插值保持类别）
        mask = cv2.resize(mask, (self.orig_w, self.orig_h), interpolation=cv2.INTER_NEAREST)
        
        print(f"\033[1;31m[UNet] 后处理耗时: {1000*(time()-begin_time):.2f}ms\033[0m")
        return mask

    def visualize(self, img: np.ndarray, mask: np.ndarray, alpha: float = 0.6):
        """
        可视化分割结果（21类颜色映射）
        """
        # 生成颜色表（参考YOLOE rdk_colors风格）
        np.random.seed(42)
        colors = np.random.randint(50, 255, (self.num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # 背景黑色
        
        # 创建彩色掩膜
        color_mask = np.zeros_like(img)
        for cls_id in range(self.num_classes):
            color_mask[mask == cls_id] = colors[cls_id]
        
        # 混合
        result = cv2.addWeighted(img, 1-alpha, color_mask, alpha, 0)
        
        # 添加图例（前5个类别）
        for i in range(min(5, self.num_classes)):
            cv2.rectangle(result, (10, 30+i*25), (30, 50+i*25), colors[i].tolist(), -1)
            cv2.putText(result, f"Class {i}", (35, 45+i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/sunrise/project/UNet/UNet-resnet-deploy_512x512_nv12_x3.bin', help='模型路径')
    parser.add_argument('--img', type=str, default='./1.jpg', help='测试图片路径')
    parser.add_argument('--save', type=str, default='unet_result.jpg', help='保存结果路径')
    parser.add_argument('--classes', type=int, default=21, help='分割类别数')
    args = parser.parse_args()
    
    # 初始化
    model = UNetInference(args.model, args.classes)
    
    # 读取图片
    img = cv2.imread(args.img)
    if img is None:
        print(f"\033[1;31m[Error] 无法读取图片: {args.img}\033[0m")
        return
    
    # 推理流程
    input_tensor = model.preprocess(img)
    outputs = model.forward(input_tensor)
    mask = model.postprocess(outputs)
    
    # 可视化
    result = model.visualize(img, mask)
    
    # 保存
    cv2.imwrite(args.save, result)
    cv2.imwrite(args.save.replace('.jpg', '_mask.png'), mask * 12)  # 乘12增加对比度
    print(f"\033[1;32m[UNet] 结果已保存: {args.save}\033[0m")
    print(f"[UNet] 预测类别: {np.unique(mask)}")

if __name__ == "__main__":
    main()
