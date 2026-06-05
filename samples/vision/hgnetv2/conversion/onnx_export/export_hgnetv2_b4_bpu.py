import timm
import torch

# 1. 加载模型并设置为推理模式
model = timm.create_model('hgnetv2_b4.ssld_stage2_ft_in1k', pretrained=True)
model.eval()

# 2. 创建示例输入 (batch_size=1, 3个颜色通道, 图像尺寸224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 4. 导出模型
torch.onnx.export(
    model,
    dummy_input,
    "hgnetv2_b4.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=11,
    dynamo=False
)

print("模型已成功导出为 hgnetv2_b4.onnx")