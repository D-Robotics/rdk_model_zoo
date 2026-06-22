import torch
import torch.onnx
import onnx
import timm
from onnxsim import simplify
from timm.models import create_model
from pathlib import Path

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None

def count_parameters(onnx_model_path):
    model_onnx = onnx.load(onnx_model_path)
    initializer = model_onnx.graph.initializer
    total_params = 0
    for tensor in initializer:
        dims = tensor.dims
        params = 1
        for dim in dims:
            params *= dim
        total_params += params
    return total_params

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'tf_efficientnet_lite0' # 修改为你的模型名称
    weights_path = Path("/path/to/your/weights.safetensors") # 修改为你的权重文件路径
    output_directory = "." # ONNX 文件输出目录，默认为当前目录。如果遇到权限问题，请修改为你有写入权限的目录。
    dummy_input = torch.randn(1, 3, 224, 224, device=device) # 修改为你的输入尺寸

    # 创建模型
    model = create_model(model_name, pretrained=False)

    # 加载权重
    if weights_path.suffix == ".safetensors":
        state_dict = load_safetensors_file(weights_path, device=str(device))
    else:
        state_dict = torch.load(weights_path, map_location=device)

    # 处理嵌套的state_dict
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # 移除module.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    data_config = timm.data.resolve_data_config({}, model=model)

    # 导出为ONNX
    onnx_path = Path(output_directory) / f"{model_name}.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
    )

    print(data_config['input_size'])
    print(data_config['mean'])
    print(data_config['std'])

    # 简化ONNX模型
    model_simp, _ = simplify(onnx.load(str(onnx_path)))
    onnx.save(model_simp, str(onnx_path))

    # 计算参数量
    total_params = count_parameters(str(onnx_path))
    print(f"模型总参数量: {total_params}")
