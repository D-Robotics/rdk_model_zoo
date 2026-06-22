import torch
import torch.onnx
import onnx
import timm
from onnxsim import simplify
from timm.models import create_model

def count_parameters(onnx_model_path):

    model = onnx.load(onnx_model_path)
    initializer = model.graph.initializer
    total_params = 0
    for tensor in initializer:
        dims = tensor.dims
        params = 1
        for dim in dims:
            params *= dim
        total_params += params

    return total_params

if __name__ == "__main__":
    model = create_model('tf_efficientnet_lite1', pretrained=True)
    data_config = timm.data.resolve_data_config({}, model=model)
    model.eval()

    dummy_input = torch.randn(1, 3, 240, 240, device="cpu")
    onnx_file_path = "./tf_efficientnet_lite1.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=False,
        input_names=["data"],
        output_names=["output"],
    )

    print('input:', data_config['input_size'])
    print('mean', data_config['mean'])
    print('std', data_config['std'])

    # Simplify the ONNX model
    model_simp, check = simplify(onnx_file_path)

    if check:
        print("Simplified model is valid.")
        simplified_onnx_file_path = "tf_efficientnet_lite1.onnx"
        onnx.save(model_simp, simplified_onnx_file_path)
        print(f"Simplified model saved to {simplified_onnx_file_path}")
    else:
        print("Simplified model is invalid!")

    onnx_model_path = simplified_onnx_file_path
    total_params = count_parameters(onnx_model_path)
    print(f"Total number of parameters in the model: {total_params}")
