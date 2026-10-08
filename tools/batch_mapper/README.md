English | [简体中文](README_cn.md)

# Batch Mapper

Batch Mapper is used to batch compile ONNX models in a specific directory according to a certain YAML configuration. Batch will help you complete the following steps:
- Traverse all ONNX model files in the directory
- Generate corresponding YAML files in the current directory
- Start compiling, with compilation results unified under the `ws_path` folder
- After compilation ends, copy the compilation logs to the compilation work directory
- Remove dequantization nodes as required
- After removing dequantization nodes, copy the removal logs to the compilation work directory
- Copy the compiled results to the release directory

How to run the compilation in the background using tmux?
```bash
# Install tmux
sudo apt update
sudo apt install tmux
# Use tmux
tmux new -s batch_mapper
# Run Docker and command, for example
sudo docker run --gpus all -it -v /ws:/open_explorer hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8 python3 batch_mapper.py
# Detach from tmux
Press Ctrl+B, then press D
# Exit terminal
exit
# List tmux sessions
tmux ls
# Reattach to tmux
tmux attach -t batch_mapper
# Kill tmux session
tmux kill-session -t batch_mapper
```
