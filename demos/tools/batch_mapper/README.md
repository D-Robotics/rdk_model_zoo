- [BATCH MAPPER CN](#batch-mapper-cn)
- [BATCH MAPPER EN](#batch-mapper-en)


# BATCH MAPPER CN

Batch Mapper用于将某一个目录下的onnx模型批量的按照某一个yaml配置进行编译，batch会帮你完成以下步骤：
- 遍历目录下的所有onnx模型文件
- 生成对应的yaml文件在当前目录
- 开始编译，编译的产物会统一在`ws_path`文件夹下
- 编译结束后，将编译日志cp到编译工作目录
- 按照要求移除反量化节点
- 移除反量化节点后，将移除日志cp到编译工作目录
- 将编译产物拷贝到发布目录

如何在后台挂起编译？
```bash
# 安装tmux
sudo apt update
sudo apt install tmux
# 使用tmux
tmux new -s batch_mapper
# 运行docker和命令, 例如
sudo docker run --gpus all -it -v /ws:/open_explorer hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
python3 batch_mapper.py
python3 batch_mapper.py 2>&1 | tee batch_mapper.txt  # 运行并保存日志
# 断开tmux
按下Ctrl + B, 然后按下D
# 断开terminal
exit
# 查看tmux的会话
tmux ls
# 重新连接到tmux
tmux attach -t batch_mapper
# 关闭tmux会话
tmux kill-session -t batch_mapper
```



# BATCH MAPPER EN

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