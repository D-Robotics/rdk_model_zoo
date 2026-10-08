[English](README.md) | 简体中文

# Batch Mapper

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

