![](imgs/RDK_LLM_Solution.jpg)

[English](./README.md) | 简体中文

# RDK LLM Solution

## 摘要

目前调研来看，带NPU的SBC运行YOLO这种视觉模型，NPU的速度是CPU速度的几十到几百倍，但是运行LLM这种语言模型，NPU的速度最多也只有CPU的1.2～1.6倍，NPU方案会限制上下文长度到256等非常短的长度，同时让NPU运行LLM会花费更多的人力和物力成本，所以现阶段使用CPU运行语言任务，NPU专注于视觉任务，也不失为一种较为合理的搭配方式。

本文参考社区大佬@潜沉10的文章，在RDK X5上使用llama.cpp框架运行语言大模型。分别测试了thread_num=4和8的情况，测试共8家的语言大模型，一共63个，涵盖0.5B到14B等不同参数量，其中能跑到10token/s以上的9个，5token/s以上的14个，1 token/s以上的52个。使用的GGUF模型文件也保存在了百度网盘，欢迎大家在RDK X5上来尝试，希望大家玩的开心。

## 测试结果
飞书文档: [https://horizonrobotics.feishu.cn/docx/LQU9dYyjcoXJ9hxJdUYc2l4InEf](https://horizonrobotics.feishu.cn/docx/LQU9dYyjcoXJ9hxJdUYc2l4InEf)

## 使用方式

### 参考:

RDK使用llama.cpp运行语言大模型: [https://developer.d-robotics.cc/forumDetail/256524800871478519](https://developer.d-robotics.cc/forumDetail/256524800871478519)

llama.cpp: [https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)


GLM的部分模型: [https://huggingface.co/THUDM/glm-edge-1.5b-chat-gguf/blob/main/README_zh.md](https://huggingface.co/THUDM/glm-edge-1.5b-chat-gguf/blob/main/README_zh.md)

### 下载编译llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### 添加环境变量

```bash
PATH=/media/rootfs/99_projects/test_llama.cpp/llama.cpp/build/bin:$PATH
```

### 使用以下命令运行

使用固定提问，使得生成128个token，然后Ctrl+C查看性能结果。

```bash
llama-cli \
-m <path to your gguf model> \
-n 512 -c 2048 \
-p "You are a helpful assistant" -co -cnv \
--threads 8 

# 如果是RWKV这种不带KV Cache的LLM，需要添加以下参数
llama-cli \
-m rwkv-6-finch-3b-Q8_0.gguf \
-n 512 -c 2048 \
-p "You are a helpful assistant" -co -cnv \
--threads 8 --no-context-shift
```