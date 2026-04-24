#!/bin/bash

set -e

MODEL_DIR=$(dirname "$0")

wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/RepViT_m0_9_224x224_nv12.bin
wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/RepViT_m1_0_224x224_nv12.bin
wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/RepViT_m1_1_224x224_nv12.bin
