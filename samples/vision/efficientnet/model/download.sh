#!/bin/bash

set -e

MODEL_DIR=$(dirname "$0")

wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B2_224x224_nv12.bin
wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B3_224x224_nv12.bin
wget -P "$MODEL_DIR" https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EfficientNet_B4_224x224_nv12.bin
