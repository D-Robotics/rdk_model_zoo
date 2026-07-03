#!/bin/bash
# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

MODEL_NAME="pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin"
DOWNLOAD_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "$DIR/$MODEL_NAME" ]; then
    echo "Model $MODEL_NAME already exists in $DIR. Skipping download."
else
    echo "Downloading $MODEL_NAME to $DIR..."
    wget -c "$DOWNLOAD_URL" -P "$DIR"
fi
