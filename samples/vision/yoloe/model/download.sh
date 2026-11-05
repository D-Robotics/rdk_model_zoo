#!/usr/bin/env bash
# S100 YOLOE-11 publications (URLs live in docs/release/s/models.yaml):
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11m_seg_pf_nashe_640x640_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11l_seg_pf_nashe_640x640_nv12.hbm
# Example: bash download.sh --target s100 --variant 11m
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/download.py" "$@"
