#!/bin/bash
# Download pre-compiled Ultralytics YOLO models for RDK platforms.
#
# Usage:
#   bash download_model.sh [soc] [yolo_type] [task] [model_size]
#
# Arguments:
#   soc       - Target SoC: s100 (default) or s100p
#   yolo_type - Model variant: yolov5u, yolov8, yolov10, yolo11, yolo12
#   task      - Task type: detect, seg, pose, cls
#   model_size - Model scale: n, s, m, l, x (default: n); yolov10 also supports b
#
# Examples:
#   bash download_model.sh s100 yolo11 detect
#   bash download_model.sh s100p yolov5u detect s

set -e

if [[ -n "$1" ]]; then
  SOC="$1"
else
  if [[ -r /sys/class/boardinfo/soc_name ]]; then
    SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
  else
    SOC="s100"
  fi
fi
YOLO_TYPE="${2:-yolo11}"
TASK="${3:-detect}"
MODEL_SIZE="${4:-n}"
if [[ "$MODEL_SIZE" == "b" && "$YOLO_TYPE" != "yolov10" ]]; then
  echo "model_size b is only available for yolov10"
  exit 1
fi
if [[ -r /sys/class/boardinfo/board_type ]]; then
  BOARD_TYPE=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/board_type)
else
  BOARD_TYPE="$SOC"
fi

# Model suffix differs by platform: S100 uses nashe; S100P uses nashm.
MODEL_MARCH="nash-e"
MODEL_SUFFIX="nashe"
if [[ "$SOC" == "s100p" || "$BOARD_TYPE" == *"p"* ]]; then
  MODEL_MARCH="nash-m"
  MODEL_SUFFIX="nashm"
fi

# Determine model file and URL base
case "$YOLO_TYPE" in
  yolov5u)
    URL_BASE="Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"
    case "$TASK" in
      detect) MODEL_FILE="yolov5${MODEL_SIZE}u_detect_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      *)      echo "YOLOv5u precompiled public S100 models only provide detect tasks"; exit 1 ;;
    esac
    ;;
  yolov8)
    URL_BASE="Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"
    case "$TASK" in
      detect) MODEL_FILE="yolov8${MODEL_SIZE}_detect_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      seg)    MODEL_FILE="yolov8${MODEL_SIZE}_seg_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      pose)   MODEL_FILE="yolov8${MODEL_SIZE}_pose_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      cls)    MODEL_FILE="yolov8${MODEL_SIZE}_cls_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      *)      echo "Unsupported task: $TASK"; exit 1 ;;
    esac
    ;;
  yolov10)
    URL_BASE="Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"
    case "$TASK" in
      detect) MODEL_FILE="yolov10${MODEL_SIZE}_detect_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      *)      echo "YOLOv10 only supports detect task"; exit 1 ;;
    esac
    ;;
  yolo11)
    URL_BASE="Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"
    case "$TASK" in
      detect) MODEL_FILE="yolo11${MODEL_SIZE}_detect_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      seg)    MODEL_FILE="yolo11${MODEL_SIZE}_seg_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      pose)   MODEL_FILE="yolo11${MODEL_SIZE}_pose_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      cls)    MODEL_FILE="yolo11${MODEL_SIZE}_cls_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      *)      echo "Unsupported task: $TASK"; exit 1 ;;
    esac
    ;;
  yolo12)
    URL_BASE="Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"
    case "$TASK" in
      detect) MODEL_FILE="yolo12${MODEL_SIZE}_detect_${MODEL_SUFFIX}_640x640_nv12.hbm" ;;
      *)      echo "YOLO12 precompiled public S100 models only provide detect tasks"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported yolo_type: $YOLO_TYPE"
    exit 1
    ;;
esac

MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/${URL_BASE}/${MODEL_FILE}"
OUTPUT_DIR="$(dirname "$0")"
OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_MARCH}"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_FILE}"
mkdir -p "${OUTPUT_DIR}"

echo "Downloading model..."
echo "  SOC       : $SOC"
echo "  Board     : $BOARD_TYPE"
echo "  March     : $MODEL_MARCH"
echo "  YOLO type : $YOLO_TYPE"
echo "  Task      : $TASK"
echo "  Size      : $MODEL_SIZE"
echo "  Model file: $MODEL_FILE"
echo "  URL       : $MODEL_URL"
echo "  Output    : $OUTPUT_PATH"

if [[ -f "$OUTPUT_PATH" ]]; then
  echo "Model already exists, skip download"
  exit 0
fi

wget -c "$MODEL_URL" -O "$OUTPUT_PATH"
echo "Model downloaded successfully"
