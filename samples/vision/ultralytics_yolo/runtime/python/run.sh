#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
TASK="${1:-detect}"

if [ "$#" -gt 0 ]; then
  shift
fi

SOC="s100"
if [ -r /sys/class/boardinfo/soc_name ]; then
  SOC="$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)"
fi

BOARD_TYPE="${SOC}"
if [ -r /sys/class/boardinfo/board_type ]; then
  BOARD_TYPE="$(tr 'A-Z' 'a-z' </sys/class/boardinfo/board_type)"
fi

MODEL_MARCH="nash-e"
MODEL_SUFFIX="nashe"
if [[ "${SOC}" == "s100p" || "${BOARD_TYPE}" == *"p"* ]]; then
  MODEL_MARCH="nash-m"
  MODEL_SUFFIX="nashm"
fi

echo "SOC        : ${SOC}"
echo "Board type : ${BOARD_TYPE}"
echo "Model march: ${MODEL_MARCH}"
echo "TASK       : ${TASK}"

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "scipy==1.15.3"
)

check_and_install() {
  local pkg="$1"
  local name="${pkg%%==*}"
  local version="${pkg##*==}"
  local installed_version
  installed_version=$(pip3 show "${name}" 2>/dev/null | awk '/^Version:/{print $2}')

  if [[ "${installed_version}" == "${version}" ]]; then
    echo "${name}==${version} already installed, skip"
  else
    if [[ -n "${installed_version}" ]]; then
      echo "${name} version mismatch (installed: ${installed_version}, need: ${version})"
    else
      echo "${name} not installed, installing ${version}"
    fi
    pip3 install "${name}==${version}"
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "${pkg}"
done

mkdir -p "${MODEL_DIR}/${MODEL_MARCH}"

USER_MODEL_PATH=""
args=("$@")
for ((idx = 0; idx < ${#args[@]}; idx++)); do
  if [[ "${args[$idx]}" == "--model-path" && $((idx + 1)) -lt ${#args[@]} ]]; then
    USER_MODEL_PATH="${args[$((idx + 1))]}"
    break
  fi
done

case "${TASK}" in
  detect)
    DEFAULT_MODEL_TYPE="yolo11"
    DEFAULT_MODEL_FILE="yolo11n_detect_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  seg)
    DEFAULT_MODEL_TYPE="yolo11"
    DEFAULT_MODEL_FILE="yolo11n_seg_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  pose)
    DEFAULT_MODEL_TYPE="yolo11"
    DEFAULT_MODEL_FILE="yolo11n_pose_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  cls)
    DEFAULT_MODEL_TYPE="yolo11"
    DEFAULT_MODEL_FILE="yolo11n_cls_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  *)
    echo "Unsupported task: ${TASK}"
    echo "Usage: bash run.sh [detect|seg|pose|cls] [main.py options...]"
    exit 1
    ;;
esac

MODEL_PATH="${USER_MODEL_PATH:-../../model/${MODEL_MARCH}/${DEFAULT_MODEL_FILE}}"
echo "Model path : ${MODEL_PATH}"

if [[ -z "${USER_MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
  echo "Default model not found, downloading..."
  (cd "${MODEL_DIR}" && bash download_model.sh "${SOC}" "${DEFAULT_MODEL_TYPE}" "${TASK}" "n")
elif [[ -n "${USER_MODEL_PATH}" && ! -f "${USER_MODEL_PATH}" ]]; then
  echo "User model path does not exist: ${USER_MODEL_PATH}"
  echo "Download it first with ../../model/download_model.sh or provide an existing --model-path."
  exit 1
else
  echo "Model already exists, skip download"
fi

cd "${SCRIPT_DIR}"
python3 main.py \
  --task "${TASK}" \
  --model-path "${MODEL_PATH}" \
  --test-img ../../test_data/bus.jpg \
  --label-file ../../test_data/coco_classes.names \
  --img-save-path result.jpg \
  --priority 0 \
  --bpu-cores 0 \
  --score-thres 0.25 \
  --nms-thres 0.45 \
  "$@"
