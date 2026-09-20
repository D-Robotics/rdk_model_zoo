#!/usr/bin/env bash
set -e

# Directory to store full COCO dataset
COCO_DIR="coco_full"

echo "Creating COCO dataset directory: ${COCO_DIR}"
mkdir -p "${COCO_DIR}"
cd "${COCO_DIR}"

echo "Downloading COCO 2017 images and annotations..."

# Download images
wget -c --no-check-certificate https://images.cocodataset.org/zips/train2017.zip
wget -c --no-check-certificate https://images.cocodataset.org/zips/val2017.zip

# Download annotations
wget -c --no-check-certificate https://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Download finished."

echo "Extracting archives..."

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

echo "Extraction finished."

echo "Cleaning up zip files..."
rm -f train2017.zip val2017.zip annotations_trainval2017.zip

echo "COCO dataset is ready in ${COCO_DIR}"
