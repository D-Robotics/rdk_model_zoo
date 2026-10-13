"""Evaluate a pinned MobileNetV3 board artifact (large, small) on its frozen dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from samples.vision.mobilenetv3.runtime.python.classify import MobileNetV3Classifier
from utils.tools.classification.board_evaluate import main

if __name__ == "__main__":
    main(MobileNetV3Classifier)
