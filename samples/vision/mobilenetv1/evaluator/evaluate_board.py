"""Evaluate a pinned MobileNetV1 board artifact (100, 125) on its frozen dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from samples.vision.mobilenetv1.runtime.python.classify import MobileNetV1Classifier
from utils.tools.classification.board_evaluate import main

if __name__ == "__main__":
    main(MobileNetV1Classifier)
