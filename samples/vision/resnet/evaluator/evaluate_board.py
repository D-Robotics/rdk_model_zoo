"""Evaluate a pinned ResNet board artifact (18, 34, 50, 101, 152) on its frozen dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from samples.vision.resnet.runtime.python.classify import ResNetClassifier
from utils.tools.classification.board_evaluate import main

if __name__ == "__main__":
    main(ResNetClassifier)
