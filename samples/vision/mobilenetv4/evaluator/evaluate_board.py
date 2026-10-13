"""Evaluate a pinned MobileNetV4 board artifact (small, medium, large) on its frozen dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.tools.classification.board_evaluate import main

if __name__ == "__main__":
    main()
