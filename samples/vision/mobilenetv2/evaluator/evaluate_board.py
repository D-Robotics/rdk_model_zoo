"""Evaluate a pinned MobileNetV2 board artifact (100, 140) on its frozen dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.tools.mobilenet.board_evaluate import main

if __name__ == "__main__":
    main()
