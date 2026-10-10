"""Run the shared MobileNetV1 export workflow from any directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.tools.mobilenet.workflow import main

if __name__ == "__main__":
    main(default_family="mobilenetv1", default_command="export")
