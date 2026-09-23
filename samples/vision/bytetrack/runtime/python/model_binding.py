"""ByteTrack owns published detector identities; tensor math is shared with YOLOv5."""
from pathlib import Path
from samples.vision.yolov5.runtime.python.model_binding import ModelSelection,ModelBinding,bind_model
from samples.vision.yolov5.runtime.python import model_binding as detector
SAMPLE_DIR=Path(__file__).resolve().parents[2]


def resolve_selection(target='auto',**kwargs):
    """Resolve only ByteTrack's exact s100/s100p/s600 detector assets."""
    return detector.resolve_selection(target,consumer='bytetrack',**kwargs)


def list_available_assets(target=None):
    """List supported detector assets, without hardware or network access."""
    return detector.list_available_assets(target,consumer='bytetrack')
