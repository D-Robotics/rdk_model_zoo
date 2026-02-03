# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import sys
import argparse
import numpy as np
import time

# Add project root to sys.path
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
from bytetrack import ByteTrack, ByteTrackConfig

def main() -> None:
    """
    @brief Run ByteTrack on a video file using YOLOv5 detector.
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/yolov5x_672x672_nv12.hbm',
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--input', type=str, default='../../test_data/track_test.mp4',
                        help='Path to input video.')
    parser.add_argument('--output', type=str, default='result.mp4',
                        help='Path to output video.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes.")
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Detection confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='NMS IoU threshold.')
    parser.add_argument('--track-thresh', type=float, default=0.3,
                        help='Tracking confidence threshold.')
    
    opt = parser.parse_args()

    # Download model if missing
    download_url = f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_{soc}/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"

    file_io.download_model_if_needed(opt.model_path, download_url)

    # Init config
    config = ByteTrackConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres,
        track_thresh=opt.track_thresh
    )

    # Instantiate ByteTrack
    tracker = ByteTrack(config)
    tracker.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model info
    inspect.print_model_info(tracker.detector.model)

    # Open Video
    cap = cv2.VideoCapture(opt.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video {opt.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    # Output writer
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing video: {opt.input} -> {opt.output}")
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t0 = time.time()
        tracks = tracker.predict(frame)
        t1 = time.time()
        
        # Draw tracks
        for t in tracks:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score
            
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Color by ID
            color = ((37 * tid) % 255, (17 * tid) % 255, (29 * tid) % 255)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        
        if frame_id % 30 == 0:
            print(f"Frame {frame_id}: {len(tracks)} targets, Time: {(t1-t0)*1000:.1f}ms")
        frame_id += 1

    cap.release()
    out.release()
    print("Done.")

if __name__ == "__main__":
    main()