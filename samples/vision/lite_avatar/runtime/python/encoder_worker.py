# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""S100 hardware encoder worker. Frame ownership is released only after ACK.

Runs as a subprocess of the sample's :class:`HardwareVideoWriter`: frames
arrive as indexes into a shared-memory ring; each consumed slot is
acknowledged with ``ACK <index>`` so the renderer never overwrites a frame
the encoder still owns.  The worker owns the ``libsrcampy`` encoder for its
whole lifetime, streams elementary H.264 to ``<output>.h264`` and remuxes
with FFmpeg after ``END``.

Color-space contract (matches the validated source pipeline): OpenCV
BGR->I420 conversion produces limited-range BT.601; the remux therefore
signals ``video_full_range_flag=0:matrix_coefficients=5`` and crops the
replicated pad columns used to reach the encoder's 8-pixel width multiple.
"""

import argparse
import mmap
import subprocess
import sys
import time

import numpy as np
import cv2
from hobot_vio import libsrcampy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    width = (args.width + 7) // 8 * 8
    pad = width - args.width
    pixels = width * args.height
    nv12 = np.empty(pixels * 3 // 2, np.uint8)
    encoder = libsrcampy.Encoder()
    code = encoder.encode(0, 1, width, args.height, 16000)
    if code:
        encoder.close()
        raise RuntimeError(f"S100 encoder initialization failed: {code}")
    with open(args.ring, "r+b") as source:
        mapping = mmap.mmap(source.fileno(), 0)
        ring = np.ndarray(
            (args.slots, args.height, args.width, 3), dtype=np.uint8, buffer=mapping
        )
        try:
            with open(args.output + ".h264", "wb") as output:
                print("READY", flush=True)
                for line in sys.stdin:
                    if line.strip() == "END":
                        break
                    index = int(line)
                    if not 0 <= index < args.slots:
                        raise ValueError(f"Invalid frame slot: {index}")
                    frame = ring[index]
                    if pad:
                        frame = cv2.copyMakeBorder(
                            frame, 0, 0, 0, pad, cv2.BORDER_REPLICATE
                        )
                    planar = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).reshape(-1)
                    nv12[:pixels] = planar[:pixels]
                    nv12[pixels::2] = planar[pixels : pixels + pixels // 4]
                    nv12[pixels + 1 :: 2] = planar[pixels + pixels // 4 :]
                    code = encoder.encode_file(nv12.tobytes())
                    packet = encoder.get_img()
                    if code or packet is None:
                        raise RuntimeError(
                            f"S100 encoder did not return a frame: {code}"
                        )
                    output.write(packet)
                    print(f"ACK {index}", flush=True)
        finally:
            encoder.close()
            if "frame" in locals():
                del frame
            del ring
            mapping.close()
    print(f"ENCODED {time.perf_counter()}", flush=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-fflags", "+genpts",
        "-r", "30", "-i", args.output + ".h264",
        "-stream_loop", str(args.repeat - 1),
        "-i", args.audio, "-c:v", "copy",
    ]
    # OpenCV BGR->I420 produces limited-range BT.601, matching the source video.
    metadata = "video_full_range_flag=0:matrix_coefficients=5"
    if pad:
        metadata += f":crop_right={pad}"
    command += ["-bsf:v", "h264_metadata=" + metadata]
    command += ["-c:a", "aac", "-shortest", "-movflags", "+faststart", args.output]
    subprocess.run(command, check=True, timeout=30)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
