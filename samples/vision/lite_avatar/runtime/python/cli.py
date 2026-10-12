# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command line surface for LiteAvatar: selection, options and video delivery.

Option declarations and validation, the published S100 asset identities and
selection (including explicitly identified local rebuilds), the model-free
listing/dry-run rendering, and the two video writers — the cross-platform
software FFmpeg pipe and the S100 hardware H.264 worker ring with its
ACK/timeout/process-group discipline.  Nothing in this module loads a board
SDK or imports NumPy at module level, so ``--help``, ``--list-models`` and
``--dry-run`` stay usable on a workstation.  The stage/pipeline composition
lives in ``avatar.py``; ``main.py`` keeps the visible
parse -> select -> construct -> predict -> deliver flow.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import queue
import re
import shutil
import signal as process_signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from utils.py_utils.assets import Asset, list_assets

# ======================================================================
# Published S100 asset identities, digests and selection.
# ======================================================================

SAMPLE_DIR = Path(__file__).resolve().parents[2]

STAGES = ("audio_encoder", "audio2mouth", "facegen_encoder", "facegen_decoder")

FILENAMES = {
    "audio_encoder": "s100/lite_avatar_encoder_400x560_s100.hbm",
    "audio2mouth": "s100/lite_avatar_audio2mouth_30x50x512_s100.hbm",
    "facegen_encoder": "s100/lite_avatar_net_encode_384x384_s100.hbm",
    "facegen_decoder": "s100/lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm",
}

#: Committed CMVN file pinned by digest (same bytes as the Paraformer sample).
LOCAL_DIGESTS = {
    "am.mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
}

#: Video encoders this sample can drive.  ``hardware`` is verified on S100.
ENCODERS = ("auto", "software", "hardware")


@dataclass(frozen=True)
class Selection:
    target: str
    stage: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False
    #: ``"manifest"`` keeps the publisher digest; ``"local"`` marks the digest
    #: as explicitly identified by a user-provided rebuild manifest.
    digest_source: str = "manifest"


def _validated_digest(filename, digest) -> str:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise ValueError(f"Local rebuild digest for {filename} is not SHA-256")
    return digest.lower()


def load_local_hash_manifest(path) -> dict[str, str]:
    """Read a user-provided ``{filename: sha256}`` rebuild manifest.

    Explicitly identified local rebuilds are accepted through this manifest:
    each listed file is verified against the recorded digest, and the SDK
    shape/target binding still validates the artifact structure.  Filenames
    must come from the published set (a partial list is fine; files that are
    not listed keep their published digests); unknown entries are rejected.
    """

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, dict) or not records:
        raise ValueError("Local hash manifest must be a nonempty JSON object")
    known = set(FILENAMES.values())
    hashes = {}
    for filename, digest in records.items():
        if filename not in known:
            raise ValueError(
                f"Local hash manifest names unknown asset {filename!r}; "
                f"expected entries from {sorted(known)}"
            )
        hashes[filename] = _validated_digest(filename, digest)
    return hashes


def resolve_selections_for(args):
    """Resolve the four stage selections for the parsed arguments.

    The model-free listing mode defaults an ``auto`` target to the published
    S100 set; host dry-run requires the explicit target because no board
    detection happens there.
    """

    target = args.target
    if target == "auto" and args.list_models:
        target = "s100"
    if target == "auto" and args.dry_run:
        raise ValueError("Host dry-run requires --target s100")
    return resolve_selections(
        target,
        model_dir=args.model_dir,
        local_hashes=(
            load_local_hash_manifest(args.models_manifest)
            if args.models_manifest
            else None
        ),
    )


def resolve_selections(
    target="auto",
    *,
    model_dir=None,
    local_hashes=None,
):
    """Select the four published models for one target.

    Args:
        target: ``auto`` (hardware detection) or an explicit board target.
        model_dir: Optional directory holding the manifest-relative paths
            (default: the sample's ``model/`` directory).
        local_hashes: Optional ``{filename: sha256}`` mapping from
            :func:`load_local_hash_manifest` (validated here as well, so
            direct API callers get the same checks).  A partial mapping is
            allowed: every listed filename switches to its recorded digest
            and is marked ``digest_source="local"``; unlisted files keep
            their published digests.

    Returns:
        tuple: Four :class:`Selection` records in stage order.

    Raises:
        ValueError: The target has no published set, or a manifest row is
            missing/unusable.  (Manual availability: no download URLs.)
    """

    if target == "auto":
        from utils.py_utils.platforms import detect_target

        target = detect_target()
    if target != "s100":
        raise ValueError("LiteAvatar publishes the four-model set only for s100")
    if local_hashes is not None:
        unknown = set(local_hashes) - set(FILENAMES.values())
        if unknown:
            raise ValueError(
                f"Local hashes reference files outside the published set: {sorted(unknown)}"
            )
        for filename, digest in local_hashes.items():
            _validated_digest(filename, digest)
    assets = {asset.filename: asset for asset in list_assets("s", "lite_avatar")}
    root = Path(model_dir).expanduser() if model_dir else SAMPLE_DIR / "model"
    selected = []
    for stage in STAGES:
        asset = assets.get(FILENAMES[stage])
        if asset is None or asset.format != "hbm":
            raise ValueError(f"Missing published {stage} HBM asset")
        local_digest = (local_hashes or {}).get(asset.filename)
        if asset.sha256 is None and local_digest is None:
            raise ValueError(
                f"Published {stage} asset has no digest; pass --models-manifest "
                "for an explicitly identified local rebuild"
            )
        if local_digest is not None and local_digest != asset.sha256:
            asset = replace(asset, sha256=local_digest)
        selected.append(
            Selection(
                target,
                stage,
                asset,
                root / asset.filename,
                model_dir is not None,
                "local" if local_digest is not None else "manifest",
            )
        )
    return tuple(selected)


def missing_models(selections) -> tuple[Path, ...]:
    """Return the model files that still need local preparation."""

    return tuple(
        selection.model_path
        for selection in selections
        if not selection.model_path.is_file()
    )


def resolve_encoder(value, target):
    """Map the encoder choice to a concrete writer for one target.

    ``auto`` resolves to the hardware H.264 encoder on S100 (the only board
    where it is verified) and to software elsewhere.  An explicit choice is
    never silently changed: ``hardware`` on an unverified target is an error.
    """

    if value not in ENCODERS:
        raise ValueError(f"--encoder must be one of {ENCODERS}")
    if value == "auto":
        return "hardware" if target == "s100" else "software"
    if value == "hardware" and target != "s100":
        raise ValueError(
            "The hardware H.264 encoder is verified on S100 only; use "
            "--encoder software on other targets"
        )
    return value


# ======================================================================
# Argument parsing and validation.
# ======================================================================


def build_parser():
    """Build the SDK-free parser for the LiteAvatar entry."""

    parser = argparse.ArgumentParser(
        description="LiteAvatar: render an audio-driven talking-avatar video "
        "from a whole audio clip (offline, no live capture)."
    )
    parser.add_argument(
        "--target",
        choices=("auto", "x5", "s100", "s100p", "s600"),
        default="auto",
        help="Execution target (auto, x5, s100, s100p or s600).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-models",
        action="store_true",
        help="List manifest-backed asset references without board access.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve/check the four-model selection without loading an SDK.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Input WAV (any sample rate; first channel of stereo is used).",
    )
    parser.add_argument(
        "--avatar-data",
        type=Path,
        default=SAMPLE_DIR / "test_data" / "preload",
        help="Avatar data directory (bg_video.mp4, face_box.txt, ref_frames/).",
    )
    parser.add_argument(
        "--cmvn-path",
        type=Path,
        default=SAMPLE_DIR / "model" / "am.mvn",
        help="CMVN file for the audio frontend (pinned am.mvn).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lite_avatar/avatar.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--encoder",
        choices=ENCODERS,
        default="auto",
        help="Video encoder: auto (hardware on S100, software elsewhere), "
        "software (FFmpeg pipe) or hardware (S100 H.264 worker).",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Skip video writing entirely (renderer-only run).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory holding the manifest-relative model paths.",
    )
    parser.add_argument(
        "--models-manifest",
        type=Path,
        help="JSON {filename: sha256} manifest identifying local rebuilds.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        help="Runtime scheduling priority (0-255).",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        help="Runtime BPU core indexes.",
    )
    return parser


def normalize_args(args) -> None:
    """Validate flag combinations and expand path arguments in place."""

    if args.priority is not None and not 0 <= args.priority <= 255:
        raise ValueError("priority must be in [0,255]")
    if args.bpu_cores is not None and (
        not args.bpu_cores or any(core < 0 for core in args.bpu_cores)
    ):
        raise ValueError("bpu-cores must be nonnegative")
    if args.no_encode and args.encoder != "auto":
        raise ValueError("--no-encode skips video writing; do not pass --encoder")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.list_models or args.dry_run:
        return
    if args.audio is None:
        raise ValueError("--audio is required for a real run")
    if not args.audio.is_file():
        raise ValueError(f"Missing audio file: {args.audio}")
    if args.encoder == "hardware" and args.target not in ("auto", "s100"):
        raise ValueError(
            "The hardware H.264 encoder is verified on S100 only; use "
            "--encoder software on other targets"
        )


def read_audio(path):
    """Return (float32 waveform, sample rate) exactly as the source loaded."""

    import soundfile as sf

    data, sample_rate = sf.read(str(path))
    import numpy as np

    return np.asarray(data, dtype=np.float32), int(sample_rate)


# ======================================================================
# Model-free listing and dry-run rendering.
# ======================================================================


def print_model_list(target) -> int:
    """Print the manifest-backed references for ``target`` (SDK-free)."""

    try:
        selections = resolve_selections(target)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("Manifest-backed LiteAvatar asset references:")
    for selection in selections:
        asset = selection.asset
        digest = asset.sha256 or "<no digest>"
        state = "present" if selection.model_path.is_file() else "missing (prepare locally)"
        print(f"  asset_id: {asset.reference}")
        print(f"    stage: {selection.stage}")
        print(f"    filename: {asset.filename}")
        print(f"    format: {asset.format}")
        print(f"    sha256: {digest}")
        print(f"    local: {selection.model_path} [{state}]")
    print("Availability is manual: prepare the artifacts per model/README.md.")
    return 0


def print_resolution(args, selections) -> int:
    """Print the concrete four-model selection without loading an SDK."""

    print("Selection:")
    for selection in selections:
        print(f"  stage: {selection.stage}")
        print(f"    target: {selection.target}")
        print(f"    asset_id: {selection.asset.reference}")
        print(f"    model_path: {selection.model_path}")
        print(f"    model_path_exists: {selection.model_path.is_file()}")
    missing = missing_models(selections)
    if missing:
        print("Missing artifacts (prepare per model/README.md):")
        for path in missing:
            print(f"  {path}")
    print("No model is downloaded and no SDK is loaded.")
    return 0 if not missing else 1


# ======================================================================
# Video writers.  The software pipe is the cross-platform default; the
# hardware writer keeps the validated ring/ACK worker protocol.
# ======================================================================


class SoftwareVideoWriter:
    """Bounded raw-frame pipe; FFmpeg (libx264) encodes concurrently.

    The encoder runs in its own session so the whole process group can be
    killed on abort: killing only the FFmpeg parent would leave any child
    holding the stdin pipe open, and the writer thread would block on it
    past its join timeout.
    """

    #: Hard ceiling for one frame waiting on a stalled encoder (seconds).
    STALL_TIMEOUT = 120

    def __init__(self, path, audio, width, height):
        import queue as _queue

        self.log = open(f"{path}.ffmpeg.log", "wb")
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
            "-framerate", "30", "-i", "pipe:0", "-i", str(audio),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-tune", "zerolatency", "-crf", "18", "-pix_fmt", "yuv420p",
            "-threads", "2", "-c:a", "aac", "-shortest", str(path),
        ]
        # POSIX only: the boards, the mac host and Linux CI all are.
        popen_kwargs = (
            {"start_new_session": True} if os.name == "posix" else {}
        )
        try:
            self.proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=self.log,
                **popen_kwargs,
            )
        except BaseException:
            self.log.close()
            Path(f"{path}.ffmpeg.log").unlink(missing_ok=True)
            raise
        self.frames = _queue.Queue(maxsize=2)
        self.error = None
        self._done = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _kill(self):
        """Kill the encoder's whole process group, then reap the parent."""

        if os.name == "posix":
            try:
                os.killpg(self.proc.pid, process_signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        else:  # pragma: no cover - non-POSIX hosts are not delivery targets
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass

    def _close_stdin(self):
        """Close the parent's pipe write-end; a no-op after a clean drain."""

        stream = self.proc.stdin
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except (BrokenPipeError, OSError):
                pass

    def _write(self):
        try:
            while True:
                frame = self.frames.get()
                if frame is None:
                    break
                self.proc.stdin.write(memoryview(frame))
            self.proc.stdin.close()
        except Exception as error:
            self.error = error

    def submit(self, frame):
        deadline = time.perf_counter() + self.STALL_TIMEOUT
        while True:
            if self.error is not None:
                raise RuntimeError("Video encoder failed") from self.error
            if self.proc.poll() is not None:
                # Give the writer thread a moment to record its error, then
                # surface the early exit instead of blocking on a full queue.
                self.thread.join(timeout=1.0)
                if self.error is not None:
                    raise RuntimeError("Video encoder failed") from self.error
                raise RuntimeError(
                    f"Video encoder exited early with code {self.proc.returncode}"
                )
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"Video encoder stalled for more than {self.STALL_TIMEOUT} s"
                )
            try:
                self.frames.put(frame, timeout=min(0.1, remaining))
                return
            except queue.Full:
                continue

    def close(self):
        if self._done:
            return
        self._done = True
        try:
            self.submit(None)
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                self._kill()
                raise RuntimeError("Video encoder did not drain within 30 seconds")
            code = self.proc.wait(timeout=30)
            if self.error is not None or code:
                raise RuntimeError(
                    f"Video encoder failed (exit {code})"
                ) from self.error
        finally:
            if self.thread.is_alive():
                self._kill()
            self._close_stdin()
            self.log.close()

    def abort(self):
        if self._done and not self.thread.is_alive():
            return
        self._done = True
        self._kill()
        try:
            self.frames.put_nowait(None)  # unblock the writer thread
        except queue.Full:
            pass
        self.thread.join(timeout=5)
        self._close_stdin()
        if not self.log.closed:
            self.log.close()


class HardwareVideoWriter:
    """S100 hardware H.264 ring: a worker process owns the encoder.

    A bounded shared-memory ring separates encoding from the renderer's
    GIL.  The worker acknowledges each consumed slot (``ACK <index>``); a
    slot is reused only after its ACK, never overwritten speculatively.
    """

    SLOTS = 4
    INIT_TIMEOUT = 15
    ACK_TIMEOUT = 10
    CLOSE_TIMEOUT = 40

    def __init__(self, path, audio, width, height, *, worker_command=None, ring_dir="/dev/shm"):
        import numpy as np

        self.encoded_at = None
        self._done = False
        self.file = tempfile.NamedTemporaryFile(
            prefix="lite-avatar-", suffix=".ring", dir=ring_dir, delete=False
        )
        self.file.truncate(self.SLOTS * width * height * 3)
        self.mapping = mmap.mmap(self.file.fileno(), 0)
        self.ring = np.ndarray(
            (self.SLOTS, height, width, 3), np.uint8, buffer=self.mapping
        )
        self.free = queue.Queue(maxsize=self.SLOTS)
        for index in range(self.SLOTS):
            self.free.put(index)
        self.ready = threading.Event()
        self.started = False
        self.done = False
        self.log = open(f"{path}.encoder.log", "w")
        try:
            self.proc = subprocess.Popen(
                worker_command
                or [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "encoder_worker.py"),
                    "--ring", self.file.name,
                    "--width", str(width),
                    "--height", str(height),
                    "--slots", str(self.SLOTS),
                    "--output", str(path),
                    "--audio", str(audio),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except BaseException:
            self._release()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        if (
            not self.ready.wait(timeout=self.INIT_TIMEOUT)
            or not self.started
            or self.proc.poll() is not None
        ):
            self.abort()
            raise RuntimeError("Hardware encoder worker did not initialize")

    def _read(self):
        for line in self.proc.stdout:
            # The native codec writes ANSI reset codes without a newline.
            line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
            if line.strip() == "READY":
                self.started = True
                self.ready.set()
            elif line.startswith("ACK "):
                self.free.put(int(line[4:]))
            elif line.strip() == "DONE":
                self.done = True
            elif line.startswith("ENCODED "):
                self.encoded_at = float(line[8:])
            else:
                self.log.write(line)
                self.log.flush()
        self.ready.set()

    def _acquire(self):
        deadline = time.perf_counter() + self.ACK_TIMEOUT
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Hardware encoder worker exited: {self.proc.returncode}"
                )
            try:
                return self.free.get(timeout=0.1)
            except queue.Empty:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        "Hardware encoder did not acknowledge a frame within "
                        f"{self.ACK_TIMEOUT} seconds"
                    )

    def _send(self, index):
        try:
            self.proc.stdin.write(f"{index}\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            code = self.proc.poll()
            raise RuntimeError(
                f"Hardware encoder worker exited before consuming slot {index} "
                f"(exit {code})"
            ) from error

    def submit(self, frame):
        import numpy as np

        index = self._acquire()
        np.copyto(self.ring[index], frame)
        self._send(index)

    def render(self, result, index):
        """Render presentation frame ``index`` directly into a ring slot."""

        slot = self._acquire()
        result.render_frame(index, out=self.ring[slot])
        self._send(slot)

    def _release(self):
        """Drop the ring mapping/file and close pipes and the log exactly once."""

        self.ring = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if not self.file.closed:
            self.file.close()
        proc = getattr(self, "proc", None)  # Popen may have failed to start
        if proc is not None:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except (BrokenPipeError, OSError):
                        pass
        try:
            os.unlink(self.file.name)
        except FileNotFoundError:
            pass
        if not self.log.closed:
            self.log.close()

    # Kept under the historical name for callers/tests already using it.
    def _cleanup(self):
        self._release()

    def close(self):
        if self._done:
            return
        try:
            try:
                self.proc.stdin.write("END\n")
                self.proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass  # worker already gone; wait() reports the real exit below
            try:
                # Allow the worker's 30-second remux timeout plus queue drain.
                code = self.proc.wait(timeout=self.CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "Hardware encoder worker did not finish within "
                    f"{self.CLOSE_TIMEOUT} s"
                ) from error
            self.reader.join(timeout=2)
            self._release()
            self._done = True
            if code or not self.done:
                raise RuntimeError(
                    f"Hardware encoder worker failed: exit {code}, "
                    f"finished={self.done}"
                )
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if self._done:
            return
        self._done = True
        try:
            os.killpg(self.proc.pid, process_signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.reader.join(timeout=2)
        self._release()


def make_writer(kind, path, audio, width, height):
    """Construct the requested writer; FFmpeg presence is checked first."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for video output (not on PATH)")
    if kind == "hardware":
        return HardwareVideoWriter(path, audio, width, height)
    return SoftwareVideoWriter(path, audio, width, height)


def deliver_frames(result, writer) -> float:
    """Render every frame and drain the writer; abort on any failure.

    Hardware-writer runs render straight into the next free ring slot;
    software-writer runs render an owned frame then submit it.  Any
    exception aborts the writer (bounded, idempotent cleanup) before
    propagating, so the original error is never masked by cleanup noise.
    Returns the delivery wall time in seconds.
    """

    start = time.perf_counter()
    try:
        for index in range(result.frame_count):
            if writer is None:
                result.render_frame(index)
            elif isinstance(writer, HardwareVideoWriter):
                writer.render(result, index)
            else:
                writer.submit(result.render_frame(index))
        if writer is not None:
            writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    return time.perf_counter() - start


def print_result(result, output_path, encoder_kind, seconds, encoded, timings=None) -> None:
    """Print the delivery summary for one finished render."""

    print(
        f"Rendered {result.frame_count} frames "
        f"({result.width}x{result.height} @ 30 fps) in {seconds:.2f}s "
        f"({result.frame_count / seconds:.2f} fps)"
    )
    if encoded:
        print(f"Video written: {output_path} (encoder: {encoder_kind})")
    else:
        print("Video writing skipped (--no-encode)")
    audio_seconds = (timings or {}).get("audio")
    renderer_seconds = (timings or {}).get("renderer_init")
    if audio_seconds is not None and renderer_seconds is not None:
        print(
            f"Offline phases: audio features+models {audio_seconds:.2f}s, "
            f"renderer init {renderer_seconds:.2f}s (excluded from the "
            "render rate above)"
        )


__all__ = [
    "ENCODERS",
    "FILENAMES",
    "HardwareVideoWriter",
    "LOCAL_DIGESTS",
    "STAGES",
    "SAMPLE_DIR",
    "Selection",
    "SoftwareVideoWriter",
    "build_parser",
    "deliver_frames",
    "load_local_hash_manifest",
    "make_writer",
    "missing_models",
    "normalize_args",
    "print_model_list",
    "print_resolution",
    "print_result",
    "read_audio",
    "resolve_encoder",
    "resolve_selections",
]
