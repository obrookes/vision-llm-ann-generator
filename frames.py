"""Pure-cv2/PIL frame decode + fps subsampling helper.

No torch/sam3 import here (only cv2, PIL, math) so this module runs on a CPU-only
login node with no torch installed, and is importable by sam3_runner.py (video mode),
annotate.py (frame counting) and tests alike.
"""
from __future__ import annotations

import math


def decode_frames(video_path: str, sample_fps: float | None, max_frames: int | None = None):
    """Decode a video, keeping only the frames a sample_fps subsampling would keep.

    Source frame i is kept when floor(i * sample_fps / src_fps) is strictly greater than
    the bucket value of the last kept frame (frame 0 is always kept, since its bucket is
    0 > -1). If sample_fps is None/0, or >= src_fps, every frame is kept (stride 1).

    Skipped frames are only cap.grab()'d (cheap, no decode); kept frames are cap.retrieve()'d
    and converted BGR -> RGB -> PIL.Image. Stops early once max_frames frames have been kept.

    Returns (frame_indices, frames, src_fps):
        frame_indices: list[int], the source-video indices of the kept frames
        frames: list[PIL.Image.Image], RGB
        src_fps: float, the source video's reported fps (falls back to 24.0 if cv2 reports 0)
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    every_frame = not sample_fps or sample_fps >= src_fps

    frame_indices: list[int] = []
    frames: list = []
    last_bucket = -1
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break

            if every_frame:
                keep = True
                bucket = i
            else:
                bucket = int(math.floor(i * sample_fps / src_fps))
                keep = bucket > last_bucket

            if keep:
                ok2, frame_bgr = cap.retrieve()
                if not ok2:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_indices.append(i)
                last_bucket = bucket
                if max_frames is not None and len(frame_indices) >= max_frames:
                    break

            i += 1
    finally:
        cap.release()

    return frame_indices, frames, src_fps
