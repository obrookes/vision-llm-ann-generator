"""Render a tracks JSON doc over its source video: mask blend, box, "id N" label.

Mirrors the rendering style of wcf-annotation-pipeline/pipeline/io.py::write_overlay
(cv2, 0.5 alpha mask blend, 2px box, id label) but colours by a fixed 20-colour
palette indexed by `id % 20` (schema-driven; no per-flag colouring here).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tracks import read_tracks, rle_decode

# 20-colour palette (BGR, for cv2), reasonably distinct / high-contrast.
_PALETTE = [
    (66, 133, 244), (52, 168, 83), (234, 67, 53), (251, 188, 5),
    (154, 92, 232), (0, 172, 193), (255, 112, 67), (156, 204, 101),
    (240, 98, 146), (3, 169, 244), (139, 195, 74), (255, 160, 0),
    (171, 71, 188), (0, 150, 136), (233, 30, 99), (121, 85, 72),
    (96, 125, 139), (255, 87, 34), (76, 175, 80), (63, 81, 181),
]


def _colour_for(obj_id: int) -> tuple[int, int, int]:
    return _PALETTE[obj_id % len(_PALETTE)]


def render(video_path: str, doc: dict, out_path: str, alpha: float = 0.5) -> Path:
    import cv2

    by_frame = {fr["frame"]: fr.get("objects", []) for fr in doc.get("frames", [])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for obj in by_frame.get(idx, []):
                color = _colour_for(int(obj["id"]))
                mask_rle = obj.get("mask_rle")
                if mask_rle:
                    m = rle_decode(mask_rle)
                    frame[m] = (alpha * frame[m] + (1 - alpha) * np.array(color)).astype(np.uint8)
                box = obj.get("box_xywh")
                if box:
                    x, y, bw, bh = box
                    x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"id {obj['id']}"
                    label_pos = (x1, max(12, y1 - 6))
                    # dark outline for legibility, then the coloured fill on top
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv2.LINE_AA)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render an overlay video from an existing tracks JSON")
    ap.add_argument("tracks_json")
    ap.add_argument("--video", help="source video path; default: the 'video' field in the tracks JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    doc = read_tracks(args.tracks_json)
    video_path = args.video or doc["video"]
    out = render(video_path, doc, args.out, alpha=args.alpha)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
