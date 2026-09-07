"""Build manifests: walk a video directory, or list rendered overlays from an output dir.

    python manifest.py --video-dir DIR [--ext .MP4,.mp4,.avi] --write out.txt
    python manifest.py --from-outputs OUTDIR --write out.txt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_manifest(path: str) -> list[str]:
    """One path per line; blank lines and '#' comments are skipped."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _write(paths: list[str], out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in paths:
            f.write(p + "\n")


def from_video_dir(video_dir: str, exts: list[str]) -> list[str]:
    exts_lower = {e.lower() for e in exts}
    paths = [
        str(p.resolve())
        for p in Path(video_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in exts_lower
    ]
    return sorted(paths)


def from_outputs(out_dir: str) -> list[str]:
    paths = [str(p.resolve()) for p in Path(out_dir).glob("*.mp4")]
    return sorted(paths)


def frames_by_video(csv_path: str) -> dict[str, set[int]]:
    """Read an annotations CSV (columns include `video_file`, `frame_idx`) and return
    {basename(video_file): {int(frame_idx), ...}}.

    Uses only the stdlib `csv` module (DictReader). Tolerates a CSV built by concatenating
    several per-video exports that each carried their own header row: any data row whose
    `video_file` value is literally "video_file" (i.e. a repeated header line) is skipped.
    `frame_idx` is read via `int(float(...))` since it may be written as "26.0".
    """
    result: dict[str, set[int]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_file = (row.get("video_file") or "").strip()
            if not video_file or video_file == "video_file":
                continue
            frame_idx = row.get("frame_idx")
            if frame_idx is None or frame_idx.strip() == "":
                continue
            key = Path(video_file).name
            result.setdefault(key, set()).add(int(float(frame_idx)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", help="walk this dir recursively for video files")
    ap.add_argument("--ext", default=".MP4,.mp4,.avi", help="comma-separated extensions (with --video-dir)")
    ap.add_argument("--from-outputs", help="list overlay .mp4 files in this annotate.py output dir")
    ap.add_argument("--write", required=True, help="output manifest path")
    args = ap.parse_args()

    if bool(args.video_dir) == bool(args.from_outputs):
        ap.error("exactly one of --video-dir or --from-outputs is required")

    if args.video_dir:
        exts = [e.strip() for e in args.ext.split(",") if e.strip()]
        paths = from_video_dir(args.video_dir, exts)
    else:
        paths = from_outputs(args.from_outputs)

    _write(paths, args.write)
    print(f"wrote {len(paths)} paths to {args.write}")


if __name__ == "__main__":
    main()
