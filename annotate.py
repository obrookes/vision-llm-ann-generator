"""SAM3 text-prompted annotation over a video directory/manifest.

    python annotate.py --manifest manifests/dev32.txt --prompt animal --out outputs/dev32_animal \
        [--checkpoint $SCRATCH/weights/sam3/sam3-safari-pos.pt] [--score-thresh 0.5] \
        [--mode video|image] [--sample-fps 6] [--overlay-all-frames] \
        [--max-frames N] [--no-overlay] [--overwrite] [--video-dir DIR instead of --manifest]

Per video: run sam3_runner.Sam3Runner.track once (predictor loaded once per process, one
session per video), write a tracks JSON (docs/OUTPUT_SCHEMA.md), render an overlay video
unless --no-overlay, and append one status line to <out>/index.jsonl. Resumable: videos
whose <out>/<stem>.json already exists are skipped unless --overwrite. Per-video errors are
caught, logged to index.jsonl with status "error", and do not stop the run (mirrors
verify_vllm.py's decode-failure convention).

By default only every ~4th-5th source frame is decoded and run through the model
(--sample-fps 6, matching SAM3's training/eval fps; 0 = every source frame), and the
overlay video is rendered from just those sampled frames (--overlay-all-frames renders
every source frame, holding the most recent annotation in between).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from manifest import read_manifest, from_video_dir
from tracks import stem_for, write_tracks, summarise


def process_video(runner, video_path: str, prompt: str, checkpoint: str, out_dir: Path,
                   stem: str, score_thresh, max_frames, no_overlay: bool,
                   overlay_all_frames: bool) -> dict:
    t0 = time.perf_counter()
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_results, track_meta = runner.track(video_path, prompt, max_frames=max_frames)

    from tracks import rle_encode

    frames = []
    for fr in frame_results:
        objects = []
        ids = fr["ids"]
        probs = fr["probs"]
        boxes_norm = fr["boxes_xywh_norm"]
        masks = fr["masks"]
        for i in range(len(ids)):
            x, y, w, h = boxes_norm[i]
            box_abs = [x * width, y * height, w * width, h * height]
            obj = {
                "id": ids[i],
                "score": probs[i] if i < len(probs) else None,
                "box_xywh": box_abs,
            }
            if i < len(masks):
                obj["mask_rle"] = rle_encode(masks[i])
            objects.append(obj)
        frames.append({"frame": fr["frame"], "objects": objects})

    doc = {
        "schema_version": 1,
        "video": str(video_path),
        "prompt": prompt,
        "checkpoint": os.path.basename(checkpoint),
        "mode": track_meta["mode"],
        "sample_fps": track_meta["sample_fps"],
        "n_frames_sampled": track_meta["n_frames_sampled"],
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": n_frames_src,
        "frames": frames,
    }

    json_path = out_dir / f"{stem}.json"
    write_tracks(json_path, doc)
    # fps_processed is measured over tracking + json write, i.e. before the overlay render below
    # (overlay rendering is a display/QA convenience, not part of "processing" a video).
    t_tracked = time.perf_counter()

    if not no_overlay:
        from overlay import render

        render(video_path, doc, out_dir / f"{stem}.mp4", sampled_only=not overlay_all_frames)

    stats = summarise(doc)
    stats["mode"] = track_meta["mode"]
    stats["sample_fps"] = track_meta["sample_fps"]
    stats["n_frames_sampled"] = track_meta["n_frames_sampled"]
    # fps_processed: sampled frames / (tracking + json write) seconds, i.e. excluding overlay
    # render time (see the t_tracked comment above)
    tracking_seconds = t_tracked - t0
    stats["fps_processed"] = track_meta["n_frames_sampled"] / tracking_seconds if tracking_seconds > 0 else 0.0
    stats["seconds"] = time.perf_counter() - t0
    return stats


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="text file, one video path per line")
    src.add_argument("--video-dir", help="walk this dir recursively for video files")
    ap.add_argument("--ext", default=".MP4,.mp4,.avi", help="comma-separated extensions (with --video-dir)")
    ap.add_argument("--prompt", required=True, help="text prompt, e.g. 'animal'")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="default: $SCRATCH/weights/sam3/sam3-safari-pos.pt",
    )
    ap.add_argument("--score-thresh", type=float, default=None)
    ap.add_argument("--mode", choices=["video", "image"], default="video",
                     help="video: tracked propagation (persistent ids); "
                          "image: per-frame detection only, no tracking")
    ap.add_argument("--sample-fps", type=float, default=6.0,
                     help="subsample source video to this fps before inference "
                          "(SAM3's training/eval fps; 0 = every source frame)")
    ap.add_argument("--overlay-all-frames", action="store_true",
                     help="render every source frame in the overlay (default: only the "
                          "sampled frames the model actually saw)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--root", default=None, help="corpus root for stem flattening (see tracks.stem_for)")
    ap.add_argument("--index-name", default="index.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            ap.error("--checkpoint not given and $SCRATCH is not set")
        checkpoint = str(Path(scratch) / "weights" / "sam3" / "sam3-safari-pos.pt")

    if args.manifest:
        videos = read_manifest(args.manifest)
    else:
        exts = [e.strip() for e in args.ext.split(",") if e.strip()]
        videos = from_video_dir(args.video_dir, exts)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / args.index_name

    already = set()
    if index_path.exists():
        with index_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already.add(rec["stem"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = []
    for v in videos:
        stem = stem_for(v, root=args.root)
        json_path = out_dir / f"{stem}.json"
        if not args.overwrite and json_path.exists():
            continue
        todo.append((v, stem))

    print(f"{len(videos)} videos listed, {len(videos) - len(todo)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    # Heavy import after arg parsing so --help is fast.
    from sam3_runner import Sam3Runner

    runner = Sam3Runner(
        checkpoint=checkpoint, device=args.device, score_thresh=args.score_thresh,
        mode=args.mode, sample_fps=args.sample_fps,
    )

    t_start = time.perf_counter()
    n_done = 0
    with index_path.open("a") as fidx:
        for video_path, stem in todo:
            rec = {"video": video_path, "stem": stem, "prompt": args.prompt, "checkpoint": os.path.basename(checkpoint)}
            try:
                stats = process_video(
                    runner, video_path, args.prompt, checkpoint, out_dir, stem,
                    args.score_thresh, args.max_frames, args.no_overlay,
                    args.overlay_all_frames,
                )
                rec.update(status="ok", **stats)
            except Exception as e:  # per-video failure: log, keep going
                import traceback
                traceback.print_exc()
                rec.update(status="error", error=repr(e))
            fidx.write(json.dumps(rec) + "\n")
            fidx.flush()
            n_done += 1
            elapsed = time.perf_counter() - t_start
            rate = n_done / elapsed * 60 if elapsed > 0 else 0.0
            print(f"[{n_done}/{len(todo)}] {stem} status={rec['status']} ({rate:.1f} videos/min)", flush=True)

    print(f"done: {n_done} videos in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
