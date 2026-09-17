"""Split image-mode tracks JSONs into per-annotated-frame mask JSONs + PNG overlays.

For every `<tracks-dir>/<stem>.json` (from `annotate.py --frames-csv <index>
--extra-frames-only`) and the `status == ok` rows of `<dataset>/annotated_frames_index.csv`
for that video, writes one file per distinct annotated frame, named after its JPEG:

    <dataset>/seg_masks/<stem>/<jpg stem>.json   COCO uncompressed RLE (tracks schema objects)
    <dataset>/overlay/<stem>/<jpg stem>.png      masks/boxes/ids drawn on frames/<stem>/<jpg>

    python tools/export_frame_masks.py --dataset-dir $SCRATCH/single-animal-distance/pss_p1 \
        --tracks-dir $SCRATCH/single-animal-distance/pss_p1/sam3_image_animal [--only STEM_SUBSTR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from overlay import _draw_objects  # noqa: E402
from tracks import read_tracks  # noqa: E402


def export_video(doc: dict, stem: str, rows: pd.DataFrame, ds: Path, alpha: float) -> tuple[int, int]:
    import cv2

    by_frame = {int(f["frame"]): f["objects"] for f in doc["frames"]}
    mask_dir, ov_dir = ds / "seg_masks" / stem, ds / "overlay" / stem
    mask_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)
    n, n_missing = 0, 0
    for image_rel, frame_idx in rows[["image_path", "frame_idx"]].drop_duplicates().itertuples(index=False):
        frame_idx = int(float(frame_idx))
        if frame_idx not in by_frame:
            print(f"  {stem}: frame {frame_idx} missing from tracks JSON", flush=True)
            n_missing += 1
            continue
        objects = by_frame[frame_idx]
        name = Path(image_rel).stem
        (mask_dir / f"{name}.json").write_text(json.dumps({
            "video": doc["video"], "frame_idx": frame_idx, "image": image_rel,
            "prompt": doc["prompt"], "checkpoint": doc["checkpoint"], "mode": doc.get("mode"),
            "width": doc["width"], "height": doc["height"], "objects": objects,
        }))
        img = cv2.imread(str(ds / image_rel))
        if img is None:
            raise RuntimeError(f"cannot read {ds / image_rel}")
        _draw_objects(img, objects, alpha)
        cv2.imwrite(str(ov_dir / f"{name}.png"), img)
        n += 1
    return n, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--tracks-dir", required=True, type=Path)
    ap.add_argument("--only", help="comma-separated stem substrings")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    ds = args.dataset_dir.resolve()
    idx = pd.read_csv(ds / "annotated_frames_index.csv", dtype=str, keep_default_na=False)
    idx = idx[(idx["status"] == "ok") & (idx["image_path"] != "")]
    idx["stem"] = idx["video_file"].map(lambda f: Path(f).stem)

    toks = [t.strip() for t in args.only.split(",")] if args.only else None
    total_missing = 0
    for json_path in sorted(args.tracks_dir.glob("*.json")):
        stem = json_path.stem
        if toks and not any(t in stem for t in toks):
            continue
        rows = idx[idx["stem"] == stem]
        if rows.empty:
            print(f"{stem}: no ok index rows, skipped", flush=True)
            continue
        n, n_missing = export_video(read_tracks(json_path), stem, rows, ds, args.alpha)
        total_missing += n_missing
        print(f"{stem}: wrote {n} mask JSONs + overlays ({n_missing} missing)", flush=True)
    sys.exit(1 if total_missing else 0)


if __name__ == "__main__":
    main()
