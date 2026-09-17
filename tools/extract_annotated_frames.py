"""Extract OCR-verified annotated frames for a camtrap-align dataset directory whose
`frame_assignments.csv` already exists, WITHOUT re-running `camtrap-align reconcile`
(whose xlsx/drive-inventory inputs aren't on the cluster).

Reuses camtrap-align's per-video worker unchanged (`pipeline._process_video_task`:
OCR start time -> locate + verify one frame per annotated second -> JPEG), so outputs
match `camtrap-align extract-frames`:

    <dataset>/frames/<stem>/<stem>_tNN_fNNNN.jpg
    <dataset>/annotated_frames_index.csv   (merged per video_key across runs)
    <dataset>/videos.csv
    <dataset>/ocr_cache.json

Local videos are resolved via <dataset>/videos/manifest.csv (video_key -> local_name).

    python tools/extract_annotated_frames.py --dataset-dir $SCRATCH/single-animal-distance/pss_p1 \
        --config ~/camtrap-align/configs/pss_p1.yaml [--only 04010054] [--workers 4] [--force]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from camtrapalign.config import load_config
from camtrapalign.ocr import OverlayConfig
from camtrapalign.pipeline import _make_video_row, _process_video_task
from camtrapalign.store import load_json, save_json
from camtrapalign.extract import write_index, write_videos_table
from camtrapalign.videos import probe_video

# frame_assignments.csv columns that are pipeline-internal, not annotation attributes
def _init_ocr(threads: int) -> None:
    """Pre-build camtrap-align's cached RapidOCR engine with a bounded onnxruntime thread pool.
    The default (-1) sizes each pool to every core on the node, so N workers on a Slurm allocation
    oversubscribe the CPUs by orders of magnitude and OCR crawls."""
    if threads <= 0:
        return
    import camtrapalign.ocr as ocr
    from rapidocr_onnxruntime import RapidOCR

    ocr._ENGINE = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1)


_INTERNAL = {
    "filepath", "file", "video_key", "annot_time", "transect", "cam", "cam_num", "video_id",
    "file_transect", "file_cam", "file_stem", "file_mission", "match_strategy", "duplicate_filepaths",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write frames/, annotated_frames_index.csv, videos.csv, ocr_cache.json "
                         "(default: --dataset-dir, i.e. current in-place behaviour). "
                         "frame_assignments.csv and videos/manifest.csv are always read from --dataset-dir.")
    ap.add_argument("--config", required=True, help="camtrap-align dataset YAML (ocr/output blocks only)")
    ap.add_argument("--only", help="comma-separated substrings matched against video_key / local video name")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run OCR and rewrite JPEGs")
    ap.add_argument("--ocr-threads", type=int, default=2,
                    help="onnxruntime intra-op threads per worker (<=0: library default, all cores)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = args.dataset_dir.resolve()
    out = args.out_dir.resolve() if args.out_dir else ds
    out.mkdir(parents=True, exist_ok=True)

    fa = pd.read_csv(ds / "frame_assignments.csv", dtype=str, keep_default_na=False)
    fa["annot_time"] = pd.to_datetime(fa["annot_time"].replace("", None), errors="coerce")
    fa = fa.rename(columns={"cam": "cam_num"})  # extract_video reads transect/cam_num/video_id
    attr_cols = [c for c in fa.columns if c not in _INTERNAL]

    manifest = pd.read_csv(ds / "videos" / "manifest.csv", dtype=str)
    local_by_key = dict(zip(manifest["video_key"], manifest["local_name"]))

    keys = sorted(fa["video_key"].unique())
    if args.only:
        toks = [t.strip() for t in args.only.split(",") if t.strip()]
        keys = [k for k in keys if any(t in k or t in local_by_key.get(k, "") for t in toks)]
    print(f"{len(keys)} video(s) selected", flush=True)

    # Seed from the dataset dir's cache (read-only) so an --out-dir run reuses OCR results
    # already computed there, but always write the merged cache under --out-dir.
    ocr_cache = load_json(ds / "ocr_cache.json", default={})
    if out != ds:
        ocr_cache = {**ocr_cache, **load_json(out / "ocr_cache.json", default={})}
    overrides = load_json(ds / "overrides.json", default={})
    ocr_cfg = OverlayConfig.from_config(cfg.ocr)

    tasks = []
    for key in keys:
        local = local_by_key.get(key)
        video_file = ds / "videos" / local if local else None
        if video_file is None or not video_file.exists():
            print(f"{key}: missing local video ({local}), skipped", flush=True)
            continue
        annots = fa[fa["video_key"] == key].copy()
        info = probe_video(video_file)
        first = annots.iloc[0]
        tasks.append({
            "video_file": str(video_file),
            "file": video_file.name,
            "stem": video_file.stem,
            "video_key": key,
            "annots": annots,
            "fps": info.fps,
            "n_frames": info.n_frames,
            "duration_probe": info.duration,
            "match_strategy": first.get("match_strategy", ""),
            "duration_sheet": first.get("duration", ""),
            "overrides": overrides,
            "ocr_cache_entry": ocr_cache.get(video_file.name),
            "force": args.force,
            "no_ocr": False,
            "ocr_cfg": ocr_cfg,
            "out_dir": out,
            "search_window_s": cfg.ocr.search_window_s,
            "attr_cols": attr_cols,
            "write_images": True,
            "jpeg_quality": cfg.output.jpeg_quality,
            "frames_subdir": cfg.output.frames_subdir,
        })

    if args.workers > 1:
        with Pool(args.workers, initializer=_init_ocr, initargs=(args.ocr_threads,)) as pool:
            results = pool.map(_process_video_task, tasks)
    else:
        _init_ocr(args.ocr_threads)
        results = [_process_video_task(t) for t in tasks]

    new_rows, new_video_rows = [], []
    for task, r in zip(tasks, results):
        if r["ocr_result"] is not None:
            ocr_cache[r["file"]] = r["ocr_result"]
        new_rows.extend(r["rows"])
        new_video_rows.append(_make_video_row(task, r, overrides, "species"))
        s = r["summary"] or {}
        print(f"{task['video_key']}: status={r['status']} start={r['start']} "
              f"ok={s.get('ok')} verified={s.get('verified')} corrected={s.get('corrected')} "
              f"unverified={s.get('unverified')} not_found={s.get('not_found')} "
              f"out_of_range={s.get('out_of_range')} {r.get('error', '')}", flush=True)
    save_json(out / "ocr_cache.json", ocr_cache)

    # merge with rows from earlier runs, replacing any re-processed video_key
    done = {t["video_key"] for t in tasks}
    index_path, videos_path = out / "annotated_frames_index.csv", out / "videos.csv"
    new_index = write_index(new_rows, index_path.with_suffix(".new.csv"), attr_cols=attr_cols)
    new_videos = write_videos_table(new_video_rows, videos_path.with_suffix(".new.csv"))
    for path, new in ((index_path, new_index), (videos_path, new_videos)):
        if path.exists():
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            new = pd.concat([old[~old["video_key"].isin(done)], new.fillna("").astype(str)], ignore_index=True)
        new.to_csv(path, index=False)
        path.with_suffix(".new.csv").unlink()
    print(f"wrote {index_path} and {videos_path}", flush=True)


if __name__ == "__main__":
    main()
