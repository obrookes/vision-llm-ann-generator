# vision-llm-ann-generator

Runs SAM3 (official `facebookresearch/sam3`, SA-FARI fine-tuned weights) text-prompted
tracking over a directory/manifest of videos and writes, per video, an overlay video plus
a machine-readable tracks JSON. Partner repo to
[`vision-llm-ann-verifier`](../vision-llm-ann-verifier): this repo *produces* the
overlay clips + manifests that repo *reviews*.

## Layout

- `annotate.py` — main entry point: manifest/dir + prompt + checkpoint -> per-video
  overlay + tracks JSON + `index.jsonl`.
- `sam3_runner.py` — thin wrapper over the SAM3 video predictor (the only module that
  imports `sam3`/`torch`; everything else runs on CPU without them).
- `overlay.py` — render a tracks JSON over its source video (cv2, palette-per-id mask
  blend, box, `id N` label). Also runnable standalone on an existing tracks JSON.
- `tracks.py` — tracks schema (`schema_version: 1`), pure-numpy COCO-compatible RLE
  codec, JSON read/write, `stem_for()` path flattening, `summarise()`.
- `manifest.py` — build manifests: walk a video dir -> `videos.txt`, or list an
  `annotate.py` output dir's overlays -> a verifier-ready manifest.
- `container/` — `sam3.def` (NGC PyTorch 25.06 arm64 + sam3), `build_sandbox.sh`, README with the
  build gotchas.
- `slurm/` — `run.sh` / `submit.sbatch` / `build_sif.sbatch`.
- `tools/container_check.py` — python/torch/cuda/sam3 import check inside the container.
- `manifests/` — `smoke.txt`, `dev8.txt`, `dev32.txt`.
- `outputs/` — gitignored; per-run subdirectories go here.
- `docs/OUTPUT_SCHEMA.md` — the tracks JSON / `index.jsonl` contract shared with the
  verifier.

## Setup (Isambard-AI)

```bash
# weights (base + three SA-FARI fine-tunes, 3.4 GB each); default is sam3-safari-pos.pt
mkdir -p $SCRATCH/weights/sam3 && cp ~/Unmarked-Anything/weights/sam3/*.pt $SCRATCH/weights/sam3/
# container: sandbox on login-node local disk -> tarball on $SCRATCH -> SIF built in a job (~35 min total)
setsid -f nohup container/build_sandbox.sh > $SCRATCH/logs/build-sam3-sandbox.log 2>&1 < /dev/null
NAME=sam3 sbatch slurm/build_sif.sbatch        # -> $SCRATCH/containers/sam3.sif (12.8 GB)
```

`container/README.md` explains why the build is shaped like that (Lustre chown refusal, scratch
inode quota, tmpfs `/tmp` on nodes).

## Usage

CLI, on a GPU node (or inside the container):

```bash
python annotate.py --manifest manifests/dev32.txt --prompt animal --out outputs/dev32_animal \
    [--checkpoint $SCRATCH/weights/sam3/sam3-safari-pos.pt] [--score-thresh 0.5] \
    [--mode video|image] [--sample-fps 6] [--overlay-all-frames] \
    [--frames-csv annotations.csv] [--extra-frames 26,75] [--stop-after-last-extra-s 5.0] \
    [--max-frames N] [--no-overlay] [--overwrite] [--root /scratch/.../vids]
# or: --video-dir DIR instead of --manifest
```

Resumable: re-running with the same `--out` skips videos whose `<out>/<stem>.json`
already exists (unless `--overwrite`). Per-video errors are logged to `index.jsonl` with
`status: "error"` and do not stop the run.

`--frames-csv`/`--extra-frames` force specific source-frame indices into the sampled set
even when they're off the `--sample-fps` grid (e.g. so a human-annotated frame lands in
the tracks JSON exactly): `--frames-csv` matches rows to each video by basename
(`manifest.frames_by_video`), `--extra-frames` applies the same ad hoc comma-separated
indices to every video in the run, and both can be combined. `--stop-after-last-extra-s`
optionally stops decoding a fixed number of seconds after the last extra frame requested
for a video (default: decode the whole video). Recorded in the tracks JSON as
`extra_frames`/`frames_csv` (see `docs/OUTPUT_SCHEMA.md`).

Hand off the rendered overlays to the verifier:

```bash
python manifest.py --from-outputs outputs/dev32_animal \
    --write ../vision-llm-ann-verifier/manifests/dev32_animal.txt
```

Build a manifest from a raw video directory:

```bash
python manifest.py --video-dir /scratch/.../vids --ext .MP4,.mp4,.avi --write manifests/all.txt
```

Render (or re-render) an overlay from an existing tracks JSON directly:

```bash
python overlay.py outputs/dev32_animal/<stem>.json --out /tmp/preview.mp4  # [--all-frames]
```

## Slurm

`slurm/submit.sbatch` runs `slurm/run.sh`, which reads:

- `MANIFEST` — manifest to shard across the job's array tasks.
- `PROMPT` — required text prompt.
- `CHECKPOINT` — default `$SCRATCH/weights/sam3/sam3-safari-pos.pt`.
- `MODE` — `video` (default) or `image`; see "Modes" below.
- `SAMPLE_FPS` — default `6`; frame rate SAM3 processes the video at (subsampled from
  source fps). `0` processes every source frame.
- `FRAMES_CSV` — optional; forwarded as `annotate.py --frames-csv`.
- `OUT` — default `outputs/<manifest-stem>_<prompt>` (or `..._<prompt>_<mode>` when
  `MODE` is not `video`, so video and image runs don't collide).
- `SIF` — path to the built Apptainer image.
- `EXTRA_ARGS` — passed through to `annotate.py` verbatim (e.g. `--extra-frames`,
  `--stop-after-last-extra-s`).

```bash
PROMPT=chimpanzee sbatch slurm/submit.sbatch                                   # smoke: 1 clip
MANIFEST=manifests/dev32.txt PROMPT=animal sbatch --array=0-3 slurm/submit.sbatch  # 4 GPUs, 8 clips each
MODE=image PROMPT=animal sbatch slurm/submit.sbatch                            # detector-only, no tracker
```

Mirrors this repo's sibling verifier's `slurm/run_vllm.sh` conventions (thin sbatch
header, `awk 'NR % n == i'` sharding, `APPTAINERENV_*` exports, `TMPDIR=/tmp` forced
inside the container).

## Resolution and frame rate (why the defaults are what they are)

**Spatial: fixed at 1008x1008, not a flag.** SAM3's checkpoints hardcode a square
1008x1008 resize everywhere (`model_builder.py`: `ViT(img_size=1008, patch_size=14)` ->
72x72 tokens, decoder `resolution=1008`, tracker `feat_sizes=[72,72]`); worse, the RoPE
`freqs_cis` buffers are baked into the checkpoint at 72x72/24x24 and `vitdet.py` asserts
the exact shape -- 504 px and 728 px inputs raise `AssertionError`. Running at a smaller
size would need multiple builder monkeypatches, non-strict loading, and a rescaled RoPE
-> out of distribution for no real gain (we already run at exactly the training
resolution). So there is no `--max-side`/resolution flag; this cost is fixed.

**Temporal: SAM3 is trained/evaluated at 6 fps, so we subsample to it.** The SAM 3 paper
(section 5) reports training videos "have 84.1 frames at 6 fps"; the SA-FARI paper says
annotations were made by the SAM 3 data engine "at 6 fps"; and `visualization_utils.py`
maps the `sa_fari_test` split to `sa_fari/JPEGImages_6fps`. The video predictor itself
has no stride/fps option -- `io_utils.py` decodes every frame it's given, resizes to
1008 px fp16, and puts the whole tensor on the GPU (~6 MB/frame), so feeding it our
24-30 fps corpus clips directly runs the model at 4-5x its trained rate for no benefit.
This repo does the decimation itself (`--sample-fps`, default `6`) and passes SAM3 a
list of PIL frames at the target rate, which also matches the training/eval
distribution rather than fighting it.

## Modes

- **`video`** (default): the detector runs on every *sampled* frame, followed by
  SAM2-style memory tracking across frames (`num_maskmem=7`, `hotstart_delay=15`,
  `max_trk_keep_alive=30`, `recondition_every_nth_frame=16`). Object `id`s are stable
  per-video track ids: the same physical object keeps the same `id` across frames.
- **`image`**: the detector alone, run independently per sampled frame -- no temporal
  linking. `id`s are a per-frame detection index (`0..n-1`) and are **not** stable
  across frames.
- **When to use which**: `image` for pure detection QA, a fast baseline, or when you
  only care about per-frame presence/counts; `video` when the verifier (or any
  downstream consumer) needs to refer to the same object across frames via a stable id.

## Measured (2026-09-03, 1 GH200, prompt "chimpanzee", sam3-safari-pos.pt)

| smoke, `--sample-fps 6`, video mode (job 6274195) | 1 | 364 | 1 min 26 s | 4.3 | 0.70 |
| smoke, `--sample-fps 6`, image mode (job 6274229) | 1 | 364 | 43 s | 8.8 | 1.40 |
| dev8, `--sample-fps 6`, video mode (job 6274234) | 8 | 2912 | 5 min 49 s | 9.2 | 1.38 |
| dev8, `--sample-fps 6`, image mode (job 6274289) | 8 | 2912 | 4 min 14 s | 12.2 | 1.89 |

At the 6 fps default, dev8 costs 43.6 s/clip in video mode (3.8x faster than every-frame) and
31.8 s/clip in image mode, i.e. ~720 GPU-h (video) or ~530 GPU-h (image) for the 59,656-clip
corpus, from ~2700 GPU-h before. fps columns are sampled frames per second of tracking time
(`fps_processed`); wall includes ~1 min engine start per task. Track counts on dev8 differ
between modes only because image-mode "tracks" are per-frame detection slots.

| run | clips | frames | wall | fps | videos/min/GPU |
|---|---|---|---|---|---|
| smoke (03190251.MP4, 720x404 @ 24 fps, first run incl. Triton compile) | 1 | 1454 | 5 min 32 s | 4.4 | 0.18 |
| dev8 (corpus MP4s, `manifests/dev8.txt`) | 8 | 11632 | 22 min | 8.9 | 0.37 |

Engine start (model load + Triton NMS kernel compile) is ~1 min per task. The first two rows
ran every source frame (the pre-`--sample-fps` default): ~2.7 GPU-min per clip, ~2700 GPU-h
for the corpus. Smoke clip: 8 tracks, max 6 concurrent, 2.5 MB tracks JSON, 29 MB overlay.

## Gotchas

- **SA-FARI checkpoints are a different SAM3 variant** from Meta's `sam3.pt`: no decoder presence
  token, but a DotProductScoring presence head on the segmentation head. `sam3_runner.py` detects
  this from the state-dict keys and patches the two builder factories so strict loading passes
  (sam3 0.1.0's `has_presence_token` flag is ignored by its own transformer builder). The video
  inference path never reads the seg-head presence logit, so it only matters for loading.
- **Triton needs `libcuda.so`**: inside the NGC image `ldconfig -p` points at a compat dir that
  is absent on the node, and `apptainer --nv` only provides `libcuda.so.1`. `run.sh` creates
  `$SCRATCH/lib/triton-libcuda/{libcuda.so,libcuda.so.1}` -> `/.singularity.d/libs/libcuda.so.1`
  and sets `TRITON_LIBCUDA_PATH` to it.
- `annotate.py` uses a single predictor per process; each array task is one GPU.
- **Image mode's presence-token gotcha**: the image builder (`build_sam3_image_model`)
  loads the SA-FARI checkpoint with `strict=False` after stripping the `detector.`
  prefix, which silently drops the seg-head's DotProductScoring presence weights and
  leaves the decoder's presence token randomly initialised. Since the final score
  multiplies by that presence logit, an unpatched image-mode run produces
  near-degenerate scores. `sam3_runner.py` applies the same builder patch it uses for
  video mode and adjusts scoring accordingly for image mode too -- see the
  `_patch_segmentation_head_with_presence()` call sites there.

## Output contract

See `docs/OUTPUT_SCHEMA.md` for the full tracks JSON / `index.jsonl` field reference.
