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
    [--max-frames N] [--no-overlay] [--overwrite] [--root /scratch/.../vids]
# or: --video-dir DIR instead of --manifest
```

Resumable: re-running with the same `--out` skips videos whose `<out>/<stem>.json`
already exists (unless `--overwrite`). Per-video errors are logged to `index.jsonl` with
`status: "error"` and do not stop the run.

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
python overlay.py outputs/dev32_animal/<stem>.json --out /tmp/preview.mp4
```

## Slurm

`slurm/submit.sbatch` runs `slurm/run.sh`, which reads:

- `MANIFEST` — manifest to shard across the job's array tasks.
- `PROMPT` — required text prompt.
- `CHECKPOINT` — default `$SCRATCH/weights/sam3/sam3-safari-pos.pt`.
- `OUT` — default `outputs/<manifest-stem>_<prompt>`.
- `SIF` — path to the built Apptainer image.
- `EXTRA_ARGS` — passed through to `annotate.py` verbatim.

```bash
PROMPT=chimpanzee sbatch slurm/submit.sbatch                                   # smoke: 1 clip
MANIFEST=manifests/dev32.txt PROMPT=animal sbatch --array=0-3 slurm/submit.sbatch  # 4 GPUs, 8 clips each
```

Mirrors this repo's sibling verifier's `slurm/run_vllm.sh` conventions (thin sbatch
header, `awk 'NR % n == i'` sharding, `APPTAINERENV_*` exports, `TMPDIR=/tmp` forced
inside the container).

## Measured (2026-09-03, 1 GH200, prompt "chimpanzee", sam3-safari-pos.pt)

| run | clips | frames | wall | fps | videos/min/GPU |
|---|---|---|---|---|---|
| smoke (03190251.MP4, 720x404 @ 24 fps, first run incl. Triton compile) | 1 | 1454 | 5 min 32 s | 4.4 | 0.18 |
| dev8 (corpus MP4s, `manifests/dev8.txt`) | 8 | 11632 | 22 min | 8.9 | 0.37 |

Engine start (model load + Triton NMS kernel compile) is ~1 min per task. SAM3 runs every frame
at 1008 px, so a clip costs ~2.7 GPU-min and the 59,656-clip corpus would be ~2719 GPU-h
(~680 node-h) at this rate. Frame subsampling / lower resolution are the obvious levers
and are not implemented yet. Smoke clip: 8 tracks, max 6 concurrent, 2.5 MB tracks JSON, 29 MB overlay.

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

## Output contract

See `docs/OUTPUT_SCHEMA.md` for the full tracks JSON / `index.jsonl` field reference.
