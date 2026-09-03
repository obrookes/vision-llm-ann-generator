#!/bin/bash
# Body of a SAM3 annotate job: SIF selection, manifest sharding, container env, apptainer exec.
# `exec bash slurm/run.sh` from within slurm/submit.sbatch, or source the same env vars
# (MANIFEST, PROMPT, ...) and run it directly on an interactive allocation.

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

MANIFEST="${MANIFEST:-manifests/smoke.txt}"

if [ -z "${PROMPT:-}" ]; then
    echo "run.sh: PROMPT is required, e.g. PROMPT=animal sbatch slurm/submit.sbatch" >&2
    exit 1
fi

CHECKPOINT="${CHECKPOINT:-$SCRATCH/weights/sam3/sam3-safari-pos.pt}"

SIF="${SIF:-$SCRATCH/containers/sam3.sif}"
[ -e "$SIF" ] || SIF="$SCRATCH/containers/sam3-sandbox"   # login-node mksquashfs fails (pids limit); sandbox works too
if [ ! -e "$SIF" ]; then
    echo "run.sh: no container image found." >&2
    echo "  looked for: \$SCRATCH/containers/sam3.sif and \$SCRATCH/containers/sam3-sandbox" >&2
    echo "  build it with: NAME=sam3 sbatch slurm/build_sif.sbatch" >&2
    echo "  (sandbox itself is built first: apptainer build --sandbox \$SCRATCH/containers/sam3-sandbox container/sam3.def)" >&2
    exit 1
fi

N="${SLURM_ARRAY_TASK_COUNT:-1}"; I="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p outputs/shards "$SCRATCH/hf-cache" "$SCRATCH/torch-cache" "$SCRATCH/triton-cache"
SHARD="outputs/shards/$(basename "$MANIFEST" .txt)_${I}_of_${N}.txt"
awk -v n="$N" -v i="$I" 'NR % n == i' "$MANIFEST" > "$SHARD"

PROMPT_SLUG=$(echo "$PROMPT" | tr ' ' '_')
OUT="${OUT:-outputs/$(basename "$MANIFEST" .txt)_${PROMPT_SLUG}}"
mkdir -p "$OUT"

export APPTAINERENV_HF_HOME="$SCRATCH/hf-cache"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TMPDIR=/tmp
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torch-cache"
export APPTAINERENV_TRITON_CACHE_DIR="$SCRATCH/triton-cache"
# Triton (used by sam3.perflib NMS kernels) finds libcuda via `ldconfig -p`, which inside the NGC
# image points at /usr/local/cuda/compat/lib (absent on the node), and it links with -lcuda so it
# needs a `libcuda.so` name, while `apptainer --nv` only provides libcuda.so.1 under
# /.singularity.d/libs. Give it a dir with both names (symlink targets resolve inside the container).
TRITON_LIBCUDA_DIR="$SCRATCH/lib/triton-libcuda"
mkdir -p "$TRITON_LIBCUDA_DIR"
ln -sfn /.singularity.d/libs/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so.1"
ln -sfn /.singularity.d/libs/libcuda.so.1 "$TRITON_LIBCUDA_DIR/libcuda.so"
export APPTAINERENV_TRITON_LIBCUDA_PATH="$TRITON_LIBCUDA_DIR"

GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPUS=$(nvidia-smi -L | wc -l)
fi

echo "== run.sh: MANIFEST=$SHARD PROMPT=$PROMPT CHECKPOINT=$CHECKPOINT OUT=$OUT SIF=$SIF task=${I}/${N} GPUs=$GPUS =="

START=$(date +%s)

apptainer exec --nv --bind "/lus,/scratch,$HOME" "$SIF" \
    python3 annotate.py \
        --manifest "$SHARD" \
        --prompt "$PROMPT" \
        --checkpoint "$CHECKPOINT" \
        --out "$OUT" \
        --index-name "index_${I}_of_${N}.jsonl" \
        ${EXTRA_ARGS}

STATUS=$?
END=$(date +%s)
echo "== run.sh: done in $((END - START))s, exit=$STATUS =="
exit $STATUS
