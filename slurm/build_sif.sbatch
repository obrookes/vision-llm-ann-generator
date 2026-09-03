#!/bin/bash
#SBATCH --job-name=build-sif
#SBATCH --account=brics.b5bd
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm-build-sif-%j.out
# Convert a sandbox into a SIF on a compute node (login-node mksquashfs dies in the session cgroup,
# and apptainer 1.4.1 has no --mksquashfs-procs/-mem flags to cap it). No network needed.
#
# Input is either a sandbox dir on $SCRATCH or, preferably, a tarball of one (the login node cannot
# copy a 25 GB sandbox tree onto Lustre reliably, but streaming `tar -cf` onto Lustre works; see
# container/README.md). The tarball is unpacked to node-local /tmp first.
#
# Usage:
#   NAME=sam3 sbatch slurm/build_sif.sbatch                       # uses $SCRATCH/containers/sam3-sandbox.tar if present, else -sandbox/
#   TARBALL=$SCRATCH/containers/sam3-sandbox.tar sbatch slurm/build_sif.sbatch
set -euo pipefail
NAME="${NAME:-sam3}"
TARBALL="${TARBALL:-$SCRATCH/containers/${NAME}-sandbox.tar}"
SANDBOX="${SANDBOX:-$SCRATCH/containers/${NAME}-sandbox}"

export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache APPTAINER_TMPDIR=/tmp TMPDIR=/tmp
if [ -f "$TARBALL" ]; then
    echo "unpacking $TARBALL to /tmp"; df -h /tmp
    tar -C /tmp -xf "$TARBALL"
    SANDBOX="/tmp/$(basename "$TARBALL" .tar)"
fi
[ -d "$SANDBOX" ] || { echo "build_sif.sbatch: no sandbox at $SANDBOX and no tarball at $TARBALL" >&2; exit 1; }
time apptainer build --force "$SCRATCH/containers/${NAME}.sif" "$SANDBOX"
ls -la "$SCRATCH/containers/${NAME}.sif"
