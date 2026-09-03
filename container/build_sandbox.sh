#!/bin/bash
# Build the sam3 sandbox on the login node's LOCAL disk and stream it to $SCRATCH as a tarball.
# Then: NAME=sam3 sbatch slurm/build_sif.sbatch   (unpacks to node /tmp, writes $SCRATCH/containers/sam3.sif)
#
# Why this shape (all learned the hard way, see container/README.md):
#  - def-file builds chown files in the build tmp dir -> Lustre refuses -> TMPDIR must be local disk
#  - the final rootfs->sandbox copy onto Lustre fails with a tar EOF; cp -a of the tree gets killed
#  - streaming `tar -cf` of the finished sandbox onto Lustre works (25 GB in ~5 min)
#  - mksquashfs on the login node dies in the session cgroup, so the SIF is made in a job
# Run detached: setsid -f nohup container/build_sandbox.sh > $SCRATCH/logs/build-sam3-sandbox.log 2>&1 < /dev/null
set -x
U=$(id -u)
export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache APPTAINER_TMPDIR=/local/user/$U/apptainer-tmp TMPDIR=/local/user/$U/apptainer-tmp
mkdir -p "$APPTAINER_TMPDIR" "$SCRATCH/containers"
cd "$(dirname "$0")/.."
rm -rf /local/user/$U/sam3-sandbox
apptainer build --sandbox /local/user/$U/sam3-sandbox container/sam3.def && echo BUILD_OK || { echo BUILD_FAIL; exit 1; }
tar -C /local/user/$U -cf "$SCRATCH/containers/sam3-sandbox.tar" sam3-sandbox && echo TAR_OK
