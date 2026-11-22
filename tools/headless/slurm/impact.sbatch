#!/bin/bash
# The Moon-forming impact for the paper's figure (paper/paper/figures/make_impact.py): the preset
# moon-forming-impact in a box 10 Earth radii across at 1024 finest cells with 4 levels of
# refinement (102 cells to an Earth radius), run by paper/paper/figures/impact_run.mjs in one job.
# Every DT of simulated time it saves slices through the orbital plane (density, the passive
# scalar that tells Theia from the proto-Earth, the iron fraction, the refinement level; 2048^2
# float32 .npy), two renderings (isosurfaces of Theia's share and of the density, from
# runs/impact-shots.json beside this script), the diagnostics and the impact() report. Submit:
#   sbatch tools/headless/slurm/impact.sbatch
# Settings (environment):
#   RUN_DIR  where the outputs go (default runs/impact-1024-box10 in the checkout); about 70 MB
#            an output
#   DT       simulated time between outputs (default 0.25, a time unit is 805.5 s)
#   UNTIL    the last output (default 16, 3.6 h)
#   N, AMR, BOX  the finest grid, the levels and the box in Earth radii (default 1024, 4, 10)
# Stop early: scancel the job; the outputs already written stay.
#SBATCH --job-name=shaderous-impact
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-impact-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"

N="${N:-1024}" AMR="${AMR:-4}" BOX="${BOX:-10}" DT="${DT:-0.25}" UNTIL="${UNTIL:-16}"
RUN_DIR="${RUN_DIR:-$SHADEROUS_DIR/runs/impact-$N-box$BOX}"
mkdir -p "$RUN_DIR"
gb=$(gpu_memory_gb)
times=$(awk -v dt="$DT" -v end="$UNTIL" 'BEGIN { s = ""; for (i = 1; i * dt <= end + 1e-9; i++) s = s (s ? "," : "") i * dt; print s }')
{
  echo "node: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo "checkout: $(git -C "$SHADEROUS_DIR" log --oneline -1)"
  echo "--gpu-memory $gb; outputs every $DT to $UNTIL"
} | tee "$RUN_DIR/device.txt"

in_container node paper/paper/figures/impact_run.mjs "$RUN_DIR" \
  --setup "problem=moon-forming-impact&n=$N&amr=$AMR" --doc "domain.box=$BOX" \
  --gpu-memory "$gb" --times "0,$times" --every "$DT" --res 2048 \
  --vars rho,r0,fe85si15,level --shots tools/headless/slurm/impact-shots.json \
  2>&1 | tee -a "$RUN_DIR/log.txt"
echo "runner exit ${PIPESTATUS[0]}"
