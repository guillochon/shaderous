#!/bin/bash
# The magnetized double white dwarf (examples/setups/double-wd-random-field.shaderous.json) on
# an adaptive mesh (default 2048^3 with 5 levels), as a chain of jobs: each carries on from the newest checkpoint, stops
# itself before the job's time runs out (writing a checkpoint), and submits the next, until the
# simulated time reaches UNTIL. Submit once:
#   sbatch tools/headless/slurm/dwd.sbatch
# Settings (environment, passed on to the jobs that follow):
#   N, AMR     the finest grid and the number of levels (default 2048 and 5); they apply to a new
#              run only, as a resumed one keeps its checkpoint's mesh
#   RUN_DIR    where the run's segments go (default runs/dwd-N in the checkout). Checkpoints
#              are several GB each: point it at a work or scratch space if home is small, and add
#              that space to SHADEROUS_BINDS
#   UNTIL      the simulated time to stop at (default 100, about five orbits; one is ~19.5)
#   EVERY      simulated time between screenshots: evenly spaced, at multiples of it across jobs.
#              Unset: SHOT_EVERY seconds of wall time instead (default 1800)
#   ATHDF_EVERY  a checkpoint every this many screenshots (default 4: every 2 hours)
#   KEEP       checkpoints to keep (default 2); older ones are deleted as new ones land
#   KEEP_EVERY also keep, for good, the first checkpoint at or after each multiple of this much
#              simulated time (default 0: none)
#   MARGIN     seconds left at the end of a job for the last checkpoint (default 1800)
#   MAX_JOBS   the most jobs the chain may run (default 20)
# Partitions: gpu-highvram, or gpu-preempt-highvram (more 80 GB cards, often free at once) where
# a job can be preempted; it is then requeued and loses only the time since its last checkpoint.
# Stop the chain: touch RUN_DIR/seg-<job>/STOP (the job stops at its next screenshot and
# writes a checkpoint) and scancel any job waiting, or touch RUN_DIR/DONE before the next starts.
#SBATCH --job-name=shaderous-dwd
#SBATCH --partition=gpu-highvram,gpu-preempt-highvram
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-dwd-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"

export N="${N:-2048}" AMR="${AMR:-5}" EVERY="${EVERY:-}"
export RUN_DIR="${RUN_DIR:-$SHADEROUS_DIR/runs/dwd-$N}"
export UNTIL="${UNTIL:-100}" SHOT_EVERY="${SHOT_EVERY:-1800}" ATHDF_EVERY="${ATHDF_EVERY:-4}"
export KEEP="${KEEP:-2}" KEEP_EVERY="${KEEP_EVERY:-0}" MARGIN="${MARGIN:-1800}" MAX_JOBS="${MAX_JOBS:-20}"
export JOB_NUMBER=$(( ${JOB_NUMBER:-0} + 1 ))
ckpt() { in_container node tools/headless/slurm/checkpoints.mjs "$@"; }

if [ -e "$RUN_DIR/DONE" ]; then echo "$RUN_DIR/DONE exists: nothing to do"; exit 0; fi
mkdir -p "$RUN_DIR"
# A preempted job is requeued under the same id: it starts again from the newest checkpoint, in
# a segment of its own.
seg="$RUN_DIR/seg-$SLURM_JOB_ID${SLURM_RESTART_COUNT:+-r$SLURM_RESTART_COUNT}"
from=$(ckpt latest "$RUN_DIR")
wall=$(( $(job_seconds_left) - MARGIN ))
gb=$(gpu_memory_gb)
echo "job $JOB_NUMBER of at most $MAX_JOBS on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), --gpu-memory $gb, --wall $wall"
echo "from: ${from:-the initial conditions}"

if [ -n "$from" ]; then
  start=(--from "$from")
else
  start=(--setup examples/setups/double-wd-random-field.shaderous.json --n "$N" --amr "$AMR")
fi

if [ -n "$EVERY" ]; then spacing=(--every "$EVERY"); else spacing=(--every-wall "$SHOT_EVERY"); fi

# Older checkpoints are deleted while the run goes on, so they do not fill the disk.
( while sleep 600; do ckpt prune "$RUN_DIR" "$KEEP" "$KEEP_EVERY"; done ) &
pruner=$!

in_container node tools/headless/cli.mjs run "${start[@]}" \
  --until "$UNTIL" --wall "$wall" "${spacing[@]}" --athdf-every "$ATHDF_EVERY" \
  --gpu-memory "$gb" --width 1024 --overlays \
  --view field=0 --view log=true --view "tf=0:0,0.15:0.15,0.4:0.25,0.7:0.3,1:0.5" \
  --shot "vol:mode=volume,showBlocks=false,fieldLines=true,fieldLineCount=400,fieldLineFloor=0.001,dist=0.7,pitch=0.45,yaw=1.2,minRho=0.05,opacity=120,box=false" \
  --shot "slice:mode=slice,sliceAxis=2,zoom=1.5,showBlocks=false,fieldLines=true,fieldLineCount=300,fieldLineFloor=0.00001" \
  --out "$seg"
status=$?
kill "$pruner" 2>/dev/null
ckpt prune "$RUN_DIR" "$KEEP" "$KEEP_EVERY"

stopped=$(ckpt stopped "$RUN_DIR")
echo "runner exit $status, stopped: ${stopped:-unknown}"
if [ "$status" -eq 0 ] && [ "$stopped" = "wall" ] && [ "$JOB_NUMBER" -lt "$MAX_JOBS" ]; then
  sbatch "$SHADEROUS_DIR/tools/headless/slurm/dwd.sbatch"
elif [ "$stopped" = "time" ]; then
  touch "$RUN_DIR/DONE"
  echo "reached t = $UNTIL"
fi
