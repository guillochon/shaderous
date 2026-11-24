#!/bin/bash
# The magnetized double white dwarf (examples/setups/double-wd-random-field.shaderous.json) on
# an adaptive mesh (default 2048^3 with 5 levels), as a chain of jobs: each carries on from the newest checkpoint, stops
# itself before the job's time runs out (writing a checkpoint), and submits the next, until the
# simulated time reaches UNTIL. Submit once:
#   sbatch tools/headless/slurm/dwd.sbatch
# Settings (environment, passed on to the jobs that follow):
#   SETUP      the setup document (default examples/setups/double-wd-random-field.shaderous.json);
#              like N and AMR it applies to a new run only
#   N, AMR     the finest grid and the number of levels (default 2048 and 5); they apply to a new
#              run only, as a resumed one keeps its checkpoint's mesh
#   ZOOM       the screenshots' framing, for the setup's box: the slice's zoom and the volume's
#              distance are the box of 8's times this (default 1; 2 for a box of 16)
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
#   MEM        host memory for the jobs that follow (sbatch --mem, such as 64G for a run with the
#              aprox13 species); unset, the script's own
# A setup that watches for burning (he4, c12 and o16 alone with the network's burning:
# burn/watch.ts) stops with 'burn' once it is found: the next job carries on from that
# checkpoint with all thirteen of the network's species (the runner's --switch-burn), and the
# chain goes on from there as usual. SWITCH_BURN is the chain's own: 1 for that one job.
# Partition: umd-cscdr-gpu, the UMass Dartmouth groups' own (A100 80 GB, never preempted, jobs of
# up to 5 days). Without access to it: sbatch -p gpu-highvram --time=2-00:00:00 (or with
# gpu-preempt-highvram as well, where a job can be preempted: it is then requeued and loses only
# the time since its last checkpoint). The jobs that follow keep the partition and time limit
# the job ran with.
# Host memory: about 8-9 times a checkpoint's size at its peak (35 GB for the 1024^3 run's 4.2 GB
# checkpoints, 22 GB so far for the 2048^3 run's 2 GB). Check with sstat -j <job>.batch
# --format=MaxRSS, and pass sbatch --mem for a run with species or larger checkpoints.
# Stop the chain: touch RUN_DIR/seg-<job>/STOP (the job stops at its next screenshot and
# writes a checkpoint) and scancel any job waiting, or touch RUN_DIR/DONE before the next starts.
#SBATCH --job-name=shaderous-dwd
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm-dwd-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"

export SETUP="${SETUP:-examples/setups/double-wd-random-field.shaderous.json}"
export N="${N:-2048}" AMR="${AMR:-5}" EVERY="${EVERY:-}" ZOOM="${ZOOM:-1}"
export RUN_DIR="${RUN_DIR:-$SHADEROUS_DIR/runs/dwd-$N}"
export UNTIL="${UNTIL:-100}" SHOT_EVERY="${SHOT_EVERY:-1800}" ATHDF_EVERY="${ATHDF_EVERY:-4}"
export KEEP="${KEEP:-2}" KEEP_EVERY="${KEEP_EVERY:-0}" MARGIN="${MARGIN:-1800}" MAX_JOBS="${MAX_JOBS:-20}"
export JOB_NUMBER=$(( ${JOB_NUMBER:-0} + 1 ))
switch_burn="${SWITCH_BURN:-0}"
# Only the job after a 'burn' stop switches: the jobs this one submits start without it.
export SWITCH_BURN=0
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
  if [ "$switch_burn" = "1" ]; then
    start+=(--switch-burn)
    echo "switching to all thirteen of the network's species"
  fi
else
  start=(--setup "$SETUP" --n "$N" --amr "$AMR")
fi

if [ -n "$EVERY" ]; then spacing=(--every "$EVERY"); else spacing=(--every-wall "$SHOT_EVERY"); fi
# Framed for a box of 8; a larger box is seen from nearer, and its slice zoomed in, by ZOOM.
dist=$(awk -v z="$ZOOM" 'BEGIN { print 0.7 / z }')
zoom=$(awk -v z="$ZOOM" 'BEGIN { print 1.5 * z }')

# Older checkpoints are deleted while the run goes on, so they do not fill the disk.
( while sleep 600; do ckpt prune "$RUN_DIR" "$KEEP" "$KEEP_EVERY"; done ) &
pruner=$!

in_container node tools/headless/cli.mjs run "${start[@]}" \
  --until "$UNTIL" --wall "$wall" "${spacing[@]}" --athdf-every "$ATHDF_EVERY" \
  --gpu-memory "$gb" --width 1024 --overlays \
  --view field=0 --view log=true --view "tf=0:0,0.15:0.15,0.4:0.25,0.7:0.3,1:0.5" \
  --shot "vol:mode=volume,showBlocks=false,fieldLines=true,fieldLineCount=400,fieldLineFloor=0.001,dist=$dist,pitch=0.45,yaw=1.2,minRho=0.05,opacity=120,box=false" \
  --shot "slice:mode=slice,sliceAxis=2,zoom=$zoom,showBlocks=false,fieldLines=true,fieldLineCount=300,fieldLineFloor=0.00001" \
  --out "$seg"
status=$?
kill "$pruner" 2>/dev/null
ckpt prune "$RUN_DIR" "$KEEP" "$KEEP_EVERY"

stopped=$(ckpt stopped "$RUN_DIR")
echo "runner exit $status, stopped: ${stopped:-unknown}"
next() {
  local limit
  limit=$(scontrol show job "$SLURM_JOB_ID" -o | grep -oP 'TimeLimit=\K\S+')
  sbatch -p "$SLURM_JOB_PARTITION" ${limit:+--time="$limit"} ${MEM:+--mem="$MEM"} "$SHADEROUS_DIR/tools/headless/slurm/dwd.sbatch"
}
if [ "$status" -eq 0 ] && [ "$stopped" = "wall" ] && [ "$JOB_NUMBER" -lt "$MAX_JOBS" ]; then
  next
elif [ "$status" -eq 0 ] && [ "$stopped" = "burn" ] && [ "$JOB_NUMBER" -lt "$MAX_JOBS" ]; then
  echo "burning found: the next job carries on with all thirteen species"
  SWITCH_BURN=1 next
elif [ "$stopped" = "time" ]; then
  touch "$RUN_DIR/DONE"
  echo "reached t = $UNTIL"
fi
