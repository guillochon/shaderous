#!/bin/bash
# The double white dwarf fill sweep (plans/dwd-second-run.md section 1): one run per Roche lobe
# fill, from tools/headless/slurm/dwd-sweep/f<FILL>.shaderous.json (box 16, 2048^3 with 5 levels,
# one orbit to t = 20). One job per fill, no chaining:
#   for f in 0.935 0.940 0.945 0.950 0.955 0.960 0.965; do
#     FILL=$f sbatch --job-name=dwd-f$f tools/headless/slurm/dwd-sweep.sbatch
#   done
# Settings (environment):
#   FILL       the fill, as the setup file names it (required)
#   SETUPS     the folder of setups, f<FILL>.shaderous.json (default tools/headless/slurm/dwd-sweep;
#              tools/headless/slurm/dwd-he-sweep for the helium donor, with UNTIL=32 and ATHDF_EVERY=8)
#   SWEEP_DIR  where the runs go, each in f<FILL>/ (default the group's work space)
#   UNTIL      the simulated time to stop at (default 20, one orbit)
#   EVERY      simulated time between screenshots (default 0.5)
#   ATHDF_EVERY  an .athdf file every this many screenshots (default 5: every 2.5)
#   GPU_MEMORY GB the simulation may take (default 74)
#   SLICE_ZOOM, VOL_DIST  the shots' framing (default 3 and 0.35; 2.4 and 0.45 for the wider
#              helium-donor pair)
# The views are dwd.sbatch's, framed for the box of 16 (twice its 8): dist halved, zoom doubled.
# Measure the mass transfer from the .athdf files with dwd-mdot.mjs (see its header).
# Host memory peaks at about 8-9 times a checkpoint's size: 17-24 GB for the CO sweep (2 GB
# checkpoints), 36-45 GB with the 13 aprox13 species (4.8 GB): sbatch --mem=64G for those.
#   RESUME     1: carry on from the newest checkpoint in f<FILL>/ or a folder below it, into
#              f<FILL>/resume-<job>/ (dwd-mdot.mjs reads the folders below a fill's too)
#   STALL      seconds without a file written before a run counts as hung (default 5400; an
#              output comes every 10 to 25 minutes, the start takes about 10): the job then stops
#              it and submits itself again with RESUME=1, on what is likely another GPU
#   RESUMES    how many times it may do that (default 3)
#   POLL       seconds between the watchdog's looks (default 60)
# Six runs of the helium-donor sweep hung for hours (2026-11-26): the GPU idle, Chrome waiting in
# the driver, no error, and other runs on the same nodes went on. Hence the watchdog.
# Stop a run: touch SWEEP_DIR/f<FILL>/STOP (it stops at its next output, writing an .athdf file).
#SBATCH --job-name=dwd-sweep
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --account=pi_rfisher1_umassd_edu
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-dwd-sweep-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"

: "${FILL:?set FILL, such as FILL=0.950}"
SWEEP_DIR="${SWEEP_DIR:-/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/dwd-sweep}"
UNTIL="${UNTIL:-20}" EVERY="${EVERY:-0.5}" ATHDF_EVERY="${ATHDF_EVERY:-5}" GPU_MEMORY="${GPU_MEMORY:-74}"
SLICE_ZOOM="${SLICE_ZOOM:-3}" VOL_DIST="${VOL_DIST:-0.35}"
export SHADEROUS_BINDS="${SHADEROUS_BINDS:-/work/pi_rfisher1_umassd_edu}"
SETUPS="${SETUPS:-tools/headless/slurm/dwd-sweep}"
setup="$SETUPS/f$FILL.shaderous.json"
out="$SWEEP_DIR/f$FILL"
RESUME="${RESUME:-0}" STALL="${STALL:-5400}" RESUMES="${RESUMES:-3}" POLL="${POLL:-60}"
[ -f "$SHADEROUS_DIR/$setup" ] || { echo "no setup $setup"; exit 1; }
if [ "$RESUME" != 1 ] && [ -e "$out/manifest.json" ] && grep -q '"stopped"' "$out/manifest.json" 2>/dev/null; then
  echo "$out has a finished run: move it away to run again"; exit 1
fi
mkdir -p "$out"
echo "fill $FILL on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), --gpu-memory $GPU_MEMORY, out $out"

sim() {
  in_container node tools/headless/cli.mjs run "$@" \
    --until "$UNTIL" --every "$EVERY" --athdf-every "$ATHDF_EVERY" \
    --gpu-memory "$GPU_MEMORY" --width 1024 --overlays \
    --view field=0 --view log=true --view "tf=0:0,0.15:0.15,0.4:0.25,0.7:0.3,1:0.5" \
    --shot "vol:mode=volume,showBlocks=false,fieldLines=true,fieldLineCount=400,fieldLineFloor=0.001,dist=$VOL_DIST,pitch=0.45,yaw=1.2,minRho=0.05,opacity=120,box=false" \
    --shot "slice:mode=slice,sliceAxis=2,zoom=$SLICE_ZOOM,showBlocks=false,fieldLines=true,fieldLineCount=300,fieldLineFloor=0.00001"
}
state() { in_container node tools/headless/slurm/sweep-state.mjs "$@"; }
kill_tree() {
  local p
  for p in $(ps -o pid= --ppid "$1"); do kill_tree "$p"; done
  kill -KILL "$1" 2>/dev/null
}
# A run into folder $1 (the rest: the runner's arguments), watched: its exit status, or 124 if
# nothing was written to the folder for STALL seconds and it was stopped.
watched() {
  local dir=$1
  shift
  mkdir -p "$dir"
  sim "$@" --out "$dir" &
  local pid=$! last newest
  last=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL"
    newest=$(find "$dir" -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    newest=${newest%.*}
    [ -n "$newest" ] && [ "$newest" -gt "$last" ] && last=$newest
    if [ $(( $(date +%s) - last )) -gt "$STALL" ]; then
      echo "nothing written to $dir for over $STALL s: stopping the run"
      kill_tree "$pid"
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}
# After a hang: this job again with RESUME=1, as long as resumes are left.
again() {
  if [ "$RESUMES" -gt 0 ]; then
    echo "submitting again to resume ($RESUMES resumes left)"
    RESUME=1 RESUMES=$((RESUMES - 1)) sbatch --job-name="$SLURM_JOB_NAME" ${SLURM_MEM_PER_NODE:+--mem="${SLURM_MEM_PER_NODE}M"} "$SHADEROUS_DIR/tools/headless/slurm/dwd-sweep.sbatch"
  else
    echo "no resumes left"
  fi
}

from=""
[ "$RESUME" = 1 ] && from=$(state latest "$out")
if [ -n "$from" ]; then
  seg="$out/resume-$SLURM_JOB_ID"
  echo "resuming from $from into $seg"
  watched "$seg" --from "$from"
else
  seg="$out"
  watched "$seg" --setup "$setup"
fi
status=$?
echo "runner exit $status"
[ "$status" -eq 124 ] && { again; exit 1; }
# A setup that watches for burning (he4, c12 and o16 alone) stops with 'burn' once it is found:
# the run carries on in the same job from that checkpoint with all thirteen of the network's
# species, into f<FILL>/switched-<job>/ (dwd-mdot.mjs reads the folders below a fill's too).
stopped=$(state stopped "$seg")
if [ "$status" -eq 0 ] && [ "$stopped" = "burn" ]; then
  from=$(state last "$seg")
  echo "burning found: carrying on from $from with all thirteen species"
  watched "$out/switched-$SLURM_JOB_ID" --from "$from" --switch-burn
  status=$?
  echo "runner exit $status"
  [ "$status" -eq 124 ] && { again; exit 1; }
fi
exit $status
