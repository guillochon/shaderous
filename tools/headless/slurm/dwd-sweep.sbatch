#!/bin/bash
# The double white dwarf fill sweep (plans/dwd-second-run.md section 1): one run per Roche lobe
# fill, from tools/headless/slurm/dwd-sweep/f<FILL>.shaderous.json (box 16, 2048^3 with 5 levels,
# one orbit to t = 20). One job per fill, no chaining:
#   for f in 0.935 0.940 0.945 0.950 0.955 0.960 0.965; do
#     FILL=$f sbatch --job-name=dwd-f$f tools/headless/slurm/dwd-sweep.sbatch
#   done
# Settings (environment):
#   FILL       the fill, as the setup file names it (required)
#   SWEEP_DIR  where the runs go, each in f<FILL>/ (default the group's work space)
#   UNTIL      the simulated time to stop at (default 20, one orbit)
#   EVERY      simulated time between screenshots (default 0.5)
#   ATHDF_EVERY  an .athdf file every this many screenshots (default 5: every 2.5)
#   GPU_MEMORY GB the simulation may take (default 74)
# The views are dwd.sbatch's, framed for the box of 16 (twice its 8): dist halved, zoom doubled.
# Measure the mass transfer from the .athdf files with dwd-mdot.mjs (see its header).
# Stop a run: touch SWEEP_DIR/f<FILL>/STOP (it stops at its next output, writing an .athdf file).
#SBATCH --job-name=dwd-sweep
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --account=pi_rfisher1_umassd_edu
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-dwd-sweep-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"

: "${FILL:?set FILL, such as FILL=0.950}"
SWEEP_DIR="${SWEEP_DIR:-/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/dwd-sweep}"
UNTIL="${UNTIL:-20}" EVERY="${EVERY:-0.5}" ATHDF_EVERY="${ATHDF_EVERY:-5}" GPU_MEMORY="${GPU_MEMORY:-74}"
export SHADEROUS_BINDS="${SHADEROUS_BINDS:-/work/pi_rfisher1_umassd_edu}"
setup="tools/headless/slurm/dwd-sweep/f$FILL.shaderous.json"
out="$SWEEP_DIR/f$FILL"
[ -f "$SHADEROUS_DIR/$setup" ] || { echo "no setup $setup"; exit 1; }
if [ -e "$out/manifest.json" ] && grep -q '"stopped"' "$out/manifest.json" 2>/dev/null; then
  echo "$out has a finished run: move it away to run again"; exit 1
fi
mkdir -p "$out"
echo "fill $FILL on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), --gpu-memory $GPU_MEMORY, out $out"

in_container node tools/headless/cli.mjs run --setup "$setup" \
  --until "$UNTIL" --every "$EVERY" --athdf-every "$ATHDF_EVERY" \
  --gpu-memory "$GPU_MEMORY" --width 1024 --overlays \
  --view field=0 --view log=true --view "tf=0:0,0.15:0.15,0.4:0.25,0.7:0.3,1:0.5" \
  --shot "vol:mode=volume,showBlocks=false,fieldLines=true,fieldLineCount=400,fieldLineFloor=0.001,dist=0.35,pitch=0.45,yaw=1.2,minRho=0.05,opacity=120,box=false" \
  --shot "slice:mode=slice,sliceAxis=2,zoom=3,showBlocks=false,fieldLines=true,fieldLineCount=300,fieldLineFloor=0.00001" \
  --out "$out"
status=$?
echo "runner exit $status"
exit $status
