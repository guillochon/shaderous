#!/bin/bash
# DAgger rounds of the burning surrogate (run_rounds.py) on one Unity A100, in the training image
# (ghcr.io/guillochon/shaderous-burn, pulled to ~/containers/shaderous-burn.sif) but from a
# checkout in the work folder, where the models, logs and DAgger data are written:
#
#   W=/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/burn-train   (git clone of the repo)
#   copy tools/burn/data/, the parent model, targets_test.npz, tpairs.npz (and stepref.npz) there
#   sbatch --export=ALL,FIRST=39,PARENT=model_dagger_36a.pt,HOURS=0,BURN_UNROLL=0.3 tools/burn/unity/rounds.sbatch
#
# HOURS: rounds start until this many hours (plus half an hour) after the job begins; 0 runs
# one round.
# BURN_UNROLL, BURN_VARIANTS, BURN_TEACHER pass through to run_rounds.py (teacher off by default
# here: the reference changed after the shipped surrogate was trained).
#SBATCH --job-name=burn-rounds
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=burn-rounds-%j.log

set -euo pipefail
W=${W:-/work/pi_rfisher1_umassd_edu/guilloch_access-ci_org/burn-train}
SIF=${SIF:-$HOME/containers/shaderous-burn.sif}
FIRST=${FIRST:?round number}
PARENT=${PARENT:?parent model}
HOURS=${HOURS:-0}
export BURN_TEACHER=${BURN_TEACHER:--} BURN_UNROLL=${BURN_UNROLL:-0} PYTHONIOENCODING=utf-8
command -v apptainer >/dev/null 2>&1 || module load apptainer/latest

# run_rounds.py starts a round while now + the last round's length < DEADLINE; scoring the parent
# comes first and takes minutes, so even one round needs half an hour of margin.
DEADLINE=$(( $(date +%s) + ${HOURS%.*} * 3600 + 1800 ))
cd "$W/tools/burn"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv --bind "$W:$W" "$SIF" bash -c "
  set -e
  [ -f aprox13.py ] && [ -f aprox13_features.py ] || python build_net.py > build_net.log
  python run_rounds.py $FIRST $DEADLINE $PARENT
"
