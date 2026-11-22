#!/bin/bash
# The paper's throughput benchmark (Figure 12, paper/paper/figures/make_performance.py) on one of
# the cluster's GPUs: the whole set three times, each in a fresh page, at normal priority (a
# benchmark), as make_performance.py --run does on a desktop. Pick the GPU at submission:
#   uv run python paper/figures/make_performance.py --grids > grids.json   (on your machine; copy it over)
#   GRIDS=~/grids.json sbatch -p gpu-highvram -C a100-80g tools/headless/slurm/throughput.sbatch
#   GRIDS=~/grids.json sbatch -p gpu-highvram -C h100 tools/headless/slurm/throughput.sbatch
# Results: runs/throughput-<job>/throughput_{0,1,2}.json and device.txt (the GPU, its memory and
# driver, the browser), merged at home with
#   uv run python paper/figures/make_performance.py --merge throughput_*.json --device NAME
#SBATCH --job-name=shaderous-throughput
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-throughput-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"
out="$SHADEROUS_DIR/runs/throughput-$SLURM_JOB_ID"
mkdir -p "$out"
GRIDS="${GRIDS:?set GRIDS to the file make_performance.py --grids writes}"

{
  echo "node: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id --format=csv
  echo "browser: Playwright $PLAYWRIGHT_VERSION's Chromium headless shell ($(in_container sh -c 'ls -d /ms-playwright/chromium_headless_shell-*' 2>/dev/null))"
  echo "checkout: $(git -C "$SHADEROUS_DIR" log --oneline -1)"
} | tee "$out/device.txt"

for i in 0 1 2; do
  echo "== set $i"
  in_container node paper/paper/figures/collect.mjs throughput "$out/throughput_$i.json" "$(cat "$GRIDS")" --normal
  grep -o '"adapter":"[^"]*"' "$out/throughput_$i.json"
done
echo "== done: $out"
