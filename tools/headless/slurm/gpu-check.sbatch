#!/bin/bash
# Does Shaderous run on this cluster's GPUs? A short job: the driver, the Vulkan ICD, a small
# Sedov blast, then the double white dwarf at 2048^3 with 5 levels for 20 steps, to read the
# mesh's size and the time a step takes before committing to the long run.
#   sbatch tools/headless/slurm/gpu-check.sbatch
# Results: runs/slurm-check-<job>/ (manifest.json names the adapter; the PNG's status line has
# the block count) and slurm-check-<job>.out.
#SBATCH --job-name=shaderous-check
#SBATCH --partition=gpu-highvram
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-check-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"
out="$SHADEROUS_DIR/runs/slurm-check-$SLURM_JOB_ID"

echo "== node $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== Vulkan ICD files on the host"
ls -l /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || echo "none found"
echo "== Vulkan driver library"
ldconfig -p | grep -E 'libGLX_nvidia|libnvidia-glvkspirv' || echo "no NVIDIA Vulkan libraries in ldconfig"

echo "== Sedov blast, 64^3, 20 steps"
in_container node tools/headless/cli.mjs run --problem sedov --dim 3 --n 64 --steps 20 --png --out "$out/sedov"
grep -o '"adapter": *"[^"]*"' "$out/sedov/manifest.json" || echo "no adapter in the manifest: WebGPU did not start"

gb=$(gpu_memory_gb)
# Seconds per step from a run's diag.jsonl: the wall time between its first and last outputs.
per_step() {
  in_container node -e "const l=require('fs').readFileSync(process.argv[1],'utf8').trim().split('\n').map(JSON.parse);const a=l[0],b=l.at(-1);console.log(((b.wall-a.wall)/(b.steps-a.steps)).toFixed(3)+' s/step over '+(b.steps-a.steps)+' steps, t = '+b.t)" "$1/diag.jsonl"
}
# The same load with the Helmholtz equation of state (the run's) and the plain degenerate gas:
# the difference is what Helmholtz costs on this GPU.
for eos in helmholtz degenerate; do
  echo "== Double white dwarf, 2048^3 with 5 levels, $eos, --gpu-memory $gb, 100 steps"
  in_container node tools/headless/cli.mjs run --setup examples/setups/double-wd-random-field.shaderous.json \
    --n 2048 --amr 5 --set eos=$eos --gpu-memory "$gb" --steps 100 --png --overlays \
    --view mode=slice --view sliceAxis=2 --view zoom=1.5 --view showBlocks=true \
    --out "$out/dwd-$eos"
  per_step "$out/dwd-$eos"
done
echo "== done: $out (the PNGs' status line gives the block count)"
