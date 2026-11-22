#!/bin/bash
# The unit tests and the GPU suite (tests/gpu, Playwright) on one of the cluster's GPUs: the
# checkout is built to a static page, served on a port of its own, and the specs run against it
# with Playwright's Chromium on Vulkan (playwright.config.ts in this folder).
#   SHADEROUS_DIR=~/shaderous-x sbatch tools/headless/slurm/tests.sbatch
#   PW_GREP=stored-primitives sbatch ...           (only the specs whose titles match)
#   PW_GREP_INVERT='cut-[ts][ue][rc]' sbatch ...   (all but those; this leaves out the seven long
#                                                   cut-cell tests, up to 13 hours between them)
# Results: runs/tests-<job>/ (unit.log, gpu.log, device.txt) and slurm-tests-<job>.out.
#SBATCH --job-name=shaderous-tests
#SBATCH --partition=umd-cscdr-gpu
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-tests-%j.out
set -uo pipefail
source "${SHADEROUS_DIR:-$HOME/shaderous}/tools/headless/slurm/common.sh"
out="$SHADEROUS_DIR/runs/tests-$SLURM_JOB_ID"
mkdir -p "$out"
{
  echo "node: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "checkout: $(git -C "$SHADEROUS_DIR" log --oneline -1)"
} | tee "$out/device.txt"

echo "== unit tests"
in_container node node_modules/vitest/vitest.mjs run > "$out/unit.log" 2>&1
echo "unit tests: exit $?"
grep -E "Test Files|Tests " "$out/unit.log"

echo "== static build"
in_container node node_modules/vite/bin/vite.js build --outDir "$out/build" --emptyOutDir > "$out/build.log" 2>&1
if ! grep -q "built in" "$out/build.log" || grep -q "Build failed" "$out/build.log"; then
  echo "the build failed: $out/build.log"
  exit 1
fi
port=$((5200 + SLURM_JOB_ID % 500))
in_container node node_modules/vite/bin/vite.js preview --outDir "$out/build" --port "$port" --strictPort > "$out/preview.log" 2>&1 &
server=$!
for _ in $(seq 1 60); do curl -s -o /dev/null "http://localhost:$port/" && break; sleep 1; done

echo "== GPU suite on port $port${PW_GREP:+, -g $PW_GREP}${PW_GREP_INVERT:+, --grep-invert $PW_GREP_INVERT}"
grep_args=()
[ -n "${PW_GREP:-}" ] && grep_args+=(-g "$PW_GREP")
[ -n "${PW_GREP_INVERT:-}" ] && grep_args+=(--grep-invert "$PW_GREP_INVERT")
in_container env PW_PORT="$port" PW_GLOBAL_TIMEOUT="${PW_GLOBAL_TIMEOUT:-2100}" \
  node node_modules/@playwright/test/cli.js test --config tools/headless/slurm/playwright.config.ts "${grep_args[@]}" > "$out/gpu.log" 2>&1
echo "GPU suite: exit $?"
grep -E "^\s+[0-9]+ (passed|failed|flaky|skipped|did not run)" "$out/gpu.log"
kill "$server" 2>/dev/null
echo "== done: $out"
