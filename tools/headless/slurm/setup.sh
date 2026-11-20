#!/bin/bash
# One-time setup on a login node (it needs the internet): clone or update the checkout, pull the
# Playwright image as an Apptainer file, and install the Node packages inside it.
#   bash tools/headless/slurm/setup.sh      (or, before a clone exists, curl it from GitHub)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -d "$SHADEROUS_DIR/.git" ]; then
  git -C "$SHADEROUS_DIR" pull --ff-only
else
  git clone --depth 1 https://github.com/guillochon/shaderous.git "$SHADEROUS_DIR"
fi

if [ ! -f "$SIF" ]; then
  mkdir -p "$(dirname "$SIF")"
  # Apptainer unpacks layers in its cache (home by default); a small home quota may want it elsewhere.
  apptainer pull "$SIF" "docker://mcr.microsoft.com/playwright:v${PLAYWRIGHT_VERSION}-noble"
fi

# npm ci with the container's Node, so native packages match it. No GPU needed for this.
apptainer exec --pwd "$SHADEROUS_DIR" "$SIF" npm ci --no-audit --no-fund
echo "setup done: $SHADEROUS_DIR, $SIF"
