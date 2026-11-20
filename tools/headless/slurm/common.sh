# Shared settings for the Slurm scripts in this folder (sourced, not run). See README.md.
#
# Shaderous runs WebGPU in headless Chromium on Vulkan, inside the Playwright image under
# Apptainer. The image has Node and Playwright's own Chromium shell, not Chrome.

# The checkout (shared between login and GPU nodes, as home directories are on Unity).
SHADEROUS_DIR="${SHADEROUS_DIR:-$HOME/shaderous}"
# The container image, pulled once by setup.sh. Its version matches package-lock.json's Playwright.
PLAYWRIGHT_VERSION="${PLAYWRIGHT_VERSION:-1.65.0}"
SIF="${SIF:-$HOME/containers/playwright-v${PLAYWRIGHT_VERSION}-noble.sif}"

# Load Apptainer if the cluster has it as a module (Unity: apptainer/latest).
if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer/latest 2>/dev/null || module load apptainer 2>/dev/null || true
fi

# Run a command in the container with the GPU. --nv binds the NVIDIA driver's libraries; the
# Vulkan driver also needs its ICD file, which --nv may not bind, so bind it when the host has one.
in_container() {
  local binds=()
  for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    if [ -f "$icd" ]; then binds+=(--bind "$icd:/usr/share/vulkan/icd.d/nvidia_icd.json"); break; fi
  done
  # Extra binds for output folders outside home: SHADEROUS_BINDS="/work/pi_x,/scratch".
  [ -n "${SHADEROUS_BINDS:-}" ] && binds+=(--bind "$SHADEROUS_BINDS")
  apptainer exec --nv "${binds[@]}" \
    --env PW_CHANNEL=shell \
    --env SHADEROUS_NO_GPU_COMPOSITING=1 \
    --pwd "$SHADEROUS_DIR" \
    "$SIF" "$@"
}

# GPU memory for the simulation, in GB: the card's, less 6 GB for Chromium and the driver, at
# most 120 (a magnetized mesh stops at 35^3 blocks, about 110 GB, anyway).
gpu_memory_gb() {
  local mib
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  local gb=$(( mib / 1024 - 6 ))
  [ "$gb" -gt 120 ] && gb=120
  echo "$gb"
}

# Seconds left in this job (squeue's %L: [days-]hours:minutes:seconds, or minutes:seconds).
job_seconds_left() {
  local left d=0 h=0 m=0 s=0
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
  if [[ "$left" == *-* ]]; then d=${left%%-*}; left=${left#*-}; fi
  IFS=: read -r -a parts <<< "$left"
  case ${#parts[@]} in
    3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
    2) m=${parts[0]}; s=${parts[1]} ;;
    1) s=${parts[0]} ;;
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
