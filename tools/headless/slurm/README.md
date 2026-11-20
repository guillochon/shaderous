# Shaderous on a Slurm cluster

Scripts for running the command-line runner on a GPU cluster with Slurm and Apptainer, written
for UMass's Unity cluster (`gpu` partition, `apptainer/latest` module). Untested there so far:
start with the check job.

Shaderous needs the GPU's Vulkan driver (it runs WebGPU in headless Chromium on Vulkan), not
CUDA. The scripts run the Playwright Docker image under Apptainer with `--nv`, bind the host's
NVIDIA Vulkan ICD file into it when there is one, and set `PW_CHANNEL=shell` and
`SHADEROUS_NO_GPU_COMPOSITING=1` (needed on H100 and H200, harmless elsewhere).

| File | What it does |
| --- | --- |
| `setup.sh` | Once, on a login node: clone or update the checkout in `~/shaderous`, pull the image to `~/containers/`, `npm ci` inside it |
| `gpu-check.sbatch` | A one-hour job: the driver and Vulkan files, a small Sedov blast (its `manifest.json` should name an NVIDIA adapter), then the double white dwarf at 2048^3 with 5 levels for 100 steps with Helmholtz and with the degenerate gas: the mesh's block count (on the PNGs) and seconds per step |
| `dwd-2048.sbatch` | The long run, as a chain of jobs that each carry on from the newest checkpoint |
| `checkpoints.mjs` | Finds the newest whole checkpoint of a chained run, prunes old ones, says why the last job stopped |
| `common.sh` | Shared settings: paths, the container call, GPU memory, time left in the job |

```bash
bash ~/shaderous/tools/headless/slurm/setup.sh     # after cloning, or fetch the script first
cd ~/shaderous
sbatch tools/headless/slurm/gpu-check.sbatch
# Read slurm-check-<job>.out and runs/slurm-check-<job>/, then:
RUN_DIR=/work/pi_<group>/dwd-2048 SHADEROUS_BINDS=/work/pi_<group> sbatch tools/headless/slurm/dwd-2048.sbatch
```

To check before the first job: the GPU constraint. The scripts ask for `--constraint=vram80` (an
80 GB card or more); `sinfo -o "%20N %G %f"` lists the cluster's real feature names. The
magnetized mesh can hold at most 35^3 = 42,875 blocks (about 110 GB); the double white dwarf
at 2048^3 was estimated at 25 to 30 thousand blocks, 70 to 80 GB, so an 80 GB card is close
and the H200 is safe. The check job's PNG gives the real count.

The chain: each job runs with `--wall` set to the time the job has left less `MARGIN` (30
minutes, for the last checkpoint), screenshots every `SHOT_EVERY` seconds (`NNNN-tT-vol.png` and
`-slice.png` in `RUN_DIR/seg-<job>/`), a checkpoint every `ATHDF_EVERY` screenshots, and keeps
the newest `KEEP` checkpoints. When it stops on its wall budget it submits the next job; when
the run reaches `UNTIL` it writes `RUN_DIR/DONE`. A job killed while writing a checkpoint
leaves a file its manifest does not list, which the next job ignores. Stop early by touching
`STOP` in the running segment's folder.
