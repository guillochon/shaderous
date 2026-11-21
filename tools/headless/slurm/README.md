# Shaderous on a Slurm cluster

Scripts for running the command-line runner on a GPU cluster with Slurm and Apptainer, written
for UMass's Unity cluster (log in as `<user>@unity.rc.umass.edu`; an ACCESS CI account's user is
`<access id>_access-ci_org`). From Git Bash or Windows OpenSSH, add `-c aes256-gcm@openssh.com`
(the default cipher fails with "Corrupted MAC on input").

Shaderous needs the GPU's Vulkan driver (it runs WebGPU in headless Chromium on Vulkan), not
CUDA. The scripts run the Playwright Docker image under Apptainer with `--nv`, bind the host's
NVIDIA Vulkan ICD file into it when there is one, and set `PW_CHANNEL=shell` and
`SHADEROUS_NO_GPU_COMPOSITING=1` (needed on H100 and H200, harmless elsewhere).

| File | What it does |
| --- | --- |
| `setup.sh` | Once, on a login node: clone or update the checkout in `~/shaderous`, pull the image to `~/containers/`, `npm ci` inside it |
| `gpu-check.sbatch` | A one-hour job: the driver and Vulkan files, a small Sedov blast (its `manifest.json` should name an NVIDIA adapter), then the double white dwarf at 2048^3 with 5 levels for 100 steps with Helmholtz and with the degenerate gas: the block count (`mesh` in each `diag.jsonl` line, and printed) and seconds per step |
| `dwd.sbatch` | The long run (grid `N`, levels `AMR`), as a chain of jobs that each carry on from the newest checkpoint |
| `checkpoints.mjs` | Finds the newest whole checkpoint of a chained run, prunes old ones, says why the last job stopped |
| `common.sh` | Shared settings: paths, the container call, GPU memory, time left in the job |

```bash
bash ~/shaderous/tools/headless/slurm/setup.sh     # after cloning, or fetch the script first
cd ~/shaderous
sbatch tools/headless/slurm/gpu-check.sbatch
# Read slurm-check-<job>.out and runs/slurm-check-<job>/, then:
N=1024 EVERY=0.5 RUN_DIR=/work/pi_<group>/dwd-1024 SHADEROUS_BINDS=/work/pi_<group> sbatch tools/headless/slurm/dwd.sbatch
```

The GPU: the scripts ask for the `umd-cscdr-gpu` partition and `--constraint=vram80`. That
partition belongs to the UMass Dartmouth groups (the account `pi_rfisher1_umassd_edu` among
them): its 80 GB cards are 12 A100s (umd-cscdr-gpu001 and -002; its other nodes have 16 GB A16s),
its jobs are never preempted, and they may run 5 days, so a long run reloads its checkpoint (20
minutes at 17,000 blocks) less often. Its CPU partition, `umd-cscdr-cpu`, is for CPU-only work.
The same nodes are also in `gpu-preempt-highvram`, so other users' jobs may be on them: a job in
`umd-cscdr-gpu` preempts those, it does not wait for them.

Without access to it, pass another partition: `sbatch -p gpu-highvram --time=2-00:00:00 ...`
(A100 80 GB nodes and one H100 node; `sinfo -p gpu-highvram -o "%N %G %f"` lists them), or
`-p gpu-highvram,gpu-preempt-highvram`, which adds the H200 node and many more A100s but where a
job can be preempted (it is requeued and carries on from its newest checkpoint, in a segment of
its own). The `gpu` partition has no 80 GB cards. A chained job resubmits itself to the
partition it ran in, with its time limit. The magnetized mesh can hold at most 35^3 = 42,875 blocks (about
110 GB); the double white dwarf at 2048^3 was estimated at 25 to 30 thousand blocks, 70 to 80
GB, so an 80 GB card is close.

Measured on Unity (A100-SXM4-80GB, `--gpu-memory 74`, 2026-11-20): the double white dwarf at
2048^3 with 5 levels loads with 23,024 blocks at all 5 levels (1% of uniform, budget not full)
and peaks at 75.2 GB of GPU memory. A step takes 1.88 s with Helmholtz (dt 0.00097) and 0.58 s
with the degenerate gas (dt 0.0020): Helmholtz costs 6.6 times as much per unit of time, about
10.5 hours per orbit (19.5 code units) at the start. The budget holds roughly 28,000 blocks,
about a fifth more than at load; `mesh` in `diag.jsonl` says when it fills.

The chain: each job runs with `--wall` set to the time the job has left less `MARGIN` (30
minutes, for the last checkpoint), screenshots every `EVERY` of simulated time, or every `SHOT_EVERY` seconds of wall time without it (`NNNN-tT-vol.png` and
`-slice.png` in `RUN_DIR/seg-<job>/`), a checkpoint every `ATHDF_EVERY` screenshots, and keeps
the newest `KEEP` checkpoints (and, with `KEEP_EVERY`, the first at or after each multiple of that
much simulated time, for good). When it stops on its wall budget it submits the next job; when
the run reaches `UNTIL` it writes `RUN_DIR/DONE`. A job killed while writing a checkpoint
leaves a file its manifest does not list, which the next job ignores. Stop early by touching
`STOP` in the running segment's folder.
