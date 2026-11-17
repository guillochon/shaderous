# Plan: running Shaderous on NASA's GPU nodes (Pleiades / Cabeus)

Status: draft, 2026-11-17. Nothing here is built or tested yet.

## 1. Goal

A colleague with a NAS account wants to run Shaderous on NASA's supercomputers. The end
product is a Claude Code skill, `.claude/skills/shaderous-nas/SKILL.md`, next to
`shaderous-sim`, that takes a setup from a person ("run the MRI disk at 1024^3 for two orbits
on a GH200") to a submitted PBS job, watches it, and brings its outputs back. It is the NAS
counterpart of what we do by hand on Runpod (memory: shaderous-on-runpod).

| Stage | Result | Size |
|---|---|---|
| 1 | Smoke test: one `gpu_devel` job prints a hardware WebGPU adapter | Small |
| 2 | Front-end install: container image, checkout, built page in `/nobackup` | Small |
| 3 | PBS job templates for `gh200` and `mil_a100` around `tools/headless/cli.mjs` | Small |
| 4 | Restarts: `cli.mjs run --restart <file.athdf>`, outputs before the wall-time limit | Medium |
| 5 | Driving it from Claude Code: SSH connection sharing, submit / watch / fetch | Small |
| 6 | The skill itself, tested end to end by the colleague | Small |

Stage 1 decides everything else; if it fails, stop and find out why before writing more.

### Non-goals

Multi-GPU or multi-node runs (Shaderous runs on one GPU), NASA's older V100 nodes, the AWS
cloud front end, and anything that needs an interactive browser window on NAS.

## 2. What NAS offers (from the HECC knowledge base, 2026-11)

Pleiades's GPU nodes have been folded into the Cabeus cluster, which runs PBS:

| Model | Hardware | Charge | PBS server | Submit from | Queues |
|---|---|---|---|---|---|
| `gh200` | 1 H100 (GH200), aarch64, 72 cores, 460 GB | 18.34 SBU per node-hour, whole node | `pbs05a` | `cghfe01`, `cghfe02` | `gpu_devel` 2 h, `gpu_normal` 24 h, `gpu_long` 120 h, `gpu_wide` |
| `mil_a100` | 4 A100, x86_64 Milan, 128 logical cores | 37.86 SBU per node-hour, by GPUs used | `pbspl4` | `cfe01`, `cfe02` | `gpu_devel` 2 h, `gpu_normal` 24 h, `np_gpu_wide` |

- Using either needs an allocation of **GPU SBUs**; an ordinary Pleiades allocation does not
  cover them.
- `gh200` is the default target: one GPU per node matches Shaderous, and it is cheaper per GPU.
  `mil_a100` is the fallback if ARM causes trouble; request `ngpus=1` there and share the node.
- Front ends have internet; compute nodes have none. Everything that downloads (git, npm,
  container images) happens on a front end.
- Containers: Apptainer (`module load apptainer/arm-toss5` on gh200, `apptainer/x86-toss4` on
  mil_a100) and Podman.
- Login goes through the secure front ends (`sfe6`, `sfe7`, `sfe8`) and asks for an RSA SecurID
  passcode on every new connection. NAS's own `ssh_config` template carries `ControlMaster`,
  `ControlPersist` and `ControlPath` lines, commented out.

What Shaderous already gives us: the headless runner needs no network once built.
`SHADEROUS_SERVE=dist` serves `dist/` from a local port (`tools/headless/session.mjs`),
`PW_CHANNEL=shell` uses Playwright's own headless Chromium, the Linux Vulkan flags are in
`session.mjs`, and `SHADEROUS_NO_GPU_COMPOSITING=1` is what an H100 needs.

## 3. Stage 1: smoke test

Unknowns, in order of risk:

1. **Vulkan inside the container.** Chrome reaches WebGPU through Vulkan. Apptainer's `--nv`
   is documented for CUDA; whether the node's NVIDIA Vulkan ICD (`nvidia_icd.json` and
   `libGLX_nvidia.so.0`) is installed and reaches the container is unknown. Without it Chrome
   falls back to SwiftShader with no error, so the test must check the adapter name, not just
   that the run finished.
2. **aarch64.** We have never run Shaderous on ARM. The Playwright image
   (`mcr.microsoft.com/playwright:v<lockfile version>-noble`) is multi-arch and Playwright ships
   a linux-arm64 Chromium, but Dawn's Vulkan backend on Grace is untested by us.
3. **Container build on a front end.** `apptainer build playwright.sif docker://...` may be slow
   or blocked on a busy front end; Podman is the alternative.

The test job, on `gpu_devel`:

```bash
#PBS -lselect=1:model=gh200
#PBS -q gpu_devel@pbs05a
#PBS -l walltime=0:30:00
module load apptainer/arm-toss5
cd /nobackup/$USER/shaderous
apptainer exec --nv -B /nobackup /nobackup/$USER/playwright.sif sh -c '
  nvidia-smi; vulkaninfo --summary || ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d
  SHADEROUS_SERVE=dist PW_CHANNEL=shell SHADEROUS_NO_GPU_COMPOSITING=1 \
    node tools/headless/cli.mjs run "problem=sedov&dim=3&n=128" --steps 50 --out runs/smoke'
```

Pass: the run's adapter reads `nvidia` (not `swiftshader`), with a speed in line with an H100
on Runpod. If the ICD is missing, write one into the job directory and point `VK_ICD_FILENAMES`
at it:

```json
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3" } }
```

and bind the host's `libGLX_nvidia.so.0` into the container if `--nv` did not. If that fails
too, ask NAS support (support@nas.nasa.gov) whether the Vulkan part of the driver is installed
on the compute nodes.

## 4. Stage 2: front-end install

Once, on `cghfe01` (or `cfe01` for A100s), into `/nobackup/$USER`:

- `apptainer build playwright.sif docker://mcr.microsoft.com/playwright:v<version>-noble`,
  the version matching `package-lock.json`'s Playwright, as on Runpod.
- `git clone --depth 1` the public repo, then `npm ci` and `npm run build` **inside the
  container** (so native bits match the runtime), producing `dist/`.
- Updating is `git pull && npm ci && npm run build` the same way. The skill records the commit
  each run used.

A script for this, `tools/nas/install.sh`, keeps it to one command.

## 5. Stage 3: job templates

`tools/nas/job.pbs.sh`, filled in by the skill: model and queue, wall time, `--gpu-memory`
(about 90 GB of the GH200's 96; per-GPU memory on the A100 nodes still to be checked), the
setup string, `--until` / `--every` / output flags, and the output folder
`/nobackup/$USER/shaderous-runs/<name>/`. The job writes a small `status.json` (step, time,
wall time, last output) the skill can read without parsing logs. Mirror the Runpod lessons:
H100 needs `SHADEROUS_NO_GPU_COMPOSITING=1`; big 3D loads spend tens of seconds to minutes in
CPU-side setup and look hung.

## 6. Stage 4: restarts

Queues stop jobs at 24 h (`gpu_normal`) or 120 h (`gpu_long`), and PBS kills the job outright.
Unlike Runpod, the deadline is fixed, so the job must save before it:

- Add `--restart <file.athdf>` to `cli.mjs run`: load the state with the page's `loadAthdf`
  (the MCP tool `load_athdf` already does this) and continue to `--until`.
- `--athdf` outputs at `--every`, plus one final save when the job is within a margin of its
  wall time (read `PBS_WALLTIME` / the job's start time, or trap the signal PBS sends first).
- The skill chains jobs: each submits the next with `qsub -W depend=afterok:<id>`, starting
  from the newest `.athdf`.

Check first that an `.athdf` round trip restores everything a run needs (AMR blocks, MHD faces,
species, time bins); if not, that is a separate fix.

## 7. Stage 5: driving it from Claude Code

SecurID means Claude cannot log in by itself. With connection sharing it does not need to:

- In `~/.ssh/config` (NAS's template), uncomment `ControlMaster auto`, set
  `ControlPersist 12h` (or what NAS allows) and a `ControlPath`, on the `sfe` host and on the
  front ends reached through it.
- The colleague logs in once in a terminal (`ssh cghfe01`, passcode); every later `ssh`,
  `scp`, `qsub`, `qstat` from Claude rides on that connection with no prompt.
- This needs macOS, Linux or WSL: Windows' built-in OpenSSH has no `ControlMaster`.
- When the master connection has expired, the skill must say so and ask the person to log in
  again, never retry in a loop (repeated failed logins can lock a NAS account).

The loop: submit (`qsub`), watch (`qstat -x`, `status.json`, `tail` of the job log), fetch
(`scp`/`rsync` of PNGs, slices, `.athdf`; large files stay on `/nobackup` or go to Lou).

## 8. Stage 6: the skill

`SKILL.md` covers: one-time setup (SSH config, stage 2 install, the smoke test), choosing the
node type and memory for a setup, writing and submitting the job, watching and fetching,
chaining restarts, and SBU cost (node-hours × rate) stated before submitting anything long.
It points to `shaderous-sim` for building setups and reading results. Tested by the colleague
on their account; we cannot test it ourselves without NAS access.

## 9. Open questions

- Is the Vulkan driver on the compute nodes (stage 1)?
- How long may `ControlPersist` keep a connection open under NAS policy?
- Does `apptainer build` from `docker://` work on the Cabeus front ends, or is Podman needed?
- GPU memory per A100 on `mil_a100` (40 or 80 GB), for `--gpu-memory`.
- Does an `.athdf` restart reproduce the run exactly, or only to round-off?

## Sources

- [Requesting Cabeus GPU Resources](https://www.nas.nasa.gov/hecc/support/kb/requesting-cabeus-gpu-resources_646.html)
- [Enabling User-Defined Software Stack with Apptainer](https://www.nas.nasa.gov/hecc/support/kb/enabling-user-defined-software-stack-with-apptainer_723.html)
- [Setting Up SSH Passthrough](https://www.nas.nasa.gov/hecc/support/kb/setting-up-ssh-passthrough_232.html)
- [Getting Started with Podman at NAS](https://www.nas.nasa.gov/hecc/support/kb/getting-started-with-podman-at-nas_698.html)
- [Cabeus Training, part 1](https://www.nas.nasa.gov/hecc/assets/pdf/training/Cabeus_Training_part_1_031224.pdf)
- [Apptainer GPU support](https://apptainer.org/docs/user/1.5/gpu.html)
