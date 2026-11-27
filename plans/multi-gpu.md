# Plan: one simulation across several GPUs

Status: draft, 2026-11-26. Nothing here is built. The estimates of traffic and overhead below
are estimates, not measurements, except where a measurement is named.

## 1. Goal

Run one adaptive-mesh simulation, with self-gravity, across several GPUs, so that a run can be
larger than one GPU's memory. The target is the double white dwarf at 2048^3/5 and beyond on
Unity's non-preemptible 80 GB GPUs (section 2), which today stops at what one A100 holds
(23,024 blocks, 75.2 GB, 1.88 s per step on Helmholtz).

Speed for runs that already fit on one GPU is not the goal. Splitting a short step only adds
communication to it; the scheme pays only where steps are long (section 4).

| Stage | Result | Size |
|---|---|---|
| 0 | Ensembles: one run per GPU, as a Slurm job array | Small |
| 1 | Uniform grid, hydro only, pieces on one device, halos copied on the GPU | Medium |
| 2 | The same pieces on separate devices, halos staged through the host | Medium |
| 3 | AMR without time bins: mirrored blocks, partition, migration at regrid | Large |
| 4 | Self-gravity: distributed fine levels, gathered coarse levels, multipole sums | Large |
| 5 | Time bins, then MHD, radiation and species, one at a time | Large |
| 6 | Runtime with one device per GPU (Dawn in Node, or one Chromium per GPU) | Medium |
| 7 | Several nodes: processes joined by sockets | Medium |

Stages 1 to 5 can be built and tested on one GPU (section 6). Only stages 6 and 7, and every
performance number, need a machine with several.

### Non-goals

- Speeding up runs that fit on one GPU.
- Mixed GPU models in one run (shared faces would no longer be bit-identical; section 5).
- The interactive page. Multi-GPU is for the headless runner; the page keeps one device.
- Ray-marched volume rendering across devices in the first version (section 5, rendering).

## 2. Hardware we can reach

On Unity (checked with `sinfo` and `scontrol`, 2026-11-26), for account `pi_rfisher1_umassd_edu`:

| Partition | GPU | Memory | Preemptible |
|---|---|---|---|
| `umd-cscdr-gpu` | A100 (12, on two nodes) | 80 GB | no |
| `gpu-highvram` | A100, `--constraint=vram80` | 80 GB | no |
| `gpu-highvram` | H100 (one node, `uri-gpu009`, 4 GPUs) | 80 GB | no |
| `arm-gpu` | GH200 (4 nodes, 1 GPU each), aarch64 | ~96 GB | no |
| `gpu-preempt-highvram` | H200 NVL (`gpu057`, 8 GPUs) | 143 GB | yes (by `gpupod-h200`) |

So the usable non-preemptible ceiling is 80 GB per GPU, and up to 6 per node on
`umd-cscdr-gpu` if the node is free. A single rented H200 (141 GB) on Runpod already holds 35^3
blocks; multi-GPU is worth building only if the big runs must stay on Unity, or must go past
what one H200 holds.

Before stage 3, check the cheaper path: per-binding limits (MHD's 2 GiB flux register, the
4 GiB `maxBufferSize`, the 31^3-block ceiling with self-gravity) cap a single GPU before its
memory does. Splitting those textures and buffers into pieces may buy a bigger run sooner.

## 3. What WebGPU allows

Measured in Chrome on the RTX 3080 (Ampere), 2026-11-26:

- An adapter makes one device. A second `requestDevice` on it is refused: "adapter is
  consumed".
- A second `navigator.gpu.requestAdapter()` gives a fresh adapter for the same GPU, and from it
  a second device. One GPU can therefore host several isolated devices.
- Devices share nothing. Copying device A's buffer on device B fails validation ("is
  associated with [Device], and cannot be used with [Device]").
- A 1 MB hop from one device to another (copy to a map-read buffer, `mapAsync`, `writeBuffer`
  on the other, `onSubmittedWorkDone`) took 0.8 ms at best and 4 ms as the median of 20.

Not measured, and to check in stage 6:

- The WebGPU API cannot list GPUs or choose one; `powerPreference` is the only hint. In a
  browser a page gets one physical GPU. Dawn's native API lists adapters; whether the Node
  binding (`webgpu` on npm) exposes the choice is unknown. With one Chromium per GPU the choice
  would come from Vulkan device selection in the environment.
- Host-staged bandwidth at production sizes (hundreds of MB per step), and whether two queues
  copying at once on one node contend.

## 4. Is the traffic affordable?

Rough estimate for the double white dwarf at 2048^3/5, split four ways:

- Each piece shares a surface of perhaps 10-20% of its blocks. Ghost layers of the state,
  for a step, come to some 100-300 MB. Through the host at ~10 GB/s that is tens of ms against
  a 1.88 s step: a few percent.
- Self-gravity needs a halo swap per smoothing sweep on the distributed levels: of the order of
  60 dispatches per V-cycle, a few V-cycles a solve, so dozens of host round trips at 0.5-4 ms
  each, 30-100 ms a solve. Fine against seconds, ruinous against milliseconds.
- One global reduction per step for dt (more with time bins, section 5).

Conclusion to test: overhead of a few to ~10% at production size, and no gain for small runs.

## 5. Design

### 5.1 Pieces and transport

A *piece* is a set of blocks (a slab on the uniform grid) owned by one `AmrSim`-like driver with
its own atlas. All exchanges between pieces go through one interface with three
implementations, so stages 1, 2 and 7 differ only in the transport:

1. **Same device:** `copyTextureToTexture` between the pieces' atlases, in the same command
   encoder. No latency; for debugging the decomposition.
2. **Same process, other device:** copy to a staging buffer, `mapAsync`, `writeBuffer` on the
   other device. Double-buffered staging so a step's halos can be in flight while the interior
   runs.
3. **Other process:** the same, through a local or network socket.

The transport can add artificial delay, to test how the schedule copes with slow links.

### 5.2 Halos: mirrored blocks

Ghost cells are computed on load from the neighbour's data in the atlas (no stored ghosts).
Across pieces, each piece's atlas holds *mirror* blocks: read-only copies of the remote blocks
that touch it, refreshed before each pass that reads neighbours. The CPU-side tree (small) marks
which slots are mirrors and which piece owns them. The hydro passes run on owned blocks only and
read mirrors as ordinary neighbours, so the WGSL changes little.

Open question: whole mirror blocks are simplest but copy 16^3 cells where a few layers suffice.
Start whole, measure, then copy face slabs only if the traffic matters.

### 5.3 Shared faces and conservation

Today each face's flux is computed once ("gathered fluxes and bit-identical faces"). With
mirrors, both pieces compute the flux of a shared face from the same inputs. On identical GPUs
and drivers that is bit-identical, so conservation holds without exchange. Stage 1 checks it.
Mixed hardware would need the refinement-style flux matching across pieces; out of scope.

### 5.4 Timestep and time bins

- **Global dt:** each piece reduces its own dt on the GPU; the host takes the minimum and
  writes it back before the next step. One forced sync a step, the first break of "the CPU
  never waits". Alternative to measure: use the previous step's global dt with a safety factor
  and the current local one, whichever is smaller, so the sync is a step late. Risky near
  shocks; decide from tests.
- **Time bins:** each piece builds its block lists and indirect dispatch arguments on its own
  GPU. Pieces must agree on the bins of blocks at shared boundaries. Plan: a joint bin pass at
  the start of each cycle (bins of boundary blocks exchanged, relaxation sweeps that cross
  pieces done with a sync per sweep), then halo swaps for the blocks active in each substep.
  Time bins come last (stage 5) because of this.

### 5.5 Self-gravity

- Fine multigrid levels are split with the hydro; each smoothing sweep, residual and
  restriction refreshes mirrors at that level.
- Coarse levels (below some size, to tune) are gathered onto one device, solved there, and the
  correction sent back. Their cost is already dispatch-bound (~1.7 us per dependent dispatch,
  memory: dispatch-overhead); per-level host round trips would dwarf it.
- Multipole moments for the isolated boundary: a reduction across pieces, then the boundary
  potential evaluated on each piece. Same for point masses, sinks, the debris census and
  diagnostics: small global reductions, each one more sync.

### 5.6 AMR and load balance

- The tree stays on one host, as now. Partition the leaves along a Morton curve into pieces of
  equal cost (cells weighted by bin rate once time bins are in).
- At regrid: flags from every piece reach the host (already read back asynchronously), the
  host refines and coarsens the one tree, repartitions, and moves blocks between pieces (state,
  primitives, gravity, species, magnetic field, radiation).
- Repartition only when imbalance exceeds a threshold; a migration costs a full host hop per
  block moved.

### 5.7 Rendering and outputs

- Slices and probes: each piece renders its part; the host composites. Easy.
- Volume ray-march: gather a downsampled volume onto one device, or sort-last compositing.
  Not in the first version.
- `.athdf`: each piece writes its own blocks; the format already stores meshblocks
  independently. Checkpoints through the runner, as now (the ~8 GB ArrayBuffer limit of a
  Chromium page applies per process).

### 5.8 Runtime (stage 6)

Two options, to decide after stage 2:

1. **Dawn in Node, no browser.** One process holds a device per GPU and copies between them in
   memory. Drops Chromium and its ArrayBuffer limit. Costs: replacing Vite's
   `import.meta.glob` shader registry with a build step, and a headless path for anything that
   needs a canvas. Unknown: adapter choice in the Node binding.
2. **One Chromium per GPU, Node relay between them.** Fewest changes to how Shaderous runs.
   Halos cross a process boundary over a local socket, never through Playwright/CDP (base64,
   far too slow). One extra copy each way.

## 6. Testing on one GPU

Stages 1 to 5 need only the RTX 3080 (section 3), with the same check at every rung: a split
run matches the single-device run **bit for bit** on the same mesh. Both "GPUs" are the same
card and driver, so any difference is a decomposition bug, not rounding.

1. Pieces on one device, copies on the GPU (transport 1).
2. Pieces on separate devices on the same GPU (transport 2): async readback, two queues in step.
3. Pieces in separate processes on localhost (transport 3).

These go into `tests/gpu/` (Playwright) with small meshes: two devices share the 3080's 10 GB.
Run them on a static build (memory: gpu-tests-static-build).

What one GPU cannot test: speedup and scaling (the devices share cores and bandwidth, so a
split run is slower), concurrent PCIe traffic from several cards, adapter choice per physical
GPU, and NUMA effects. For those, a short job on `gpu-highvram` or `umd-cscdr-gpu` with 2 GPUs
on one node, after stage 6.

## 7. Risks and open questions

- **The CPU must now wait.** dt, time-bin agreement and gravity reductions each add syncs. If
  they cannot be overlapped with interior work, the step loses its "never waits" property and
  small runs get much slower. Acceptable only because small runs stay on one device.
- **Gravity latency.** If host round trips are nearer 4 ms than 0.8 ms, a V-cycle's dozens of
  swaps may cost more than estimated. Measure in stage 4 with transport 2; consider fewer,
  thicker halo swaps (several smoothing sweeps per swap with wider mirrors).
- **Time bins may not decompose well.** If the joint bin pass costs too much, multi-GPU runs
  could run without time bins (a fixed global dt), trading per-block steps for scale.
- **Maintenance.** Every new physics module would need its halo fields declared. Keep the
  list in one place so a module that forgets fails a test rather than silently diverging.
- **Is it needed?** Revisit section 2 before stage 3: splitting large single-GPU textures, or
  renting a 141-192 GB GPU, may meet the need first.
