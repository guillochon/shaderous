"""Reading runs of the command-line runner (tools/headless) as tensors, and restricting them."""
import json
from pathlib import Path

import numpy as np
import torch

VARS = ("rho", "vel1", "vel2", "press", "r0")


def load_run(run, gamma=1.4, dtype=torch.float64):
    """A run's outputs as conserved states: (times[T], U[T, 6, n, n]). Needs slices of rho,
    vel1, vel2, press and r0 at the grid's own resolution, as .npy."""
    run = Path(run)
    man = json.loads((run / "manifest.json").read_text())
    ts, Us = [], []
    for out in man["outputs"]:
        d = {s["variable"]: torch.from_numpy(np.load(run / s["file"]).astype(np.float64)) for s in out["slices"]}
        rho, vx, vy, p, s = (d[v] for v in VARS)
        ei = p / (gamma - 1.0)
        Us.append(torch.stack([rho, rho * vx, rho * vy, ei + 0.5 * rho * (vx * vx + vy * vy), ei, rho * s]))
        ts.append(out["t"])
    return torch.tensor(ts, dtype=torch.float64), torch.stack(Us).to(dtype), man


def restrict(U, f):
    """Conservative average of f x f cells: mass, momentum, total energy and scalar mass are
    means; the internal energy is what the coarse cell's total energy leaves after its kinetic
    energy (the fine cells' unresolved motion counts as heat on the coarse grid)."""
    if f == 1:
        return U
    *lead, c, ny, nx = U.shape
    V = U.reshape(*lead, c, ny // f, f, nx // f, f).mean(dim=(-3, -1))
    ke = 0.5 * (V[..., 1, :, :] ** 2 + V[..., 2, :, :] ** 2) / V[..., 0, :, :]
    V = V.clone()
    V[..., 4, :, :] = V[..., 3, :, :] - ke
    return V
