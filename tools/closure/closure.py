"""The sub-grid closure: per-cell features of the resolved flow, the gates that switch it off at
shocks and in flow that is not sheared, and the small network between them.

Everything here has a twin in the simulator (src/closure/closure.ts for the float64 mirror,
src/hydro/closure_net.wgsl for the shader); the constants below are exported with the weights
(export.py), so the three cannot drift apart.

A cell's coefficients, C = (a, b, nu, kappa):

  a      in [0, A_MAX]: the share of the velocity jump at a face that the Riemann solver is
         not shown. The solver's dissipation of shear is proportional to that jump, so this
         removes that share of it; the rest (at least 1 - A_MAX) always remains.
  b      in [0, B_MAX]: the same for the jump of the passive scalar, which removes that share
         of its numerical diffusion. Each face state is then held to where a limited slope
         could have put it (solver.py), so no value leaves the range of its neighbors: without
         that bound the scalar rang at sharp fronts (-0.4 to 1.1 for a dye of 0 to 1). The
         scalar is passive, so this cannot disturb the flow. (The density's jump is left alone.)
  nu     an eddy viscosity, C_MAX sigmoid(.) dx^2 |S|, never negative, at most CAP dx c_s.
  kappa  an eddy diffusivity of the same form, for the specific internal energy and the scalar.

A face takes the smaller of its two cells' values of each (solver.py, hydro/flux.wgsl), so one
gated cell switches the closure off on all of its faces.
"""
import torch
from torch import nn

from solver import _shift

N_FEATURES = 10
A_MAX = 0.8
B_MAX = 0.9
# The scalar's compression (solver.py): the share of what is left to the limiter's bound that
# a face state is moved on by, wherever the gates are open. A constant, not an output.
COMPRESS = 0.5
C_MAX = 0.5
CAP = 0.1
# Gates. Compression: closed where the dilatation is this share of the velocity gradient
# (0.43 in any one-dimensional flow). Pressure jump across the cell. Activity: the shear and
# rotation across a cell, over the sound speed, below which the flow is not turbulent.
DUCROS = (0.10, 0.35)
JUMP = (0.05, 0.15)
ACTIVE = (1.0e-3, 4.0e-3)
EPS_V = 3.0e-4


def smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _diffs(q, dim):
    """(q - q_minus, q_plus - q) along dim."""
    return q - _shift(q, -1, dim), _shift(q, 1, dim) - q


def _kink(q, floor):
    """How far from monotone and straight q is across the cell, 0 (a straight line) to 1 (an
    extremum or the foot of a step): the larger over the axes of |d+ - d-| / (|d+| + |d-|)."""
    r = None
    for dim in (-1, -2):
        dm, dp = _diffs(q, dim)
        k = (dp - dm).abs() / (dp.abs() + dm.abs() + floor)
        r = k if r is None else torch.maximum(r, k)
    return r


def features(rho, vx, vy, p, s, cs, dx):
    """Features F[B, N_FEATURES, ny, nx] and what the gates and the outputs' scales need.

    All are dimensionless and unchanged by reflecting the grid or swapping its axes; all but
    the grid-frame Mach number (which the solver's own dissipation depends on) are unchanged
    by a uniform velocity added to the flow."""
    def c(q, dim):  # dx times the centered derivative
        return 0.5 * (_shift(q, 1, dim) - _shift(q, -1, dim))

    ux, uy, wx, wy = c(vx, -1), c(vx, -2), c(vy, -1), c(vy, -2)
    D = ux + wy
    W = wx - uy
    sxx, syy, szz, sxy = ux - D / 3.0, wy - D / 3.0, -D / 3.0, 0.5 * (uy + wx)
    A2 = 2.0 * (sxx * sxx + syy * syy + szz * szz + 2.0 * sxy * sxy)
    A = torch.sqrt(A2 + 1e-30)
    e2 = (EPS_V * cs) ** 2
    tot = D * D + W * W + A2 + e2
    # Kinks of the velocity (both components together), the density and the scalar.
    kv = None
    for dim in (-1, -2):
        (am, ap), (bm, bp) = _diffs(vx, dim), _diffs(vy, dim)
        num = torch.sqrt((ap - am) ** 2 + (bp - bm) ** 2 + 1e-30)
        den = torch.sqrt(ap * ap + bp * bp + 1e-30) + torch.sqrt(am * am + bm * bm + 1e-30) + 1e-4 * cs
        k = num / den
        kv = k if kv is None else torch.maximum(kv, k)
    kr = _kink(rho, 1e-5 * rho)
    # (The scalar has no scale of its own: differences below 1e-4 of it, or below 1e-6 for a dye
    # of order one, are rounding, and count as none.)
    ks = _kink(s, 1e-4 * (s.abs() + _shift(s, 1, -1).abs() + _shift(s, -1, -1).abs()) + 1e-6)
    speed = torch.sqrt(vx * vx + vy * vy + 1e-30)
    M = speed / cs
    grho = torch.sqrt(c(rho, -1) ** 2 + c(rho, -2) ** 2 + 1e-30) / rho
    gp = torch.sqrt(c(p, -1) ** 2 + c(p, -2) ** 2 + 1e-30) / p
    F = torch.stack([
        0.5 * torch.log10(A / cs + 1e-4) + 1.0,
        (W * W - A2) / (W * W + A2 + e2),
        D / torch.sqrt(tot),
        kv,
        M / (1.0 + M),
        grho / (grho + 0.05),
        kr,
        ks,
        gp / (gp + 0.05),
        A / (A + speed + 1e-4 * cs),
    ], dim=1)
    # Gates.
    ducros = D * D / tot
    jump = None
    for dim in (-1, -2):
        pm, pp = _shift(p, -1, dim), _shift(p, 1, dim)
        j = (pp - pm).abs() / torch.minimum(pp, pm)
        jump = j if jump is None else torch.maximum(jump, j)
    active = torch.sqrt(W * W + A2 + 1e-30) / cs
    gate = (1.0 - smoothstep(*DUCROS, ducros)) * (1.0 - smoothstep(*JUMP, jump)) * smoothstep(*ACTIVE, active)
    return F, gate, A


class Closure(nn.Module):
    """A pointwise network on the features. `heads` says which of (a, b, nu, kappa) it may use:
    the others are held at zero (for the ablations: 'nk' is the eddy viscosity and diffusivity
    alone)."""

    def __init__(self, hidden=24, layers=2, heads="abnk", fixed=None):
        super().__init__()
        # A fifth output, the scalar's compression c (solver.py), for models with the head 'c'.
        # `fixed` = (a, b) holds those two at constants times the gate instead of learning them.
        self.nout = 5 if "c" in heads else 4
        self.fixed = fixed
        dims = [N_FEATURES] + [hidden] * layers + [self.nout]
        self.net = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.heads = heads
        with torch.no_grad():
            # Start close to no closure at all: small a and b, almost no diffusion.
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.copy_(torch.tensor([-2.0, -2.0, -5.0, -5.0, -2.0][: self.nout]))
        self.register_buffer("mask", torch.tensor([float(h in heads) for h in "abnkc"[: self.nout]]))

    def raw(self, F):
        """Network outputs for features F[..., N_FEATURES] (the last axis)."""
        x = F
        for lin in self.net[:-1]:
            x = torch.tanh(lin(x))
        return self.net[-1](x)

    def forward(self, rho, vx, vy, p, s, cs, dx):
        F, gate, A = features(rho, vx, vy, p, s, cs, dx)
        o = torch.sigmoid(self.raw(F.movedim(1, -1))).movedim(-1, 1) * self.mask.view(1, self.nout, 1, 1)
        cap = CAP * dx * cs
        a = A_MAX * o[:, 0] * gate
        b = B_MAX * o[:, 1] * gate
        if self.fixed is not None:
            a, b = self.fixed[0] * gate, self.fixed[1] * gate
        nu = torch.minimum(C_MAX * o[:, 2] * dx * A, cap) * gate
        kap = torch.minimum(C_MAX * o[:, 3] * dx * A, cap) * gate
        out = [a, b, nu, kap]
        # The scalar's compression: this model's own output, or the constant.
        out.append(o[:, 4] * gate if self.nout == 5 else COMPRESS * gate)
        return torch.stack(out, dim=1)


class Constant(nn.Module):
    """Fixed coefficients behind the same gates: the hand-set baselines."""

    def __init__(self, a=0.0, b=0.0, cn=0.0, ck=0.0, c=None, nu=0.0):
        super().__init__()
        self.v = (a, b, cn, ck)
        self.c = c  # the scalar's compression (solver.py), or None for none
        self.nu = nu  # a physical viscosity besides, the same everywhere and not gated

    def forward(self, rho, vx, vy, p, s, cs, dx):
        F, gate, A = features(rho, vx, vy, p, s, cs, dx)
        a, b, cn, ck = self.v
        cap = CAP * dx * cs
        one = torch.ones_like(rho)
        out = [a * one * gate, b * one * gate, torch.minimum(cn * dx * A, cap) * gate + self.nu, torch.minimum(ck * dx * A, cap) * gate]
        if self.c is not None:
            out.append(self.c * one * gate)
        return torch.stack(out, dim=1)
