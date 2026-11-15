"""The shipped equation-of-state table (src/eos/helmholtz_table.bin, made by make_table.py) for
the burner and its training: the same bicubic Hermite interpolation as eos/helmholtz.wgsl and
src/eos/helmholtz.ts, in numpy (thermo, for burner.py's heat capacity) and in torch (thermo_t
and heat_t, batched and differentiable, for train.py's rollouts). Ye = 1/2, as for every
nucleus of the alpha chain.

All per unit mass, cgs: e is the thermal energy per gram above the zero-temperature electrons.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_meta_src = (ROOT / "src" / "eos" / "helmholtz_meta.ts").read_text(encoding="utf-8")
META = json.loads(re.search(r"HELM_META = (\{.*\}) as const", _meta_src, re.S).group(1))
TAB = np.fromfile(ROOT / "src" / "eos" / "helmholtz_table.bin", dtype="<f4").astype(np.float64).reshape(META["rows"], META["width"], 4)
K, MU, MEC2, ARAD = (META["consts"][k] for k in ("k", "mu", "mec2", "arad"))
NQ = 2 * math.pi * META["consts"]["me"] * K / META["consts"]["h"] ** 2
L10 = math.log(10.0)
YE = 0.5
LNT_MIN, LNT_MAX = math.log(1e4), math.log(1e11)
_T, _P = META["table"], META["pair"]
_Q, _DQ, _D2Q = (np.array(_P[k]) for k in ("q", "dq", "d2q"))


def _basis(t):
    return [2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2]


def _basis1(t):
    return [6 * t**2 - 6 * t, 3 * t**2 - 4 * t + 1, -6 * t**2 + 6 * t, 3 * t**2 - 2 * t]


def _basis2(t):
    return [12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2]


def _cheb(c, x):
    b1 = b2 = 0.0 * x
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = 2 * x * b1 - b2 + c[k], b1
    return x * b1 - b2 + c[0]


def thermo(rho, lnT, ia):
    """numpy: (e, de/dln T) per gram at density rho, ln T, ions per atomic mass unit ia (1/Abar)."""
    rho, lnT, ia = np.broadcast_arrays(np.asarray(rho, float), np.asarray(lnT, float), np.asarray(ia, float))
    lt = lnT / L10
    x = np.clip((np.log10(rho * YE) - _T["r0"]) / _T["d"], 0, _T["nr"] - 1.0001)
    y = np.clip((lt - _T["t0"]) / _T["d"], 0, _T["nt"] - 1.0001)
    i, j = x.astype(int), y.astype(int)
    u, v = x - i, y - j
    nodes = [TAB[i + a, j + b] for a in (0, 1) for b in (0, 1)]

    def comb(A, B):
        s = 0.0
        for a in (0, 1):
            for b in (0, 1):
                q = nodes[a * 2 + b]
                s = s + A[2*a] * B[2*b] * q[..., 0] + A[2*a+1] * B[2*b] * q[..., 1] + A[2*a] * B[2*b+1] * q[..., 2] + A[2*a+1] * B[2*b+1] * q[..., 3]
        return s
    sc = 1.0 / (_T["d"] * L10)
    ht = comb(_basis(u), _basis1(v)) * sc
    htt = comb(_basis(u), _basis2(v)) * sc * sc
    T = np.exp(lnT)
    beta = K * T / MEC2
    px = np.clip((lt - _P["mid"]) / _P["half"], -1, 1)
    ps = 1.0 / (_P["half"] * L10)
    pp = np.exp(np.log(4 * K / rho) + lnT + 1.5 * np.log(NQ * T) - 1 / beta + _cheb(_Q, px))
    l1 = 2.5 + 1 / beta + _cheb(_DQ, px) * ps
    l2 = -1 / beta + _cheb(_D2Q, px) * ps * ps
    sm = YE * K * T / MU
    im = ia * K * T / MU
    rm = ARAD * T**4 / rho
    e = pp * (l1 - 1) - sm * ht + 1.5 * im + rm
    de = pp * (l1 * l1 - l1 + l2) - sm * (ht + htt) + 1.5 * im + 4 * rm
    return e, de


def ions(X, A):
    """Ions per atomic mass unit of mass fractions X (last axis) with nucleon numbers A."""
    return (np.asarray(X) / np.asarray(A)).sum(axis=-1)


# ---- torch ----

_tt = {}


def _torch_consts(torch, device, dtype):
    key = (str(device), dtype)
    if key not in _tt:
        _tt[key] = (torch.tensor(TAB, device=device, dtype=dtype), *(torch.tensor(c, device=device, dtype=dtype) for c in (_Q, _DQ, _D2Q)))
    return _tt[key]


def thermo_t(rho, lnT, ia):
    """torch, batched and differentiable: (e, de/dln T) per gram, as thermo."""
    import torch
    tab, Q, DQ, D2Q = _torch_consts(torch, lnT.device, lnT.dtype)
    lt = lnT / L10
    x = ((torch.log10(rho * YE) - _T["r0"]) / _T["d"]).clamp(0, _T["nr"] - 1.0001)
    y = ((lt - _T["t0"]) / _T["d"]).clamp(0, _T["nt"] - 1.0001)
    i, j = x.detach().floor().long(), y.detach().floor().long()
    u, v = x - i, y - j
    nodes = [tab[i + a, j + b] for a in (0, 1) for b in (0, 1)]

    def comb(A, B):
        s = 0.0
        for a in (0, 1):
            for b in (0, 1):
                q = nodes[a * 2 + b]
                s = s + A[2*a] * B[2*b] * q[..., 0] + A[2*a+1] * B[2*b] * q[..., 1] + A[2*a] * B[2*b+1] * q[..., 2] + A[2*a+1] * B[2*b+1] * q[..., 3]
        return s
    sc = 1.0 / (_T["d"] * L10)
    ht = comb(_basis(u), _basis1(v)) * sc
    htt = comb(_basis(u), _basis2(v)) * sc * sc
    T = torch.exp(lnT)
    beta = K * T / MEC2
    px = ((lt - _P["mid"]) / _P["half"]).clamp(-1, 1)
    ps = 1.0 / (_P["half"] * L10)
    pp = torch.exp(torch.log(4 * K / rho) + lnT + 1.5 * torch.log(NQ * T) - 1 / beta + _cheb(Q, px))
    l1 = 2.5 + 1 / beta + _cheb(DQ, px) * ps
    l2 = -1 / beta + _cheb(D2Q, px) * ps * ps
    sm = YE * K * T / MU
    im = ia * K * T / MU
    rm = ARAD * T**4 / rho
    e = pp * (l1 - 1) - sm * ht + 1.5 * im + rm
    de = pp * (l1 * l1 - l1 + l2) - sm * (ht + htt) + 1.5 * im + 4 * rm
    return e, de


def heat_t(rho, T0, ia0, ia1, de_released, iters=6):
    """torch: the temperature after a step that released de_released (erg/g) at density rho and
    changed the ions per amu from ia0 to ia1, as Shaderous has it: the energy added to the
    gas's, and T read back with the new composition, e(rho, T1, ia1) = e(rho, T0, ia0) + de.
    Newton in ln T from T0, in double precision, differentiable throughout."""
    import torch
    dt = rho.dtype
    r, ln0 = rho.double(), torch.log(T0.double().clamp(1e4, 1e11))
    goal = thermo_t(r, ln0, ia0.double())[0] + de_released.double()
    lnT = ln0
    for _ in range(iters):
        e, de = thermo_t(r, lnT, ia1.double())
        lnT = (lnT - ((e - goal) / de).clamp(-1.0, 1.0)).clamp(LNT_MIN, LNT_MAX)
    return torch.exp(lnT).to(dt)
