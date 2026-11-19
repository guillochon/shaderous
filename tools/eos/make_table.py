"""The tabulated electron-positron equation of state that Shaderous ships (src/eos/helmholtz.ts),
in the manner of Timmes & Swesty's Helmholtz table: the dimensionless thermal free energy h of
the electron-positron gas (electron_gas.py) on a grid in log10(rho Ye) and log10 T, with the
pair gas at zero charge split out as a Chebyshev series in log T.

One rgba32float texture, rows of W texels:
  rows 0 .. NR-1          per node (h, h_x, h_y, h_xy), x along log rho Ye and y along log T,
                          derivatives per grid step: h_x and h_y the exact first derivatives,
                          h_xy by differences; bicubic Hermite interpolation between nodes
  rows NR .. NR+GR-1      Newton's starting point from the energy: log10 T solving
                          e_th(rho, T, Abar) = e for Abar = 4, 14, 56 (r, g, b), on a grid in
                          log10 rho and log10 e_th (erg/g)
  rows NR+GR .. NR+2GR-1  the same from the pressure, against log10(p_th / rho)

e_th and p_th are everything above the zero-temperature electrons: the electron-positron gas's
thermal part, ideal ions (Abar nucleons each) and radiation; Ye = 1/2 for the guesses.

    tools/burn/.venv/Scripts/python tools/eos/make_table.py   (needs data/eos_grid.npz, gen_data.py 0.02)

Writes src/eos/helmholtz_table.bin, src/eos/helmholtz_meta.ts (the layout, for the float64
mirror) and src/eos/helmholtz_consts.wgsl (the same, for the shaders).
"""
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as C

import electron_gas as eg
from table import basis, basis1, comb


def hermite_basis(t):
    return np.array(basis(t))


def hermite_d1(t):
    return np.array(basis1(t))

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "src" / "eos"
L = math.log(10)
PAIR_LO, PAIR_HI, PAIR_DEG = 8.5, 11.0, 16
G_RHO = (0.3, 11.3, 0.1)       # log10 rho
G_X = (10.0, 31.0, 0.1)        # log10 e_th, or log10 (p_th / rho)
ABARS = (4.0, 14.0, 56.0)


class Table:
    def __init__(self):
        g = np.load(HERE / "data" / "eos_grid.npz")["data"]
        self.lr = np.unique(np.round(g[:, 0], 6))
        self.lt = np.unique(np.round(g[:, 1], 6))
        self.d = self.lr[1] - self.lr[0]
        nr, nt = len(self.lr), len(self.lt)
        H = g[:, 2].reshape(nr, nt)
        Hx = g[:, 3].reshape(nr, nt) * self.d * L
        Hy = g[:, 4].reshape(nr, nt) * self.d * L
        Hxy = np.gradient(Hx, axis=1)
        self.tab = np.stack([H, Hx, Hy, Hxy], -1)
        T = 10 ** np.linspace(PAIR_LO, PAIR_HI, 801)
        beta = eg.K * T / eg.MEC2
        P = np.array([eg.pair(t)[0] for t in T])
        P0 = 4 * eg.K * T * (2 * np.pi * eg.ME * eg.K * T / eg.H**2) ** 1.5 * np.exp(-1 / beta)
        self.mid, self.half = (PAIR_LO + PAIR_HI) / 2, (PAIR_HI - PAIR_LO) / 2
        self.q = C.chebfit((np.log10(T) - self.mid) / self.half, np.log(P / P0), PAIR_DEG)
        self.dq = C.chebder(self.q)

    def lookup(self, x, y):
        """h, dh/dln n, dh/dln T at log10 rho Ye = x, log10 T = y (arrays)."""
        nr, nt = self.tab.shape[:2]
        X = np.clip((x - self.lr[0]) / self.d, 0, nr - 1.0001)
        Y = np.clip((y - self.lt[0]) / self.d, 0, nt - 1.0001)
        i, j = X.astype(int), Y.astype(int)
        u, v = X - i, Y - j
        out = []
        for fa, fb in ((hermite_basis, hermite_basis), (hermite_d1, hermite_basis), (hermite_basis, hermite_d1)):
            nodes = [self.tab[i + a, j + b] for a in (0, 1) for b in (0, 1)]
            out.append(comb(nodes, fa(u), fb(v)))
        s = 1.0 / (self.d * L)
        return out[0], out[1] * s, out[2] * s

    def pairs(self, y):
        """P_p and dln P_p / dln T at log10 T = y."""
        T = 10.0 ** y
        beta = eg.K * T / eg.MEC2
        x = np.clip((y - self.mid) / self.half, -1, 1)
        lnP = np.log(4 * eg.K * T) + 1.5 * np.log(2 * np.pi * eg.ME * eg.K * T / eg.H**2) - 1 / beta + C.chebval(x, self.q)
        return np.exp(lnP), 2.5 + 1 / beta + C.chebval(x, self.dq) / (self.half * L)

    def thermal(self, rho, y, abar):
        """(e_th per gram, p_th) at density rho, log10 T = y, Abar; Ye = 1/2."""
        n = rho * 0.5 / eg.MU
        T = 10.0 ** y
        h, hn, ht = self.lookup(np.log10(rho * 0.5), y)
        Pp, L1 = self.pairs(y)
        s = n * eg.K * T
        ni = rho / (abar * eg.MU)
        e = (Pp * (L1 - 1) - s * ht + 1.5 * ni * eg.K * T + eg.ARAD * T**4) / rho
        p = Pp + s * hn + ni * eg.K * T + eg.ARAD * T**4 / 3
        return e, p


def guesses(t):
    """log10 T for each (log rho, log x) and Abar, from the energy and from the pressure, by
    bisection on the table (both rise monotonically with T)."""
    r0, r1, dr = G_RHO
    x0, x1, dx = G_X
    lr = np.arange(r0, r1 + 1e-9, dr)
    lx = np.arange(x0, x1 + 1e-9, dx)
    R, X = np.meshgrid(lr, lx, indexing="ij")
    rho = 10.0 ** R
    out = np.zeros(R.shape + (2, 4))
    for k, abar in enumerate(ABARS):
        for w in (0, 1):
            lo = np.full(R.shape, 4.0)
            hi = np.full(R.shape, 11.0)
            target = 10.0 ** X if w == 0 else rho * 10.0 ** X
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                e, p = t.thermal(rho, mid, abar)
                above = (e if w == 0 else p) > target
                hi = np.where(above, mid, hi)
                lo = np.where(above, lo, mid)
            out[:, :, w, k] = 0.5 * (lo + hi)
    return lr, lx, out


def main():
    t = Table()
    nr, nt = t.tab.shape[:2]
    lr, lx, gs = guesses(t)
    gr, gx = len(lr), len(lx)
    W = max(nt, gx)
    tex = np.zeros((nr + 2 * gr, W, 4), dtype="<f4")
    tex[:nr, :nt] = t.tab
    tex[nr:nr + gr, :gx] = gs[:, :, 0]
    tex[nr + gr:, :gx] = gs[:, :, 1]
    tex.tofile(SRC / "helmholtz_table.bin")
    pad = lambda c: [float(x) for x in np.pad(c, (0, len(t.q) - len(c)))]
    meta = {
        "width": W, "rows": nr + 2 * gr,
        "table": {"r0": float(t.lr[0]), "t0": float(t.lt[0]), "d": float(t.d), "nr": nr, "nt": nt},
        "guess": {"rho0": G_RHO[0], "drho": G_RHO[2], "n_rho": gr, "x0": G_X[0], "dx": G_X[2], "n_x": gx,
                  "abar": list(ABARS), "row_e": nr, "row_p": nr + gr},
        "pair": {"mid": t.mid, "half": t.half, "q": [float(c) for c in t.q], "dq": pad(t.dq), "d2q": pad(C.chebder(t.dq))},
        "consts": {"k": eg.K, "mu": eg.MU, "mec2": eg.MEC2, "arad": eg.ARAD, "me": eg.ME, "h": eg.H},
    }
    (SRC / "helmholtz_meta.ts").write_text(
        "// Generated by tools/eos/make_table.py: do not edit. The layout of helmholtz_table.bin\n"
        "// (src/eos/helmholtz.ts): the electron-positron gas's thermal free energy table, the\n"
        "// Newton starting points, and the zero-charge pair gas's Chebyshev series.\n"
        f"export const HELM_META = {json.dumps(meta)} as const;\n", encoding="utf-8")
    f = lambda x: repr(float(np.float32(x))) if np.isfinite(x) else "0.0"
    arr = lambda name, c: f"const {name} = array<f32, {len(c)}>({', '.join(f(v) for v in c)});"
    tb, gu, pr = meta["table"], meta["guess"], meta["pair"]
    (SRC / "helmholtz_consts.wgsl").write_text(chr(10).join([
        "// Generated by tools/eos/make_table.py: do not edit. The layout of the table bound as",
        "// EOS_TAB (eos/table.wgsl) for eos/helmholtz.wgsl (helmholtz_table.bin), and the pair gas's series.",
        f"const HT_R0: f32 = {f(tb['r0'])};", f"const HT_T0: f32 = {f(tb['t0'])};", f"const HT_D: f32 = {f(tb['d'])};",
        f"const HT_NR: i32 = {tb['nr']};", f"const HT_NT: i32 = {tb['nt']};",
        f"const HG_RHO0: f32 = {f(gu['rho0'])};", f"const HG_DRHO: f32 = {f(gu['drho'])};", f"const HG_NRHO: i32 = {gu['n_rho']};",
        f"const HG_X0: f32 = {f(gu['x0'])};", f"const HG_DX: f32 = {f(gu['dx'])};", f"const HG_NX: i32 = {gu['n_x']};",
        f"const HG_ROW_E: i32 = {gu['row_e']};", f"const HG_ROW_P: i32 = {gu['row_p']};",
        f"const HP_MID: f32 = {f(pr['mid'])};", f"const HP_HALF: f32 = {f(pr['half'])};",
        arr("HP_Q", pr["q"]), arr("HP_DQ", pr["dq"]), arr("HP_D2Q", pr["d2q"]), ""]), encoding="utf-8")
    print(f"wrote helmholtz_table.bin ({tex.nbytes / 1e6:.2f} MB, {W} x {nr + 2 * gr}) and helmholtz_meta.ts")
    # How good the starting points are: from the guess grid's own nodes back through the table.
    rho = 10.0 ** 7.3
    for lt in (8.0, 9.0, 9.7):
        e, p = t.thermal(np.array([rho]), np.array([lt]), 14.0)
        print(f"  check rho 2e7 T 1e{lt}: e_th {e[0]:.3e} p_th {p[0]:.3e}")


if __name__ == "__main__":
    main()
