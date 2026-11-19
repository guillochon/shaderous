"""Tables of the ANEOS materials Shaderous ships (src/eos/aneos.ts, eos/aneos.wgsl): rock, iron
and water for planetary impacts.

Each material's published input deck (its ANEOS.INPUT: sources in plans/aneos-eos.md) is run
through pyaneos, the Python port of M-ANEOS, on one grid shared by every material: uniform in
x = log10 rho and y = log10 T. NOTENSION is added to the deck, so the stretched solid below its
reference density is tabulated as the solid in equilibrium with its vapor, and the pressure is
positive everywhere.

What is tabulated is ANEOS's own pressure and energy, not its free energy: in the ionized
fluid ANEOS's p and s are not the derivatives of its F (they differ by several percent, measured
2026-11-18), so a table of F with p and s as its slopes is inconsistent with itself. Two
functions, each with its first derivatives and cross derivative at every node, for bicubic
Hermite interpolation:

    psi(x, y) = asinh(p / P_S)                P_S = 1e9 dyn/cm^2: linear below it (vapor, the
                                              surface of a body), logarithmic above
    lam(x, y) = ln(e - E_c(rho))              the energy above the cold curve, erg/g

The cold curve E_c(rho) = F(rho, T0) + E_off is the free energy of the lowest isotherm, so the
energy above it is T0 s(rho, T0) there, positive, and rises with T. Node slopes are ANEOS's own
derivatives (by differences of a part in 1e4), limited as a monotone cubic's are (Fritsch &
Carlson 1980), so the interpolant rises along T wherever the nodes do: the solve for the
temperature (by bisection on the nodes, then inside one cell) needs that. The cross derivative
is the difference of the limited slopes.

One file per material, src/eos/aneos/<name>.bin: rgba32float, NR rows (log rho) of 2 NT + 1
texels:
  texels 0 .. NT-1       (psi, psi_x, psi_y, psi_xy), derivatives per grid step
  texels NT .. 2 NT-1    (lam, lam_x, lam_y, lam_xy)
  texel 2 NT             the cold curve: (E_c, dE_c/dx per grid step, 0, 0), erg/g

    python tools/eos/aneos_table.py <pyaneos checkout> <name>=<dir with ANEOS.INPUT> ...
    python tools/eos/aneos_table.py <pyaneos checkout> --check <name>=<dir> ...   (also pyaneos
        against the published NEW-SESAME-STD.TXT in <dir>, at its own nodes)

Writes the .bin files, src/eos/aneos/<name>.json (what the page shows of the material, and the
error report), and src/eos/aneos_consts.wgsl with src/eos/aneos_meta.ts (the grid).
"""
import json
import math
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "src" / "eos"
L = math.log(10)

# The grid, shared by all materials.
X0, X1 = -8.0, 2.0        # log10 rho, g/cm^3
Y0, Y1 = 1.7, 6.0         # log10 T, K
D = 0.025
NR = round((X1 - X0) / D) + 1
NT = round((Y1 - Y0) / D) + 1
P_S = 1e9                 # dyn/cm^2
# File units of pyaneos (GPa, MJ/kg, MJ/K/kg) to cgs.
CGS = 1e10

LABELS = {
    "pyrolite": "Pyrolite (rock: a planet's mantle)",
    "forsterite": "Forsterite (rock)",
    "fe85si15": "Iron alloy, Fe85Si15 (a planet's core)",
    "iron": "Iron",
    "water": "Water and ice",
}
SOURCES = {
    "pyrolite": "Stewart 2022, Pyrolite-ANEOS-SLVTv0.2G1, doi:10.5281/zenodo.6968539",
    "forsterite": "Stewart et al. 2019, Forsterite-ANEOS-SLVTv1.0G1, doi:10.5281/zenodo.3478631",
    "fe85si15": "Stewart 2020, Fe85Si15-ANEOS-SLVTv0.2G1, doi:10.5281/zenodo.3866550",
    "iron": "Stewart 2020, Iron-ANEOS-SLVTv0.2G1, doi:10.5281/zenodo.3866507",
    "water": "Stewart 2024, Water-ANEOS-SLVTv0.3G1, doi:10.5281/zenodo.14226693",
}


def load_material(pyaneos, deck_dir):
    """The deck with NOTENSION added, as a pyaneos Material."""
    sys.path.insert(0, str(pyaneos))
    from pyaneos.state import Material

    text = (Path(deck_dir) / "ANEOS.INPUT").read_text()
    out, done = [], False
    for line in text.split("\n"):
        if not done and line.startswith("ANEOS1 ") and "NOTENSION" not in line:
            if "LONG" not in line:
                raise SystemExit(f"{deck_dir}: the ANEOS1 record has no LONG keyword")
            line = line.rstrip() + " NOTENSION"
            done = True
        out.append(line)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ANEOS.INPUT"
        p.write_text("\n".join(out))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Material.from_input(str(p))
    if not m.no_tension:
        raise SystemExit(f"{deck_dir}: NOTENSION was not taken")
    return m


def evaluate(m, rho, T):
    """ANEOS at every (T, rho): cgs arrays (NT, ND) of p, e, F, s, and the phase flag."""
    from pyaneos.driver import evaluate_grid

    f = evaluate_grid(m, np.asarray(rho, float), np.asarray(T, float))
    return {"p": f["P"] * CGS, "e": f["U"] * CGS, "F": f["A"] * CGS, "s": f["S"] * CGS, "cs": f["cs"], "kpa": f["KPA"]}


def limit(m, f, axis):
    """Slopes m of f along an axis (per grid step), limited so the cubic through each pair of
    nodes is monotone where the nodes are: zero where the secants on either side differ in
    sign, and no more than three times the smaller of them."""
    d = np.diff(f, axis=axis)
    pad = [(0, 0)] * f.ndim
    pad[axis] = (1, 0)
    dm = np.pad(d, pad, mode="edge")      # the secant below each node
    pad[axis] = (0, 1)
    dp = np.pad(d, pad, mode="edge")      # and above
    same = dm * dp > 0
    cap = 3 * np.minimum(np.abs(dm), np.abs(dp))
    sgn = np.sign(dp)
    return np.where(same, sgn * np.clip(sgn * m, 0, cap), 0.0)


def build(m):
    """The table of one material: (NR, 2 NT + 1, 4) float64, and E_off."""
    x = X0 + D * np.arange(NR)
    y = Y0 + D * np.arange(NT)
    rho = 10.0**x
    T = 10.0**y
    f = evaluate(m, rho, T)
    eps = 1e-4
    fT = [evaluate(m, rho, T * (1 + s)) for s in (eps, -eps)]
    fR = [evaluate(m, rho * (1 + s), T) for s in (eps, -eps)]
    Ec = f["F"][0]
    dEc = (fR[0]["F"][0] - fR[1]["F"][0]) / (2 * eps)            # per ln rho
    eth = lambda g, ec: np.maximum(g["e"] - ec[None, :], 1e-30)
    # In x (per grid step): d/dx = ln 10 d/dln rho.
    psi = np.arcsinh(f["p"] / P_S)
    psi_x = (np.arcsinh(fR[0]["p"] / P_S) - np.arcsinh(fR[1]["p"] / P_S)) / (2 * eps) * L * D
    psi_y = (np.arcsinh(fT[0]["p"] / P_S) - np.arcsinh(fT[1]["p"] / P_S)) / (2 * eps) * L * D
    lam = np.log(eth(f, Ec))
    lam_x = (np.log(eth(fR[0], Ec + eps * dEc)) - np.log(eth(fR[1], Ec - eps * dEc))) / (2 * eps) * L * D
    lam_y = (np.log(eth(fT[0], Ec)) - np.log(eth(fT[1], Ec))) / (2 * eps) * L * D
    eoff = max(0.0, -Ec.min()) + 1e8
    tab = np.zeros((NR, 2 * NT + 1, 4))
    for k, (v, vx, vy) in enumerate(((psi, psi_x, psi_y), (lam, lam_x, lam_y))):
        # Arrays here are (NT, NR): axis 1 is x, axis 0 is y.
        vx = limit(vx, v, 1)
        vy = limit(vy, v, 0)
        vxy = 0.5 * (np.gradient(vx, axis=0) + np.gradient(vy, axis=1))
        c = slice(k * NT, (k + 1) * NT)
        tab[:, c, 0] = v.T
        tab[:, c, 1] = vx.T
        tab[:, c, 2] = vy.T
        tab[:, c, 3] = vxy.T
    tab[:, 2 * NT, 0] = Ec + eoff
    tab[:, 2 * NT, 1] = dEc * L * D
    return tab, eoff


def hb(t):
    return np.array([2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2])


def lookup(tab, lr, lt):
    """Pressure and energy per gram (cgs) from the table, as the float64 mirror reads it."""
    xs = np.clip((lr - X0) / D, 0, NR - 1.0001)
    ys = np.clip((lt - Y0) / D, 0, NT - 1.0001)
    i = np.floor(xs).astype(int)
    j = np.floor(ys).astype(int)
    A, B = hb(xs - i), hb(ys - j)

    def comb(off):
        s = 0
        for a in range(2):
            for b in range(2):
                c = tab[i + a, off + j + b]
                s = s + A[2 * a] * B[2 * b] * c[:, 0] + A[2 * a + 1] * B[2 * b] * c[:, 1] + A[2 * a] * B[2 * b + 1] * c[:, 2] + A[2 * a + 1] * B[2 * b + 1] * c[:, 3]
        return s

    c0, c1 = tab[i, 2 * NT], tab[i + 1, 2 * NT]
    Ec = A[0] * c0[:, 0] + A[1] * c0[:, 1] + A[2] * c1[:, 0] + A[3] * c1[:, 1]
    return P_S * np.sinh(comb(0)), Ec + np.exp(comb(NT))


def report(m, tab, eoff, rng):
    """The table against ANEOS between its nodes, by phase: relative errors of p and e."""
    n = 6000
    lr = rng.uniform(X0, X1, n)
    lt = rng.uniform(Y0, Y1, n)
    from pyaneos.driver import evaluate_grid  # noqa: F401

    # Point by point: evaluate_grid takes a product grid, so one row of densities per temperature.
    order = np.argsort(lt)
    lr, lt = lr[order], lt[order]
    p = np.empty(n)
    e = np.empty(n)
    kpa = np.empty(n)
    step = 60
    for k in range(0, n, step):
        s = slice(k, k + step)
        f = evaluate(m, 10.0 ** lr[s], 10.0 ** lt[s])
        d = np.arange(len(lr[s]))
        p[s], e[s], kpa[s] = f["p"][d, d], f["e"][d, d], f["kpa"][d, d]
    pt, et = lookup(tab, lr, lt)
    et = et - eoff
    out = {}
    names = {1: "one phase", 2: "vapor dome", 4: "solid", 5: "melting", 6: "liquid"}
    rp = np.abs(pt - p) / np.maximum(np.abs(p), 1e-12)
    re_ = np.abs(et - e) / np.maximum(np.abs(e), 1e7)
    for kk in sorted(set(kpa.astype(int))):
        sel = kpa.astype(int) == kk
        out[names.get(kk, f"kpa {kk}")] = {
            "points": int(sel.sum()),
            "p_median": float(np.median(rp[sel])), "p_p90": float(np.percentile(rp[sel], 90)), "p_p99": float(np.percentile(rp[sel], 99)),
            "e_median": float(np.median(re_[sel])), "e_p90": float(np.percentile(re_[sel], 90)), "e_p99": float(np.percentile(re_[sel], 99)),
        }
    return out


def check_published(m_plain, deck_dir):
    """pyaneos against the published table at a sample of its own nodes (no NOTENSION)."""
    txt = (Path(deck_dir) / "NEW-SESAME-STD.TXT").read_text().split("\n")
    k = next(i for i, l in enumerate(txt) if "RECORD" in l and "301" in l)
    body = "".join(txt[k + 1:])
    vals = np.array([float(body[i:i + 16]) for i in range(0, len(body) - 15, 16) if body[i:i + 16].strip()])
    nd, nt = int(vals[0]), int(vals[1])
    rho, T = vals[2:2 + nd], vals[2 + nd:2 + nd + nt]
    P = vals[2 + nd + nt:2 + nd + nt + nd * nt].reshape(nt, nd)
    ri, ti = np.arange(0, nd, 9), np.arange(0, nt, 9)
    from pyaneos.driver import evaluate_grid

    f = evaluate_grid(m_plain, rho[ri], T[ti])
    ref = P[np.ix_(ti, ri)]
    rel = np.abs(f["P"] - ref) / np.maximum(np.abs(ref), 1e-30)
    return {"nodes": int(rel.size), "p_median": float(np.median(rel)), "p_max": float(rel.max())}


def write_shared():
    meta = {"x0": X0, "y0": Y0, "d": D, "nr": NR, "nt": NT, "ps": P_S}
    (SRC / "aneos_meta.ts").write_text(
        "// Generated by tools/eos/aneos_table.py: do not edit. The grid every ANEOS material's table\n"
        "// is on (src/eos/aneos.ts): log10 of g/cm^3 and of kelvin, and the pressure scale of psi, dyn/cm^2.\n"
        f"export const ANEOS_META = {json.dumps(meta)} as const;\n"
    )
    (SRC / "aneos_consts.wgsl").write_text(
        "// Generated by tools/eos/aneos_table.py: do not edit. The grid of the ANEOS tables\n"
        "// (eos/aneos.wgsl, src/eos/aneos.ts).\n"
        f"const AN_X0: f32 = {X0!r};\nconst AN_Y0: f32 = {Y0!r};\nconst AN_D: f32 = {D!r};\n"
        f"const AN_NR: i32 = {NR};\nconst AN_NT: i32 = {NT};\nconst AN_PS: f32 = {P_S!r};\n"
    )


def main():
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    pyaneos, mats = Path(args[0]), args[1:]
    (SRC / "aneos").mkdir(exist_ok=True)
    write_shared()
    rng = np.random.default_rng(1)
    for spec in mats:
        name, deck = spec.split("=", 1)
        m = load_material(pyaneos, deck)
        tab, eoff = build(m)
        if not np.all(np.isfinite(tab)):
            raise SystemExit(f"{name}: {np.count_nonzero(~np.isfinite(tab))} values of the table are not finite")
        tab.astype("<f4").tofile(SRC / "aneos" / f"{name}.bin")
        version = re.search(r"'([^']+)'", next(l for l in (Path(deck) / "ANEOS.INPUT").read_text().split("\n") if l.startswith("ANEOS1 "))).group(1)
        info = {
            "name": name, "label": LABELS.get(name, name), "source": SOURCES.get(name, ""), "deck": version,
            "eOff": eoff, "errors": report(m, tab, eoff, rng),
        }
        if check:
            sys.path.insert(0, str(pyaneos))
            from pyaneos.state import Material

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info["published"] = check_published(Material.from_input(str(Path(deck) / "ANEOS.INPUT")), deck)
        (SRC / "aneos" / f"{name}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(name, json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
