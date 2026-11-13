"""Train the burning surrogate: a small network that steps the alpha chain's mass fractions
forward by dt at fixed density, with the self-heating that burner.py integrates.

    tools/burn/.venv/Scripts/python tools/burn/train.py [steps] [width] [depth] [hard] [out.pt] [rel] [fresh] [arch] [aux] [init.pt|-] [unroll] [lr] [logw] [resw]

Input  log10 rho, log10 T, log10 dt (all scaled to about [-1, 1]) and the 13 mass fractions
       (as X^(1/4), which spreads out the trace species).
Output the change in composition over the step (Burner), from a target composition and a
       rate: the new fractions sum to one whatever the weights are. The energy released is computed from
       the change in composition and the nuclear masses, never predicted, so energy and
       composition always agree.

Pairs come from the trajectories (gen_data.py): any two of a trajectory's times, the earlier
as the input and the later as the target, so every step length is represented.

Writes tools/burn/model.pt and prints validation errors, including a rollout check: the
model applied over and over with hydro-like steps along held-out trajectories.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).parent
DATA = HERE / "data"
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_A = 6.02214076e23
A = np.array([4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56], dtype=np.float64)
MASS = None  # set from aprox13.py below
LOG_RHO = (4.0, 10.0)
LOG_T = (8.0, 10.5)
LOG_DT = (-12.0, 0.0)


def nuclear_q():
    """Energy per gram released per unit mass fraction moved into each species, relative to
    C12, in units of 1e18 erg/g."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("aprox13", HERE / "aprox13.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    q = -N_A * net.mass / net.A  # erg/g per unit X (rest-mass energy, negated)
    return (q - q[1]) / 1e18


Q = torch.tensor(nuclear_q(), dtype=torch.float32, device=dev)


def scale(v, lo_hi):
    lo, hi = lo_hi
    return 2.0 * (v - lo) / (hi - lo) - 1.0


# Reactions whose rates the network sees as inputs ('rates' models): the fusions that start
# each stage of the alpha chain. Their rates rise as exp(-tau / T^(1/3)), far more steeply
# than a small network resolves from log T to the few percent an ignition delay needs.
RATE_INPUTS = [
    # function in aprox13.py, reactant indices (he4 0, c12 1, o16 2, ne20 3, si28 5)
    ("C12_C12_to_He4_Ne20_reaclib", (1, 1)),
    ("O16_C12_to_He4_Mg24_reaclib", (2, 1)),
    ("O16_O16_to_He4_Si28_reaclib", (2, 2)),
    ("He4_He4_He4_to_C12_reaclib", (0, 0, 0)),
    ("C12_He4_to_O16_reaclib", (1, 0)),
    ("O16_He4_to_Ne20_reaclib", (2, 0)),
    ("Ne20_He4_to_Mg24_reaclib", (3, 0)),
    ("Si28_He4_to_S32_reaclib", (5, 0)),
]


def reaclib_sets(name):
    """The REACLIB fit sets of one rate, as rows (a0..a6) of
    ln rate = a0 + a1/T9 + a2 T9^(-1/3) + a3 T9^(1/3) + a4 T9 + a5 T9^(5/3) + a6 ln T9,
    read out of the generated network."""
    import re
    src = (HERE / "aprox13.py").read_text()
    body = src[src.index(f"def {name}("):]
    body = body[:body.index("rate_eval.")]
    terms = {"T9i": 1, "T913i": 2, "T913": 3, "T9": 4, "T953": 5, "lnT9": 6}
    sets = []
    for m in re.finditer(r"ln_set_rate =(.*?)\n\s*\n", body, re.S):
        expr = m.group(1).replace("\\", " ")
        a = [0.0] * 7
        for coef, term in re.findall(r"([-+]?\s*[\d.eE+-]+)\*tf\.(\w+)", expr):
            a[terms[term]] += float(coef.replace(" ", "").replace("+-", "-").replace("--", "+"))
        a[0] = float(re.match(r"\s*([-+]?[\d.eE+-]+)", expr).group(1))
        sets.append(a)
    return sets


RATE_SETS = [torch.tensor(reaclib_sets(f), dtype=torch.float32, device=dev) for f, _ in RATE_INPUTS]


def rate_features(log_rho, log_T, X):
    """log10 of each reaction's term in dY/dt, rho^(n-1) prod Y_i lambda(T) (unscreened, Y =
    X / A), clipped and scaled to about [-1, 1]. A few exponentials per reaction."""
    T9 = torch.pow(10.0, log_T - 9.0).clamp(0.01, 10.0)
    basis = torch.stack([torch.ones_like(T9), 1 / T9, T9 ** (-1 / 3), T9 ** (1 / 3), T9, T9 ** (5 / 3), torch.log(T9)], dim=1)
    Y = X.clamp_min(0.0) / torch.tensor(A, dtype=torch.float32, device=dev)
    out = []
    for (_, react), sets in zip(RATE_INPUTS, RATE_SETS):
        ln_lam = torch.logsumexp(basis @ sets.T, dim=1)    # sum of the sets' rates
        ln_y = sum(torch.log(Y[:, i] + 1e-30) for i in react)
        log10 = (ln_lam + ln_y) / math.log(10) + (len(react) - 1) * log_rho
        out.append(((log10.clamp(-30.0, 10.0) + 10.0) / 20.0)[:, None])
    return torch.cat(out, dim=1)


# The full set ('full' models): every term of aprox13's dY/dt, each reaction forward and
# backward, screened as the reference burner screens them (screen5). A term is
# (coefficient, abundance powers {species: power}, power of rho, rate); a rate is a REACLIB
# rate with its screening pairs, or ("approx", "fwd" or "rev", nucleus, nucleus + alpha,
# intermediate): aprox13's (a,g) + (a,p)(p,g) combinations up the chain from Mg24.
ZS = np.array([2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=np.float64)
SCREEN_PAIRS = {
    "He4_C12": (2, 4, 6, 12), "He4_O16": (2, 4, 8, 16), "He4_Ne20": (2, 4, 10, 20),
    "C12_C12": (6, 12, 6, 12), "C12_O16": (6, 12, 8, 16), "O16_O16": (8, 16, 8, 16),
    "C12_Ne20": (6, 12, 10, 20), "He4_Mg24": (2, 4, 12, 24), "He4_Si28": (2, 4, 14, 28),
    "He4_He4": (2, 4, 2, 4), "He4_Be8": (2, 4, 4, 8), "He4_S32": (2, 4, 16, 32),
    "He4_Ar36": (2, 4, 18, 36), "He4_Ca40": (2, 4, 20, 40), "He4_Ti44": (2, 4, 22, 44),
    "He4_Cr48": (2, 4, 24, 48), "He4_Fe52": (2, 4, 26, 52),
    "p_Al27": (1, 1, 13, 27), "p_P31": (1, 1, 15, 31), "p_Cl35": (1, 1, 17, 35),
    "p_K39": (1, 1, 19, 39), "p_Sc43": (1, 1, 21, 43), "p_V47": (1, 1, 23, 47),
    "p_Mn51": (1, 1, 25, 51), "p_Co55": (1, 1, 27, 55),
}
_CHAIN = [("Mg24", "Si28", "Al27", 4), ("Si28", "S32", "P31", 5), ("S32", "Ar36", "Cl35", 6),
          ("Ar36", "Ca40", "K39", 7), ("Ca40", "Ti44", "Sc43", 8), ("Ti44", "Cr48", "V47", 9),
          ("Cr48", "Fe52", "Mn51", 10), ("Fe52", "Ni56", "Co55", 11)]
FULL_TERMS = [
    (1.0, {0: 1, 1: 1}, 1, ("C12_He4_to_O16_reaclib", ["He4_C12"])),
    (1.0, {2: 1}, 0, ("O16_to_He4_C12_reaclib", [])),
    (1.0, {0: 1, 2: 1}, 1, ("O16_He4_to_Ne20_reaclib", ["He4_O16"])),
    (1.0, {3: 1}, 0, ("Ne20_to_He4_O16_reaclib", [])),
    (1.0, {0: 1, 3: 1}, 1, ("Ne20_He4_to_Mg24_reaclib", ["He4_Ne20"])),
    (1.0, {4: 1}, 0, ("Mg24_to_He4_Ne20_reaclib", [])),
    (0.5, {1: 2}, 1, ("C12_C12_to_He4_Ne20_reaclib", ["C12_C12"])),
    (1.0, {0: 1, 3: 1}, 1, ("Ne20_He4_to_C12_C12_reaclib", ["He4_Ne20"])),
    (1.0, {1: 1, 2: 1}, 1, ("O16_C12_to_He4_Mg24_reaclib", ["C12_O16"])),
    (1.0, {0: 1, 4: 1}, 1, ("Mg24_He4_to_C12_O16_reaclib", ["He4_Mg24"])),
    (0.5, {2: 2}, 1, ("O16_O16_to_He4_Si28_reaclib", ["O16_O16"])),
    (1.0, {0: 1, 5: 1}, 1, ("Si28_He4_to_O16_O16_reaclib", ["He4_Si28"])),
    (1.0, {1: 1, 3: 1}, 1, ("Ne20_C12_to_He4_Si28_reaclib", ["C12_Ne20"])),
    (1.0, {0: 1, 5: 1}, 1, ("Si28_He4_to_C12_Ne20_reaclib", ["He4_Si28"])),
    (1 / 6, {0: 3}, 2, ("He4_He4_He4_to_C12_reaclib", ["He4_He4", "He4_Be8"])),
    (1.0, {1: 1}, 0, ("C12_to_3He4_reaclib", [])),
] + [t for a, b, mid, i in _CHAIN for t in (
    (1.0, {0: 1, i: 1}, 1, ("approx", "fwd", a, b, mid)),
    (1.0, {i + 1: 1}, 0, ("approx", "rev", a, b, mid)))]


def _screen_consts():
    from pynucastro.constants import constants as c
    co2 = np.cbrt(27 * np.pi**2 * c.q_e**4 * c.m_u_C18 / (2 * c.k * c.hbar**2)) / 3
    aa = c.q_e**2 / (np.cbrt(3 * c.m_u_C18 / (4 * np.pi)) * c.k)
    return float(co2), float(aa)


CO2, AA_FAC = _screen_consts()


def screen5(T, rho, Y, pair):
    """pynucastro's screen5 (Wallace 1982: Graboske weak, Alastuey-Itoh strong), batched:
    the log of the factor a pair's rate is multiplied by."""
    z1, a1, z2, a2 = (float(v) for v in pair)
    Zt = torch.tensor(ZS, dtype=torch.float32, device=dev)
    ytot = Y.sum(dim=1)
    zbar = (Zt * Y).sum(dim=1) / ytot
    z2bar = (Zt**2 * Y).sum(dim=1) / ytot
    rr = rho * ytot
    qlam0z = 1.88e8 / T * torch.sqrt(rr / T * (z2bar + zbar))
    taufac = CO2 / torch.pow(T, 1 / 3)
    gamp = AA_FAC / T * torch.pow(rr * zbar, 1 / 3)
    zs13 = (z1 + z2) ** (1 / 3)
    zhat = (z1 + z2) ** (5 / 3) - z1 ** (5 / 3) - z2 ** (5 / 3)
    zhat2 = (z1 + z2) ** (5 / 12) - z1 ** (5 / 12) - z2 ** (5 / 12)
    lzav = (5 / 3) * math.log(z1 * z2 / (z1 + z2))
    aznut = (z1**2 * z2**2 * a1 * a2 / (a1 + a2)) ** (1 / 3)
    fact, bb = 2 ** (1 / 3), z1 * z2
    gamef = fact * bb / zs13 * gamp
    tau12 = taufac * aznut
    alph12 = gamef / tau12
    cap = alph12 > 1.6
    alph12 = torch.where(cap, torch.full_like(alph12, 1.6), alph12)
    gamef = torch.where(cap, 1.6 * tau12, gamef)
    gamp = torch.where(cap, gamef * zs13 / (fact * bb), gamp)
    h12w = bb * qlam0z
    cc = (0.896434 * gamp * zhat - 3.44740 * gamp**0.25 * zhat2
          - 0.5551 * (torch.log(gamp) + lzav) - 2.996)
    a3 = alph12**3
    ss = tau12 * (5 / 32 - alph12 * (0.014 + 0.0128 * alph12))
    vv = gamef * alph12 * (0.0055 + alph12 * (-0.0098 + 0.0048 * alph12))
    h12 = cc - a3 * (ss + vv) + torch.log(torch.clamp_min(1.0 - 0.0562 * a3, 0.77))
    gx, gs = 0.3, 0.8
    mix = (h12w * (gs - gamef) + h12 * (gamef - gx)) / (gs - gx)
    h = torch.where(gamef > gx, torch.where(gamef <= gs, mix, h12), h12w)
    return h.clamp_min(0.0)


_SETS = {}


def ln_reaclib(name, basis):
    if name not in _SETS:
        _SETS[name] = torch.tensor(reaclib_sets(name), dtype=torch.float32, device=dev)
    return torch.logsumexp(basis @ _SETS[name].T, dim=1)


def full_rate_terms(log_rho, log_T, X):
    """ln of every term of aprox13's dY/dt (FULL_TERMS), screened."""
    T = torch.pow(10.0, log_T)
    rho = torch.pow(10.0, log_rho)
    T9 = (T * 1e-9).clamp(0.01, 10.0)
    basis = torch.stack([torch.ones_like(T9), 1 / T9, T9 ** (-1 / 3), T9 ** (1 / 3), T9, T9 ** (5 / 3), torch.log(T9)], dim=1)
    Y = X.clamp_min(0.0) / torch.tensor(A, dtype=torch.float32, device=dev)
    scr = {k: screen5(T, rho, Y.clamp_min(1e-30), v) for k, v in SCREEN_PAIRS.items()}
    lnY = torch.log(Y + 1e-30)
    out = []
    for coef, powers, nrho, rate in FULL_TERMS:
        if rate[0] == "approx":
            _, way, a, b, mid = rate
            ag = ln_reaclib(f"{a}_He4_to_{b}_reaclib", basis) + scr[f"He4_{a}"]
            ap = ln_reaclib(f"{a}_He4_to_p_{mid}_reaclib", basis) + scr[f"He4_{a}"]
            pg = ln_reaclib(f"{mid}_p_to_{b}_reaclib", basis) + scr[f"p_{mid}"]
            pa = ln_reaclib(f"{mid}_p_to_He4_{a}_reaclib", basis) + scr[f"p_{mid}"]
            ga = ln_reaclib(f"{b}_to_He4_{a}_reaclib", basis)
            gp = ln_reaclib(f"{b}_to_p_{mid}_reaclib", basis)
            both = torch.logaddexp(pg, pa)
            ln_lam = torch.logaddexp(ag, ap + pg - both) if way == "fwd" else torch.logaddexp(ga, pa + gp - both)
        else:
            name, pairs = rate
            ln_lam = ln_reaclib(name, basis)
            for k in pairs:
                ln_lam = ln_lam + scr[k]
        ln = ln_lam + math.log(coef) + nrho * math.log(10) * log_rho
        for i, n in powers.items():
            ln = ln + n * lnY[:, i]
        out.append(ln[:, None])
    return torch.cat(out, dim=1)


def full_rate_features(log_rho, log_T, X):
    """full_rate_terms as log10, clipped and scaled to about [-1, 1]."""
    log10 = full_rate_terms(log_rho, log_T, X) / math.log(10)
    return (log10.clamp(-30.0, 10.0) + 10.0) / 20.0


def features(log_rho, log_T, log_dt, X, rates=False):
    if rates == "full":
        return torch.cat([
            scale(log_rho, LOG_RHO)[:, None],
            scale(log_T, LOG_T)[:, None],
            scale(log_dt, LOG_DT)[:, None],
            X.clamp_min(0.0) ** 0.25,
            (torch.log10(X.clamp_min(0.0) + 1e-30).clamp(-30.0, 0.0) + 15.0) / 15.0,
            full_rate_features(log_rho, log_T, X),
        ], dim=1)
    f = [
        scale(log_rho, LOG_RHO)[:, None],
        scale(log_T, LOG_T)[:, None],
        scale(log_dt, LOG_DT)[:, None],
        X.clamp_min(0.0) ** 0.25,
    ]
    if rates:
        f.append(rate_features(log_rho, log_T, X))
    return torch.cat(f, dim=1)


class Burner(nn.Module):
    """The step's change in composition, D = X' - X, as a relaxation toward a target:
    D = (1 - exp(-k dt)) (Y - X), with the target Y and the rate k both from the network. As
    dt goes to zero the change does too, whatever the weights, so a cell that is not burning
    keeps its composition however many steps it takes; a step much longer than the burn
    reaches Y. (Predicting X' directly leaves a floor of about 1e-4 in every step that adds up
    over thousands of them.)

    The model returns the change, not the new fractions, and computes Y - X without taking
    the difference of two nearly equal numbers: X' computed in single precision, with X
    subtracted from it, leaves rounding of 3e-8 in every step, and the energy computed from
    that (3e9 erg/g a step, always of one sign) ignites cool dense C/O that should simmer for
    seconds. The energy released is Q . D.

    The target mixes two parts. One scales each species the cell has, Y_i = X_i exp(a_i) /
    sum_j X_j exp(a_j), so that Y_i - X_i = X_i (expm1(a_i) - m) / (1 + m) with
    m = sum_j X_j expm1(a_j) stays exact however small the change. The other is a softmax over
    free logits, which makes species the cell does not have yet (fresh fuel, ahead of every
    burning front: with the first part alone a species at zero never appears).

    'rates2' models take two such steps one after the other, each with its own target and
    rate: the step's whole change shares one fraction 1 - exp(-k dt), so no species can lose
    more of itself in a step than that fraction, and a trace of fuel that burns in a
    millisecond while the rest simmers (He in C/O, captured onto C12 and O16) cannot burn at
    all. The first takes such a trace, the second the slow change after it."""

    def __init__(self, width=64, depth=3, arch="delta"):
        super().__init__()
        self.arch = arch
        self.rates = "full" if arch == "flow2f" else arch in ("rates", "rates2", "rates2t", "flow2")
        self.stages = 2 if arch in ("rates2", "rates2t", "flow2", "flow2f") else 1
        self.take = arch == "rates2t"
        self.flow = arch in ("flow2", "flow2f")
        self.width = 41 if self.take else 27 if self.flow else 28   # outputs per step
        layers, n = [], 16 + (13 + len(FULL_TERMS) if self.rates == "full" else len(RATE_INPUTS) if self.rates else 0)
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.SiLU()]
            n = width
        layers.append(nn.Linear(n, self.width * self.stages))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if self.flow:
            with torch.no_grad():
                for k in range(self.stages):
                    self.net[-1].bias[k * 27 + 1:k * 27 + 14] = -4.0   # each species loses 2% of itself

    def forward(self, log_rho, log_T, log_dt, X):
        X = X.clamp_min(0.0)
        z = self.net(features(log_rho, log_T, log_dt, X, self.rates))
        if self.stages == 1:
            return self.relax(z, X, log_dt)
        P = self.width
        D1 = self.relax(z[:, :P], X, log_dt)
        p1 = self.parts
        D2 = self.relax(z[:, P:], (X + D1).clamp_min(0.0), log_dt)
        # The aux losses train each step's rate off the dead zone: the first on the whole
        # change, the second on what the first left of it.
        self.parts = [p1 + (None,), self.parts + (D1.detach(),)]
        return D1 + D2

    def load_from(self, state, arch):
        """The weights of a model with fewer outputs ('rates', 'rates2') as the same steps of
        this one: a missing second step starts as no change at all at a small rate (target
        the cell itself, free share 1e-9), missing take logits as taking from every species
        in proportion (share sigmoid(2), near what the older models do, all of it)."""
        old = Burner(1, 1, arch)
        last = f"net.{len(self.net) - 1}"
        w, b = state.pop(last + ".weight"), state.pop(last + ".bias")
        self.load_state_dict(state, strict=False)
        P = self.width
        with torch.no_grad():
            head = self.net[-1]
            head.weight.zero_(); head.bias.zero_()
            for k in range(self.stages):
                o = k * P
                if k < old.stages:
                    head.weight[o:o + 28] = w[k * old.width:k * old.width + 28]
                    head.bias[o:o + 28] = b[k * old.width:k * old.width + 28]
                    if old.take:
                        head.weight[o + 28:o + 41] = w[k * old.width + 28:k * old.width + 41]
                        head.bias[o + 28:o + 41] = b[k * old.width + 28:k * old.width + 41]
                else:
                    head.bias[o + 13] = -1.4    # log10 k = -2
                    head.bias[o + 27] = -18.0
                if self.take and not (k < old.stages and old.take):
                    head.bias[o + 28:o + 41] = 2.0   # share 0.88: at 12 the sigmoid is flat and never trains

    def relax(self, z, X, log_dt):
        if self.flow:
            return self.flow_step(z, X, log_dt)
        em1 = torch.expm1(z[:, :13].clamp(-30.0, 30.0))
        m = (X * em1).sum(dim=1, keepdim=True)             # sum_j X_j expm1(a_j), with sum X = 1
        toward = X * (em1 - m) / (1.0 + m)                 # the scaled target minus X
        w = torch.sigmoid(z[:, 27] - 2.0)[:, None]         # share of the free target
        if self.take:
            # Mass for the free target taken from each species by a share of its own, not
            # from all of them in proportion: Ne20 made out of the O16 and the He alone, C12
            # left alone, as alpha capture does in simmering C/O.
            share = X * torch.sigmoid(z[:, 28:41])
            free = share.sum(dim=1, keepdim=True) * torch.softmax(z[:, 14:27], dim=1) - share
        else:
            free = torch.softmax(z[:, 14:27], dim=1) - X
        dY = (1.0 - w) * toward + w * free                 # Y - X
        log_k = 5.0 + 5.0 * z[:, 13]                       # log10 of the rate, 1/s
        kdt = torch.pow(10.0, (log_k + log_dt).clamp(max=6.0))
        c = -torch.expm1(-kdt)
        self.parts = (dY, c, log_k + log_dt)               # for the training losses
        return c[:, None] * dY


    def flow_step(self, z, X, log_dt):
        """'flow2' steps: each species loses a share of itself, sigmoid(u_i), and what is lost
        goes to the species by a softmax g, D = (1 - exp(-k dt)) (M g - X sigmoid(u)) with
        M = sum_i X_i sigmoid(u_i). Unlike the relaxation's target, which normalizes, a
        product here costs only the species that make it: Ne20 out of O16 and He, the C12
        left alone (the relaxation took it out of C12 and O16 alike, both its parts, and the
        He that should have made it stayed and burned again every step after)."""
        # Logits bounded smoothly to +-40 (shares down to 4e-18): unbounded, long training
        # drove them to thousands, saturated, and pure He stopped burning (the He it lost
        # handed back to He by a softmax at 1 to within float precision).
        u = 40.0 * torch.tanh(z[:, 1:27] / 40.0)
        loss = X * torch.sigmoid(u[:, :13])
        dY = loss.sum(dim=1, keepdim=True) * torch.softmax(u[:, 13:], dim=1) - loss
        log_k = 5.0 + 5.0 * z[:, 0]
        kdt = torch.pow(10.0, (log_k + log_dt).clamp(max=6.0))
        c = -torch.expm1(-kdt)
        self.parts = (dY, c, log_k + log_dt)
        return c[:, None] * dY


# The reaction channels of aprox13 (forward direction; a channel's net flux takes either
# sign, its reverse included): reactant and product counts by species index (he4 0, c12 1,
# o16 2, ne20 3, mg24 4, si28 5, s32 6, ar36 7, ca40 8, ti44 9, cr48 10, fe52 11, ni56 12).
CHANNELS = [
    {1: -1, 0: -1, 2: 1},            # C12 + He4 -> O16
    {2: -1, 0: -1, 3: 1},            # O16 + He4 -> Ne20
    {3: -1, 0: -1, 4: 1},            # Ne20 + He4 -> Mg24
    {1: -2, 0: 1, 3: 1},             # C12 + C12 -> He4 + Ne20
    {1: -1, 2: -1, 0: 1, 4: 1},      # C12 + O16 -> He4 + Mg24
    {2: -2, 0: 1, 5: 1},             # O16 + O16 -> He4 + Si28
    {3: -1, 1: -1, 0: 1, 5: 1},      # Ne20 + C12 -> He4 + Si28
    {0: -3, 1: 1},                   # 3 He4 -> C12
] + [{0: -1, i: -1, i + 1: 1} for i in range(4, 12)]   # Mg24 + He4 -> Si28 ... Fe52 + He4 -> Ni56


def stoichiometry():
    """Change in mass fraction per unit of each channel's extent (in moles per gram), 13 by
    16: A_i times the change in species i's count. Every column sums to zero (each reaction
    keeps its nucleons)."""
    M = np.zeros((13, len(CHANNELS)))
    for k, ch in enumerate(CHANNELS):
        for i, n in ch.items():
            M[i, k] = n * A[i]
    assert np.abs(M.sum(axis=0)).max() == 0
    return torch.tensor(M, dtype=torch.float32, device=dev)


STOICH = stoichiometry()
# Order the channels are applied in, up the chain (the fusions that make helium first, so
# that the captures after them can use it); a second pass goes back down (photodisintegration
# cascades from the iron group down).
ORDER = [3, 4, 5, 6, 7, 0, 1, 2] + list(range(8, 16))
PASSES = ORDER + ORDER[::-1]
REACT = [[(i, -n) for i, n in ch.items() if n < 0] for ch in CHANNELS]   # reactant (index, count)
PROD = [[(i, n) for i, n in ch.items() if n > 0] for ch in CHANNELS]
A_T32 = torch.tensor(A, dtype=torch.float32, device=dev)


class FluxBurner(Burner):
    """The step's change in composition through the network's reactions ('flux' models), so
    that a species appears only as its reaction's reactants disappear (a model free to change
    each species on its own learned to make Ne20 from a trace of He4 without using the He4
    up, and ran simmering C/O away).

    The 16 channels (CHANNELS) are applied one after another, twice (PASSES), each forward
    and then backward: forward it burns a fraction f of what its reactants allow, as the
    cell is at that point, backward a fraction of what its products allow. So no species
    goes below zero, a channel's products are there for the channels after it, and mass is
    conserved exactly, whatever the weights. f = 1 - exp(-10^(q + log10 dt)) with q the
    log10 of a rate (1/s) from the network: zero as dt goes to zero, all of it for a step
    much longer than 1 / rate, and never exactly zero, so that its gradient, which goes as f,
    reaches every channel at any size (a fraction of a signed output got stuck below zero,
    where a channel backward through products the cell does not have does nothing at all).
    The change is summed on its own, not as a difference of fractions, so changes far below
    the fractions' rounding survive."""

    def __init__(self, width=64, depth=3, arch="flux"):
        super().__init__(width, depth, "rates")
        self.arch = arch
        n = self.net[-1].in_features
        self.net[-1] = nn.Linear(n, 2 * len(PASSES))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_rho, log_T, log_dt, X):
        X = X.clamp_min(0.0)
        z = self.net(features(log_rho, log_T, log_dt, X, True))
        q = 10.0 * z - 5.0                                     # log10 rate, 1/s
        f = -torch.expm1(-torch.pow(10.0, (q + log_dt[:, None]).clamp(-38.0, 6.0)))
        D = torch.zeros_like(X)
        for j, k in enumerate(PASSES):
            for side, sign in ((REACT[k], 1.0), (PROD[k], -1.0)):
                Y = (X + D).clamp_min(0.0) / A_T32             # moles per gram now
                room = torch.stack([Y[:, i] / n for i, n in side], dim=1).amin(dim=1)
                D = D + (sign * f[:, 2 * j + (sign < 0)] * room)[:, None] * STOICH[:, k]
        self.parts = None
        return D


def load_model(name):
    ck = torch.load(HERE / name)
    arch = ck.get("arch", "delta")
    model = (FluxBurner if arch == "flux" else Burner)(ck["width"], ck.get("depth", 3), arch).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def load(names=None):
    """Every trajectory file in data/: fuel and mixture starts (gen_data.py) and the states
    earlier models wandered into (dagger.py). Fractions in double precision: their changes
    over a short step are far below single precision's rounding of the fractions themselves."""
    files = sorted(f for f in DATA.glob("*.npz") if f.name != "fuel_pool.npz" and (names is None or f.name in names))
    parts = [np.load(f) for f in files]
    rho = np.concatenate([p["rho"] for p in parts])
    X = np.concatenate([p["X"] for p in parts]).astype(np.float64)
    T = np.concatenate([p["T"] for p in parts])
    t = parts[0]["t"]
    keep = np.isfinite(T).all(axis=1) & np.isfinite(X).all(axis=(1, 2))
    return rho[keep], X[keep], T[keep], t


Q64 = Q.double()


class Pairs:
    def __init__(self, rho, X, T, t):
        self.log_rho = torch.tensor(np.log10(rho), dtype=torch.float32, device=dev)
        self.X = torch.tensor(X, dtype=torch.float64, device=dev)
        self.log_T = torch.tensor(np.log10(np.clip(T, 1e7, 2e10)), dtype=torch.float32, device=dev)
        self.t = torch.tensor(t, dtype=torch.float64, device=dev)
        self.n, self.k = X.shape[0], X.shape[1]

    def sample(self, b, fresh=0.0):
        """b pairs: inputs, and the reference's change in composition and energy released
        (1e18 erg/g), both taken in double precision. A share `fresh` of them start at a
        trajectory's first state (fresh fuel, or the mixture as it was made), which is 1 time
        in 74 otherwise."""
        tr = torch.randint(0, self.n, (b,), device=dev)
        i = torch.randint(0, self.k - 1, (b,), device=dev)
        i = torch.where(torch.rand(b, device=dev) < fresh, torch.zeros_like(i), i)
        j = i + 1 + (torch.rand(b, device=dev) * (self.k - 1 - i).float()).long()
        dt = (self.t[j] - self.t[i]).float()
        D = self.X[tr, j] - self.X[tr, i]
        inputs = (self.log_rho[tr], self.log_T[tr, i], torch.log10(dt), self.X[tr, i].float())
        return inputs, D.float(), (D * Q64).sum(dim=1).float()


    def sample_unrolled(self, b, fresh=0.0):
        """As sample(), with the temperature at the start and the density as well, for an
        unrolled stretch of steps."""
        tr = torch.randint(0, self.n, (b,), device=dev)
        i = torch.randint(0, self.k - 1, (b,), device=dev)
        i = torch.where(torch.rand(b, device=dev) < fresh, torch.zeros_like(i), i)
        j = i + 1 + (torch.rand(b, device=dev) * (self.k - 1 - i).float()).long()
        D = self.X[tr, j] - self.X[tr, i]
        return self.log_rho[tr], self.log_T[tr, i], (self.t[j] - self.t[i]).float(), self.X[tr, i].float(), D.float(), (D * Q64).sum(dim=1).float()


def unrolled(model, log_rho, log_T, span, X0, n):
    """The model applied n times with steps of span / n from X0 at temperature 10^log_T, the
    temperature following the energy each step releases, as in the simulator: the total
    change, differentiable throughout, so that training on it teaches the model what its
    errors do to the steps after them (an ignition delay is set by the closed loop of
    burning and heating, which single steps never show)."""
    rho = torch.pow(10.0, log_rho)
    T = torch.pow(10.0, log_T)
    log_dt = torch.log10(span / n)
    total = torch.zeros_like(X0)
    for _ in range(n):
        # The composition goes in without its gradient: the inputs' X^(1/4) and log Y have
        # infinite slope at a species' zero and turn the backward pass to NaN. The loop that
        # sets an ignition delay runs through the temperature, which keeps its gradient.
        D = model(log_rho, torch.log10(T), log_dt, (X0 + total).detach())
        total = total + D
        T = heat(D, X0 + total, rho, T)
    return total


REL_EPS = 1e-13   # in mass fraction, and in 1e18 erg/g (1e5 erg/g)
LOG_EPS = 1e-10   # below this, changes count as nothing (log_loss)


def log_loss(Dp, Dt, et):
    """Errors in each species' change and in the energy released on a log-like scale
    (asinh(x / LOG_EPS)), symmetric: a change a factor too small costs what a change the same
    factor too large does. The relative errors of loss_fn are not: doing nothing where the
    reference burns costs at most log 2 there, burning where it does not without bound, and
    over many steps, where an ignition comes a little early or late, a model trained on them
    learns never to burn at all."""
    a = lambda x: torch.asinh(x / LOG_EPS)
    l_x = (a(Dp) - a(Dt)).abs().mean(dim=1)
    l_e = (a((Dp * Q).sum(dim=1)) - a(et)).abs()
    return (l_x + l_e).mean() / 20.0   # asinh(1 / LOG_EPS) is about 24


def reservoir_loss(Dp, Dt, X0):
    """Each species' error relative to how much of it there is (before or after the step,
    whichever is more): what matters for the steps after this one. A step that leaves half
    a trace of fuel in place errs by little of the composition, but that fuel burns again
    in every step after it (He in simmering C/O: 1.6e-3 of He left, Ne20 made instead out of
    C12, and the gas runs away in 0.1 s where it should simmer for seconds)."""
    size = torch.maximum(X0, X0 + Dt).clamp_min(0.0) + 1e-6   # species below this barely matter
    return torch.log1p((Dp - Dt).abs() / size).sum(dim=1).mean()


def loss_fn(Dp, Dt, et, hard=0.0, rel=0.0, parts=None, aux=0.0):
    """Errors in the change of composition (L1) and in the energy released (1e18 erg/g),
    absolute, and if `rel`, relative to the reference's: before a runaway the energy released
    per step is tiny, invisible to the absolute errors, and sets when the gas ignites."""
    l_x = (Dp - Dt).abs().sum(dim=1)
    l_e = ((Dp * Q).sum(dim=1) - et).abs()
    # Pairs over which the composition changes a lot (ignition, the march to the iron group)
    # are few and the hardest; `hard` weights them up.
    d_t = Dt.abs().sum(dim=1)
    w = 1.0 + hard * d_t
    loss = (w * (l_x + l_e)).mean() / w.mean()
    if rel > 0:
        # Saturating smoothly: clamped, a step that does nothing where the reference burns
        # gets no gradient at all.
        r_x = l_x / (d_t + REL_EPS)
        r_e = l_e / (et.abs() + REL_EPS)
        loss = loss + rel * (torch.log1p(r_x) + torch.log1p(r_e)).mean()
    if parts is not None and aux > 0:
        # The two halves of the step on their own, so that neither can hide behind the
        # other: c = 1 - exp(-k dt) near zero leaves the target untrained and itself with no
        # gradient (k dt of 1e-21 is a dead end the first models fell into on fresh fuel).
        # c_ref is how far along (Y - X) the reference moved: the target is trained by that
        # much (relative to the change), and log c is pulled toward log c_ref, which has no
        # dead zone.
        if isinstance(parts, list):
            loss = loss + sum(aux_loss(Dt if done is None else Dt - done, *p, aux) for *p, done in parts)
        else:
            loss = loss + aux_loss(Dt, *parts, aux)
    return loss, l_x, l_e


def aux_loss(Dt, dY, c, log_kdt, aux):
    """loss_fn's aux terms for one relaxation step toward Dt."""
    d_t = Dt.abs().sum(dim=1)
    g = dY.detach()
    c_ref = ((Dt * g).sum(dim=1) / (g**2).sum(dim=1).clamp_min(1e-30)).clamp(1e-14, 1.0)
    l_y = (c_ref[:, None] * dY - Dt).abs().sum(dim=1) / (d_t + REL_EPS)
    log_c = torch.where(c > 1e-6, torch.log10(c.clamp_min(1e-30)), log_kdt)   # 1 - exp(-x) = x for small x
    l_c = torch.where(d_t > 1e-13, (log_c - torch.log10(c_ref)).abs().clamp(max=30.0), torch.zeros_like(c))
    return aux * (torch.log1p(l_y).mean() + 0.1 * l_c.mean())


@torch.no_grad()
def evaluate(model, pairs, n=200_000):
    model.eval()
    (r, T, dt, X0), Dt, et = pairs.sample(n)
    _, l_x, l_e = loss_fn(model(r, T, dt, X0), Dt, et)
    model.train()
    q = torch.tensor([0.5, 0.99, 0.999], device=dev)
    return {"X_l1": torch.quantile(l_x, q).tolist(), "e_err": torch.quantile(l_e, q).tolist(), "X_l1_max": l_x.max().item(), "e_err_max": l_e.max().item()}


KB, MU, ARAD = 1.380649e-16, 1.66053906660e-24, 7.5657e-15
A_T = torch.tensor(A, dtype=torch.float32, device=dev)
Z_T = torch.tensor([2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=torch.float32, device=dev)


def heat(D, X1, rho, T):
    """Temperature after a step that changed the composition by D (to X1) at density rho, the
    energy it released going into heat: the T1 at which the gas holds that much more thermal
    energy, c_ion (T1 - T) + a (T1^4 - T^4) / rho = de, with the ions' specific heat of X1, as
    the reference integrates c_v dT and the simulator adds the release to the internal energy.
    A single explicit step, T + de / c_v(T), overshoots badly when an ignition releases a lot
    at once: radiation's c_v grows as T^3, and evaluated at the start the gas heats far past
    where the reference does (8e9 K for 5e9), photodisintegrates to helium, and stays there.
    Newton's method from above (the left side is convex in T1), so it never undershoots."""
    # In units of 1e9 K and 1e18 erg/g: T^4 in K overflows single precision above 4e9 K.
    de = (D * Q).sum(dim=1)
    c_ion = 1.5 * KB / MU * (X1 / A_T * (Z_T + 1)).sum(dim=1) * 1e-9
    b = ARAD * 1e18 / rho
    t0 = T * 1e-9
    t1 = torch.where(de > 0, t0 + de / c_ion, t0)
    for _ in range(40):
        f = c_ion * (t1 - t0) + b * (t1**4 - t0**4) - de
        t1 = t1 - f / (c_ion + 4 * b * t1**3)
    return (t1 * 1e9).clamp(1e7, 3e10)


@torch.no_grad()
def roll(model, rho, X0, T0, dt, steps, record=0):
    """Apply the model `steps` times with step dt (per row), the temperature following the
    energy released, as the simulator would: fractions kept in single precision, the energy
    from each step's change. Returns the final (X, T) and, if record > 0, that many states
    visited along the way (rho, T, X) at random steps."""
    X = torch.as_tensor(X0, dtype=torch.float32, device=dev).clone()
    T = torch.as_tensor(T0, dtype=torch.float32, device=dev).clone()
    r = torch.as_tensor(rho, dtype=torch.float32, device=dev)
    lr, ld = torch.log10(r), torch.log10(torch.as_tensor(dt, dtype=torch.float32, device=dev))
    seen = []
    at = set(np.random.default_rng(2).choice(steps, size=min(record, steps), replace=False).tolist()) if record else set()
    for s in range(steps):
        D = model(lr, torch.log10(T), ld, X)
        X = (X + D).clamp_min(0.0)
        T = heat(D, X, r, T)
        if s in at:
            seen.append((r.clone(), T.clone(), X.clone()))
    return X, T, seen


@torch.no_grad()
def rollout(model, rho, X, T, t, dt_hydro):
    """Apply the model with steps of dt_hydro from each trajectory's start to t = 0.1 s
    (composing it many times, the temperature following the energy released), and compare
    with the reference there."""
    model.eval()
    k_end = int(np.searchsorted(t, 0.1))
    steps = int(round(t[k_end] / dt_hydro))
    Xc, _, _ = roll(model, rho, X[:, 0], T[:, 0], np.full(len(rho), dt_hydro), steps)
    X0 = torch.tensor(X[:, 0], dtype=torch.float64, device=dev)
    Dt = torch.tensor(X[:, k_end], dtype=torch.float64, device=dev) - X0
    _, l_x, l_e = loss_fn((Xc.double() - X0).float(), Dt.float(), (Dt * Q64).sum(dim=1).float())
    model.train()
    return {"steps": steps, "X_l1_p50": torch.quantile(l_x, 0.5).item(), "X_l1_p99": torch.quantile(l_x, 0.99).item(), "X_l1_max": l_x.max().item(), "e_err_p99": torch.quantile(l_e, 0.99).item(), "e_err_max": l_e.max().item()}

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    hard = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    rel = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    fresh = float(sys.argv[7]) if len(sys.argv) > 7 else 0.2
    arch = sys.argv[8] if len(sys.argv) > 8 else "delta"
    aux = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
    init = sys.argv[10] if len(sys.argv) > 10 and sys.argv[10] != "-" else None
    unroll = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
    lr = float(sys.argv[12]) if len(sys.argv) > 12 else 2e-3   # lower to fine-tune a trained model
    logw = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0
    resw = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0
    tag = f"w{width}d{depth}h{hard:g}"
    rho, X, T, t = load()
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(rho))
    nval = len(rho) // 20
    va, tr = idx[:nval], idx[nval:]
    train, val = Pairs(rho[tr], X[tr], T[tr], t), Pairs(rho[va], X[va], T[va], t)
    print(f"{len(tr)} training and {nval} validation trajectories, {X.shape[1]} times each, on {dev}")

    model = (FluxBurner if arch == "flux" else Burner)(width, depth, arch).to(dev)
    if init:
        ck = torch.load(HERE / init)
        if arch != ck["arch"]:
            model.load_from(ck["state"], ck["arch"])
        else:
            model.load_state_dict(ck["state"])
        print("starting from", init)
    print(sum(p.numel() for p in model.parameters()), "parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-20 if arch == "flux" else 1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for s in range(steps):
        (r, Tl, dt, X0), Dt, et = train.sample(8192, fresh)
        Dp = model(r, Tl, dt, X0)
        loss, _, _ = loss_fn(Dp, Dt, et, hard, rel, model.parts, aux)
        if arch == "flux":
            # The channels' fractions start at zero, where their gradient is about 1e-15:
            # errors on a log scale reach them however small the change.
            loss = loss + aux * log_loss(Dp, Dt, et)
        elif logw > 0:
            # Each species' change on a log scale: the absolute errors barely see a step that
            # makes the right amount of a product out of the wrong reactant (Ne20 out of C/O
            # with the trace He that should have made it left in place, 1.6e-3 of He in a
            # step that changes 1.5e-2), and that He then burns again every step after.
            loss = loss + logw * log_loss(Dp, Dt, et)
        if resw > 0:
            loss = loss + resw * reservoir_loss(Dp, Dt, X0)
        if unroll > 0:
            # A stretch of n steps (2 to 32) between two times of a trajectory, judged where
            # it ends: absolute and relative errors as for single steps.
            n = int(2 ** torch.randint(1, 6, ()).item())
            lr_, lT, span, U0, UD, ue = train.sample_unrolled(max(256, 8192 // n), fresh)
            l_u = log_loss(unrolled(model, lr_, lT, span, U0, n), UD, ue)
            loss = loss + unroll * l_u
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # A batch whose loss or gradient is not finite is skipped, not allowed to poison the
        # weights (one such step turned a whole run to NaN).
        if all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            # Clipped: gradients through a chain of 32 steps run 40 times those of one.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
        sched.step()
        if (s + 1) % min(5000, max(500, steps // 10)) == 0 or s == 0:
            print(f"step {s + 1}: loss {loss.item():.4g}  val {evaluate(model, val)}  {time.time() - t0:.0f} s", flush=True)
    out = sys.argv[5] if len(sys.argv) > 5 else f"model_{tag}.pt"
    torch.save({"state": model.state_dict(), "width": width, "depth": depth, "arch": arch}, HERE / out)
    for dt in (1e-5, 1e-3, 1e-2):
        print(f"rollout dt={dt:g}:", rollout(model, rho[va], X[va], T[va], t, dt), flush=True)


if __name__ == "__main__":
    main()
