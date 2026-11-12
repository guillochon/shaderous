"""Train the burning surrogate: a small network that steps the alpha chain's mass fractions
forward by dt at fixed density, with the self-heating that burner.py integrates.

    tools/burn/.venv/Scripts/python tools/burn/train.py [steps] [width] [depth] [hard] [out.pt] [rel] [fresh] [arch] [aux] [init.pt|-] [unroll] [lr]

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


def features(log_rho, log_T, log_dt, X, rates=False):
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
    burning front: with the first part alone a species at zero never appears)."""

    def __init__(self, width=64, depth=3, arch="delta"):
        super().__init__()
        self.arch = arch
        self.rates = arch == "rates"
        layers, n = [], 16 + (len(RATE_INPUTS) if self.rates else 0)
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.SiLU()]
            n = width
        layers.append(nn.Linear(n, 28))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_rho, log_T, log_dt, X):
        X = X.clamp_min(0.0)
        z = self.net(features(log_rho, log_T, log_dt, X, self.rates))
        em1 = torch.expm1(z[:, :13].clamp(-30.0, 30.0))
        m = (X * em1).sum(dim=1, keepdim=True)             # sum_j X_j expm1(a_j), with sum X = 1
        toward = X * (em1 - m) / (1.0 + m)                 # the scaled target minus X
        w = torch.sigmoid(z[:, 27] - 2.0)[:, None]         # share of the free target
        free = torch.softmax(z[:, 14:27], dim=1) - X
        dY = (1.0 - w) * toward + w * free                 # Y - X
        log_k = 5.0 + 5.0 * z[:, 13]                       # log10 of the rate, 1/s
        kdt = torch.pow(10.0, (log_k + log_dt).clamp(max=6.0))
        c = -torch.expm1(-kdt)
        self.parts = (dY, c, log_k + log_dt)               # for the training losses
        return c[:, None] * dY


def load_model(name):
    ck = torch.load(HERE / name)
    model = Burner(ck["width"], ck.get("depth", 3), ck.get("arch", "delta")).to(dev)
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
        dY, c, log_kdt = parts
        g = dY.detach()
        c_ref = ((Dt * g).sum(dim=1) / (g**2).sum(dim=1).clamp_min(1e-30)).clamp(1e-14, 1.0)
        l_y = (c_ref[:, None] * dY - Dt).abs().sum(dim=1) / (d_t + REL_EPS)
        log_c = torch.where(c > 1e-6, torch.log10(c.clamp_min(1e-30)), log_kdt)   # 1 - exp(-x) = x for small x
        l_c = torch.where(d_t > 1e-13, (log_c - torch.log10(c_ref)).abs().clamp(max=30.0), torch.zeros_like(c))
        loss = loss + aux * (torch.log1p(l_y).mean() + 0.1 * l_c.mean())
    return loss, l_x, l_e


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
    tag = f"w{width}d{depth}h{hard:g}"
    rho, X, T, t = load()
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(rho))
    nval = len(rho) // 20
    va, tr = idx[:nval], idx[nval:]
    train, val = Pairs(rho[tr], X[tr], T[tr], t), Pairs(rho[va], X[va], T[va], t)
    print(f"{len(tr)} training and {nval} validation trajectories, {X.shape[1]} times each, on {dev}")

    model = Burner(width, depth, arch).to(dev)
    if init:
        model.load_state_dict(torch.load(HERE / init)["state"])
        print("starting from", init)
    print(sum(p.numel() for p in model.parameters()), "parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for s in range(steps):
        (r, Tl, dt, X0), Dt, et = train.sample(8192, fresh)
        loss, _, _ = loss_fn(model(r, Tl, dt, X0), Dt, et, hard, rel, model.parts, aux)
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
