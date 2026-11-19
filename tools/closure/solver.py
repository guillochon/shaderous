"""A differentiable mirror of Shaderous's uniform-grid hydro step, for training the closure.

The step of src/hydro/*.wgsl (and its float64 twin, src/reference/hydro.ts) written with
PyTorch tensors: the same primitives, timestep, MUSCL-Hancock predictor with the limited
third-order reconstruction (hydro/limiters/koren.wgsl), positivity fallback, shock-flagged
HLLC/HLL blend, dual-energy update and floors. What it covers is what the closure is trained
on: a 2D periodic box of ideal gas without gravity. Every branch of the shaders is a
torch.where here, so a rollout can be differentiated end to end.

The closure enters where it does in the shaders (hydro/closure_*.wgsl, hydro/flux.wgsl):

  jump scaling   the velocity jump (zv) and the passive scalar's jump (zc) handed to the
                 Riemann solver are scaled per face, 1 leaving the flux as it is;
  diffusion      an eddy viscosity nu and diffusivity kappa per face, as the fluxes of
                 hydro/diffusion_core.wgsl, the diffusivity acting on the specific internal
                 energy and the passive scalar.

State: U[B, 6, ny, nx] = (rho, rho vx, rho vy, E, rho e, rho s); x is the last axis.
"""
from dataclasses import dataclass

import torch

RHO, MX, MY, EN, EI, RS = range(6)


@dataclass
class Config:
    gamma: float = 1.4
    cfl: float = 0.8
    rho_floor: float = 1e-10
    p_floor: float = 1e-14
    eta_dual: float = 1e-3
    shock_thresh: float = 0.5
    smooth: float = 0.01  # hydro/limiters/koren.wgsl SMOOTH
    box: float = 1.0
    riemann: str = "hllc"  # or "hllclm"
    z_min: float = 0.3  # hydro/riemann/hllclm.wgsl Z_MIN


def from_prim(rho, vx, vy, p, s, gamma=1.4):
    """Conserved state from primitives (each [..., ny, nx])."""
    ei = p / (gamma - 1.0)
    return torch.stack([rho, rho * vx, rho * vy, ei + 0.5 * rho * (vx * vx + vy * vy), ei, rho * s], dim=-3)


def primitives(U, cfg):
    """(rho, vx, vy, p, s, cs) as hydro/prim.wgsl gives them."""
    rho = U[:, RHO].clamp_min(cfg.rho_floor)
    vx = U[:, MX] / rho
    vy = U[:, MY] / rho
    p = (cfg.gamma - 1.0) * U[:, EI]
    p = torch.where(p >= cfg.p_floor, p, torch.full_like(p, cfg.p_floor))
    cs = torch.sqrt(cfg.gamma * p / rho)
    s = U[:, RS] / rho
    return rho, vx, vy, p, s, cs


def timestep(U, cfg, remaining=None):
    """The step hydro/rates.wgsl and hydro/clock.wgsl take, for each state of the batch on its
    own: dt[B]. `remaining` (a number or [B]) is the time left to the next output, which the
    step is shortened to land on."""
    rho, vx, vy, p, s, cs = primitives(U.detach(), cfg)
    dx = cfg.box / U.shape[-1]
    rate = (vx.abs() + vy.abs() + 2.0 * cs).amax(dim=(-1, -2))
    dt = cfg.cfl * dx / rate
    if remaining is not None:
        dt = torch.minimum(dt, torch.as_tensor(remaining, dtype=U.dtype).clamp_min(0.0).expand_as(dt))
    return dt


def _edge(far, near, scale, smooth):
    """hydro/limiters/koren.wgsl edge4: twice the increment from the cell to a face."""
    free = (far + 2.0 * near) / 3.0
    m = torch.minimum(torch.minimum(2.0 * far.abs(), 2.0 * near.abs()), free.abs())
    lim = torch.where(far * near > 0, torch.sign(near) * m, torch.zeros_like(m))
    sc = smooth * scale
    eta = (far * far + near * near) / (sc * sc).clamp_min(1e-30)
    w = (1.5 - eta).clamp(0.0, 1.0)
    return lim + (free - lim) * w


def _shift(a, k, dim):
    """The value k cells along +dim (periodic)."""
    return torch.roll(a, -k, dim)


def _hllc(l, r, w_hll, cfg):
    """hydro/riemann/hllc.wgsl. l, r: dicts rho, u (normal), v (transverse), p, s, cs, eint."""
    if cfg.riemann == "hllclm":
        z = torch.maximum(torch.sqrt(l["u"] ** 2 + l["v"] ** 2) / l["cs"], torch.sqrt(r["u"] ** 2 + r["v"] ** 2) / r["cs"])
        z = z.clamp(cfg.z_min, 1.0)
        l, r = dict(l), dict(r)
        for k in ("u", "v"):
            mean, half = 0.5 * (l[k] + r[k]), 0.5 * z * (l[k] - r[k])
            l[k], r[k] = mean + half, mean - half
    sl = torch.minimum(l["u"] - l["cs"], r["u"] - r["cs"])
    sr = torch.maximum(l["u"] + l["cs"], r["u"] + r["cs"])
    el = l["eint"] + 0.5 * l["rho"] * (l["u"] ** 2 + l["v"] ** 2)
    er = r["eint"] + 0.5 * r["rho"] * (r["u"] ** 2 + r["v"] ** 2)
    mlu, mlv = l["rho"] * l["u"], l["rho"] * l["v"]
    mru, mrv = r["rho"] * r["u"], r["rho"] * r["v"]
    fpl_u, fpl_v = mlu * l["u"] + l["p"], mlv * l["u"]
    fpr_u, fpr_v = mru * r["u"] + r["p"], mrv * r["u"]
    fel, fer = (el + l["p"]) * l["u"], (er + r["p"]) * r["u"]

    left, right = sl >= 0, sr <= 0
    mid = ~(left | right)
    one = torch.ones_like(sl)
    ql = l["rho"] * (sl - l["u"])
    qr = r["rho"] * (sr - r["u"])
    sm = (r["p"] - l["p"] + l["u"] * ql - r["u"] * qr) / torch.where(mid, ql - qr, one)
    inv = 1.0 / torch.where(mid, sr - sl, one)
    hm = (sr * mlu - sl * mru + sl * sr * (r["rho"] - l["rho"])) * inv
    hpu = (sr * fpl_u - sl * fpr_u + sl * sr * (mru - mlu)) * inv
    hpv = (sr * fpl_v - sl * fpr_v + sl * sr * (mrv - mlv)) * inv
    he = (sr * fel - sl * fer + sl * sr * (er - el)) * inv

    pos = sm >= 0
    # The star state on the side the contact leaves behind.
    q = torch.where(pos, ql, qr)
    s_ = torch.where(pos, sl, sr)
    rho = torch.where(pos, l["rho"], r["rho"])
    u = torch.where(pos, l["u"], r["u"])
    v = torch.where(pos, l["v"], r["v"])
    p = torch.where(pos, l["p"], r["p"])
    e = torch.where(pos, el, er)
    fm = torch.where(pos, mlu, mru)
    fpu = torch.where(pos, fpl_u, fpr_u)
    fpv = torch.where(pos, fpl_v, fpr_v)
    fe = torch.where(pos, fel, fer)
    f = q / torch.where(mid, s_ - sm, one)
    es = f * (e / rho + (sm - u) * (sm + p / torch.where(mid, q, one)))
    cm = fm + s_ * (f - rho)
    cpu = fpu + s_ * (f * sm - rho * u)
    cpv = fpv + s_ * (f * v - rho * v)
    ce = fe + s_ * (es - e)

    def pick(a, b, c):
        return torch.where(left, a, torch.where(right, b, c))

    m = pick(mlu, mru, cm + (hm - cm) * w_hll)
    mu = pick(fpl_u, fpr_u, cpu + (hpu - cpu) * w_hll)
    mv = pick(fpl_v, fpr_v, cpv + (hpv - cpv) * w_hll)
    en = pick(fel, fer, ce + (he - ce) * w_hll)
    vn = pick(l["u"], r["u"], sm)
    up = m >= 0
    ei = m * torch.where(up, l["eint"] / l["rho"], r["eint"] / r["rho"])
    sc = m * torch.where(up, l["s"], r["s"])
    return m, mu, mv, en, ei, sc, vn


def step(U, cfg, dt, closure=None):
    """One step of length dt. `closure`, if given, maps the primitives to per-cell coefficients
    C[B, 4, ny, nx] = (a, b, nu, kappa), gates applied (closure.py); a face takes the smaller of
    its two cells'. Returns the new state."""
    n = U.shape[-1]
    dx = cfg.box / n
    g = cfg.gamma
    dt = torch.as_tensor(dt, dtype=U.dtype)
    if dt.ndim == 1:
        dt = dt.view(-1, 1, 1)
    rho, vx, vy, p, s, cs = primitives(U, cfg)
    Q = torch.stack([rho, vx, vy, p, s], dim=1)  # [B, 5, ny, nx]
    scale = torch.stack([rho, cs, cs, p, torch.zeros_like(p)], dim=1)
    C = closure(rho, vx, vy, p, s, cs, dx) if closure is not None else None

    # Edge increments toward the upper face (up) and from the lower face (dn), per axis.
    up, dn = {}, {}
    for dim in (-1, -2):
        m, pl = _shift(Q, -1, dim), _shift(Q, 1, dim)
        up[dim] = _edge(Q - m, pl - Q, scale, cfg.smooth)
        dn[dim] = _edge(pl - Q, Q - m, scale, cfg.smooth)
    sx = 0.5 * (up[-1] + dn[-1])
    sy = 0.5 * (up[-2] + dn[-2])
    big_r = torch.stack([up[-1][:, 0].abs(), dn[-1][:, 0].abs(), up[-2][:, 0].abs(), dn[-2][:, 0].abs()]).amax(0)
    big_p = torch.stack([up[-1][:, 3].abs(), dn[-1][:, 3].abs(), up[-2][:, 3].abs(), dn[-2][:, 3].abs()]).amax(0)

    # Shock detector (hydro/predictor.wgsl).
    conv = (_shift(vx, 1, -1) - _shift(vx, -1, -1)) + (_shift(vy, 1, -2) - _shift(vy, -1, -2))
    jx = (_shift(p, 1, -1) - _shift(p, -1, -1)).abs() / torch.minimum(_shift(p, 1, -1), _shift(p, -1, -1))
    jy = (_shift(p, 1, -2) - _shift(p, -1, -2)).abs() / torch.minimum(_shift(p, 1, -2), _shift(p, -1, -2))
    shock = ((conv < 0) & (torch.maximum(jx, jy) > cfg.shock_thresh)).to(U.dtype)

    # Predictor: half a step in primitive form.
    h = 0.5 * dt / dx
    dv = sx[:, 1] + sy[:, 2]
    adv = vx.unsqueeze(1) * sx + vy.unsqueeze(1) * sy  # [B, 5, ...]
    rho_h = rho - h * (adv[:, 0] + rho * dv)
    vx_h = vx - h * (adv[:, 1] + sx[:, 3] / rho)
    vy_h = vy - h * (adv[:, 2] + sy[:, 3] / rho)
    p_h = p - h * (adv[:, 3] + rho * cs * cs * dv)
    s_h = s - h * adv[:, 4]
    ok = (rho_h - 0.5 * big_r > 0) & (p_h - 0.5 * big_p > 0)
    P = torch.where(ok.unsqueeze(1), torch.stack([rho_h, vx_h, vy_h, p_h, s_h], dim=1), Q)
    keep = ok.to(U.dtype).unsqueeze(1)  # 1 - first-order flag

    net = torch.zeros_like(U)
    net_vn = torch.zeros_like(rho)
    heat = torch.zeros_like(rho)
    vel = (vx, vy)
    for dim, nrm, trn in ((-1, 1, 2), (-2, 2, 1)):
        # The face between each cell (l) and the next along dim (r).
        Pr, keep_r = _shift(P, 1, dim), _shift(keep, 1, dim)
        L = P + 0.5 * keep * up[dim]
        R = Pr - 0.5 * keep_r * _shift(dn[dim], 1, dim)
        lo_r = 0.5 * torch.minimum(P[:, 0], Pr[:, 0])
        lo_p = 0.5 * torch.minimum(P[:, 3], Pr[:, 3])
        w_hll = torch.maximum(shock, _shift(shock, 1, dim))

        def side(S):
            r_ = torch.maximum(S[:, 0], lo_r.clamp_min(cfg.rho_floor))
            p_ = torch.maximum(S[:, 3], lo_p.clamp_min(cfg.p_floor))
            return {"rho": r_, "u": S[:, nrm], "v": S[:, trn], "p": p_, "s": S[:, 4]}

        l, r = side(L), side(R)
        if C is not None:
            Cf = torch.minimum(C, _shift(C, 1, dim))
            zv = 1.0 - Cf[:, 0]
            zc = 1.0 - Cf[:, 1]
            for k, z in (("u", zv), ("v", zv)):
                mean, half = 0.5 * (l[k] + r[k]), 0.5 * z * (l[k] - r[k])
                l[k], r[k] = mean + half, mean - half
            # The scalar's jump likewise, but each face state stays where a limited slope
            # could have put it (or where the reconstruction did): between its own cell's
            # value and no further toward the other cell than the cell's value lies from its
            # own upwind neighbor's. Without this bound the scalar rang at sharp fronts.
            mean, half = 0.5 * (l["s"] + r["s"]), 0.5 * zc * (l["s"] - r["s"])
            sc, scr = P[:, 4], Pr[:, 4]
            s0, s1 = s, _shift(s, 1, dim)
            for sd, new, own, a_, b_ in ((l, mean + half, sc, s0 - _shift(s, -1, dim), s1 - s0), (r, mean - half, scr, s1 - _shift(s, 2, dim), s0 - s1)):
                m_ = torch.where(a_ * b_ > 0, torch.sign(b_) * torch.minimum(a_.abs(), b_.abs()), torch.zeros_like(a_))
                d0 = sd["s"] - own
                lo_, hi_ = torch.minimum(torch.minimum(m_, d0), torch.zeros_like(m_)), torch.maximum(torch.maximum(m_, d0), torch.zeros_like(m_))
                sd["s"] = own + torch.maximum(torch.minimum(new - own, hi_), lo_)
        for sd in (l, r):
            sd["cs"] = torch.sqrt(g * sd["p"] / sd["rho"])
            sd["eint"] = sd["p"] / (g - 1.0)
        m, mu, mv, en, ei, sc, vn = _hllc(l, r, w_hll, cfg)
        F = [None] * 6
        F[RHO], F[EN], F[EI], F[RS] = m, en, ei, sc
        F[MX if nrm == 1 else MY] = mu
        F[MY if nrm == 1 else MX] = mv

        if C is not None:
            # Eddy viscosity and diffusivity (hydro/diffusion_core.wgsl), from the state at the
            # start of the step. Normal derivatives are differences of the face's two cells;
            # transverse ones the mean of the centered differences in the two cells.
            nu, kap = Cf[:, 2], Cf[:, 3]
            rr, pr_, sr_ = _shift(rho, 1, dim), _shift(p, 1, dim), _shift(s, 1, dim)
            rho_f = torch.minimum(rho, rr)
            vn_c, vt_c = vel[nrm - 1], vel[trn - 1]
            dn_n = _shift(vn_c, 1, dim) - vn_c
            dn_t = _shift(vt_c, 1, dim) - vt_c
            tdim = -2 if dim == -1 else -1

            def across(a):
                c = 0.5 * (_shift(a, 1, tdim) - _shift(a, -1, tdim))
                return 0.5 * (c + _shift(c, 1, dim))

            dt_n, dt_t = across(vn_c), across(vt_c)
            div = dn_n + dt_t
            tau_n = (2.0 * dn_n - (2.0 / 3.0) * div) * nu * rho_f / dx
            tau_t = (dn_t + dt_n) * nu * rho_f / dx
            vfn = 0.5 * (vn_c + _shift(vn_c, 1, dim))
            vft = 0.5 * (vt_c + _shift(vt_c, 1, dim))
            d_n, d_t, d_e = -tau_n, -tau_t, -(tau_n * vfn + tau_t * vft)
            F[MX if nrm == 1 else MY] = F[MX if nrm == 1 else MY] + d_n
            F[MY if nrm == 1 else MX] = F[MY if nrm == 1 else MX] + d_t
            F[EN] = F[EN] + d_e
            # What the stress leaves as heat in each of the two cells (visc_heat_d).
            hl = d_e - (vn_c * d_n + vt_c * d_t)
            hr = d_e - (_shift(vn_c, 1, dim) * d_n + _shift(vt_c, 1, dim) * d_t)
            heat = heat - hl + _shift(hr, -1, dim)
            # Diffusivity: of the specific internal energy and of the scalar.
            e_sp, e_spr = p / ((g - 1.0) * rho), pr_ / ((g - 1.0) * rr)
            qh = -kap * rho_f * (e_spr - e_sp) / dx
            F[EN] = F[EN] + qh
            F[EI] = F[EI] + qh
            F[RS] = F[RS] - kap * rho_f * (sr_ - s) / dx

        Fs = torch.stack(F, dim=1)
        net = net + _shift(Fs, -1, dim) - Fs
        net_vn = net_vn + _shift(vn, -1, dim) - vn

    dtdx = dt / dx
    rho_n = (U[:, RHO] + dtdx * net[:, RHO]).clamp_min(cfg.rho_floor)
    mx = U[:, MX] + dtdx * net[:, MX]
    my = U[:, MY] + dtdx * net[:, MY]
    E = U[:, EN] + dtdx * net[:, EN]
    ei = U[:, EI] + dtdx * (net[:, EI] + heat + P[:, 3].clamp_min(cfg.p_floor) * net_vn)
    rs = U[:, RS] + dtdx * net[:, RS]
    # Dual energy (hydro/dual_energy.wgsl).
    ke = 0.5 * (mx * mx + my * my) / rho_n
    eth = E - ke
    emin = cfg.p_floor / (g - 1.0)
    use = (eth > cfg.eta_dual * E) & (eth >= emin)
    ei_adv = torch.where(ei >= emin, ei, torch.full_like(ei, emin))
    ei_n = torch.where(use, eth, ei_adv)
    E_n = torch.where(use, E, ke + ei_adv)
    return torch.stack([rho_n, mx, my, E_n, ei_n, rs], dim=1)


def advance(U, cfg, t_span, closure=None, max_steps=100000):
    """Advance every state of the batch by t_span (a number or [B]), the last step shortened to
    land on it exactly, as the simulator lands on its output times. A state that has arrived
    takes steps of zero length while the others finish. Returns (U, steps taken, the most of
    any state)."""
    span = torch.as_tensor(t_span, dtype=U.dtype).expand(U.shape[0]).clone()
    t = torch.zeros_like(span)
    steps = 0
    while steps < max_steps:
        left = span - t
        left = torch.where(left <= 1e-6 * span.abs(), torch.zeros_like(left), left)
        if not bool((left > 0).any()):
            break
        dt = timestep(U, cfg, left)
        U = step(U, cfg, dt, closure)
        t = t + dt
        steps += 1
    return U, steps
