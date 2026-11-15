"""The electron-positron gas exactly: any degeneracy, any relativity, pairs included (no Coulomb
corrections), at a net electron density n = rho Ye / m_u and temperature T. The reference for
the network that stands in for the Helmholtz equation of state's electrons (fit.py), and for
the burner's heat capacity (tools/burn/burner.py).

What the network learns is the dimensionless thermal free energy

    g(n, T) = F_th / (n k T),   F_th = F(n, T) - F(n, 0)

with F the free energy density (rest masses included) and F(n, 0) the zero-temperature gas
that Shaderous already has exactly (eos/degenerate.wgsl). Everything thermal follows from g:

    P_th = n k T dg/dln n,   E_th = -n k T dg/dln T,   S = -n k (g + dg/dln T)

so the two derivatives are returned too, as the training targets beside g.

Electrons have chemical potential mu (rest mass included), positrons -mu. Each species is a
Fermi-Dirac integral over momentum x = p / m_e c; the net density fixes mu. Where the gas is
so degenerate that F - F(n, 0) would lose every digit (eta = (mu - m_e c^2) / kT > ETA_SOMMERFELD),
the Sommerfeld expansion is used instead: F_th = -(pi^2 / 6) (kT)^2 D, D = dn/dmu at the Fermi
surface, exact to order (kT / E_F)^2 there.

    tools/burn/.venv/Scripts/python tools/eos/electron_gas.py   (checks against known limits)
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

K = 1.380649e-16
ME = 9.1093837015e-28
C = 2.99792458e10
H = 6.62607015e-27
MU = 1.66053906660e-24
ARAD = 7.565723e-15
MEC2 = ME * C * C
LC3 = (H / (ME * C)) ** 3          # Compton wavelength cubed
PREF = 8.0 * np.pi / LC3           # states per unit volume per unit x^3 / 3, both spins
ETA_SOMMERFELD = 300.0


def _fd(eta, beta, kind):
    """One species' integral over x of x^2 f (kind 0: number), x^2 (gamma - 1) f (1: kinetic
    energy) or x^4 / (3 gamma) f (2: pressure), f = 1 / (exp((gamma - 1) / beta - eta) + 1).
    In units of PREF (number) or PREF m_e c^2 (energy, pressure)."""
    def f(x):
        g = np.sqrt(1.0 + x * x)
        arg = (g - 1.0) / beta - eta
        occ = np.exp(-arg) / (1.0 + np.exp(-arg)) if arg > 0 else 1.0 / (1.0 + np.exp(arg))
        if kind == 0:
            return x * x * occ
        if kind == 1:
            return x * x * (x * x / (g + 1.0)) * occ
        return x ** 4 / (3.0 * g) * occ
    # The occupation drops at the Fermi surface, x_F with gamma - 1 = eta beta, over a width of
    # about beta / (x_F / gamma_F) in x: split there, and stop where it is e^-60 of its peak.
    ek = max(eta * beta, 0.0)
    xf = np.sqrt(ek * (ek + 2.0))
    w = 60.0 * beta
    xmax = np.sqrt((ek + w) * (ek + w + 2.0))
    pts = [p for p in (xf,) if 0 < p < xmax]
    return quad(f, 0.0, xmax, points=pts or None, limit=500, epsabs=0.0, epsrel=1e-13)[0]


def _x_fermi(n):
    return (3.0 * n / PREF) ** (1.0 / 3.0)


def cold(n):
    """Zero-temperature gas at net density n: (mu, P, E) with E the kinetic energy density."""
    x = _x_fermi(n)
    g = np.sqrt(1.0 + x * x)
    f = x * (2.0 * x * x - 3.0) * g + 3.0 * np.arcsinh(x)
    if x < 1e-3:
        P = PREF * MEC2 * x ** 5 / 15.0
        E = PREF * MEC2 * x ** 5 / 10.0
    else:
        P = PREF * MEC2 * f / 24.0
        E = PREF * MEC2 * (x ** 3 * (g - 1.0) / 3.0 - f / 24.0)
    return MEC2 * g, P, E


def state(n, T):
    """(g, dg/dln n, dg/dln T, eta, positrons per net electron) at net density n and temperature T."""
    beta = K * T / MEC2
    kT = K * T
    # Sommerfeld where very degenerate: F_th = -(pi^2/6) (kT)^2 D(mu_F), D = dn/dmu.
    x = _x_fermi(n)
    gF = np.sqrt(1.0 + x * x)
    eta_cold = (gF - 1.0) / beta
    if eta_cold > ETA_SOMMERFELD:
        D = PREF * x * gF / MEC2
        q = -(np.pi ** 2 / 6.0) * kT * D / n          # g = F_th / (n kT) = q
        # d ln(D / n) / d ln n: D/n ~ gF / x^2, x ~ n^(1/3), d gF / d ln n = x^2 / (3 gF)
        dln = (x * x / (3.0 * gF * gF)) - 2.0 / 3.0
        return q, q * dln, q, eta_cold, 0.0
    # Solve for eta: n_- - n_+ = n, positrons at eta_+ = -eta - 2 / beta.
    net = lambda eta: PREF * (_fd(eta, beta, 0) - _fd(-eta - 2.0 / beta, beta, 0)) - n
    lo, hi = -50.0, max(eta_cold, 0.0) + 50.0
    while net(lo) > 0:
        lo -= 50.0
    while net(hi) < 0:
        hi *= 2.0
    eta = brentq(net, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)
    ep = -eta - 2.0 / beta
    nm, npl = PREF * _fd(eta, beta, 0), PREF * _fd(ep, beta, 0)
    P = PREF * MEC2 * (_fd(eta, beta, 2) + _fd(ep, beta, 2))
    # Energy with rest masses, less the net electrons' rest mass (kept out of F and F(n, 0) alike).
    E = PREF * MEC2 * (_fd(eta, beta, 1) + _fd(ep, beta, 1)) + 2.0 * MEC2 * npl
    mu = MEC2 + eta * kT                               # electron chemical potential, rest mass included
    F = (mu - MEC2) * n - P                            # F = mu n - P, net rest mass taken out
    mu0, P0, E0 = cold(n)
    F0 = (mu0 - MEC2) * n - P0
    Fth = F - F0
    Eth = E - E0
    Pth = P - P0
    s = n * kT
    return Fth / s, Pth / s, -Eth / s, eta, npl / n


def pair(T):
    """The electron-positron gas at zero net charge (mu = 0): its pressure P_p(T), and
    dP_p / dln T = E_p + P_p (E_p its energy density, rest masses included; F = -P_p, S = dP_p/dT).
    It holds nearly all of the pairs, and depends on T alone, so the network's target leaves it
    out: F_th = -P_p(T) + n k T h(n, T), with h of order one everywhere."""
    beta = K * T / MEC2
    eta = -1.0 / beta
    P = 2.0 * PREF * MEC2 * _fd(eta, beta, 2)
    E = 2.0 * PREF * MEC2 * (_fd(eta, beta, 1) + _fd(eta, beta, 0))
    return P, E + P


def target(n, T):
    """(h, dh/dln n, dh/dln T, eta, positrons per net electron): what the network learns."""
    g, gn, gT, eta, pf = state(n, T)
    Pp, dPp = pair(T)
    s = n * K * T
    return g + Pp / s, gn - Pp / s, gT + (dPp - Pp) / s, eta, pf


def _check():
    # Classical, nonrelativistic: g = ln(n lambda^3 / 2) - 1, P = n k T, E = 3/2 n k T.
    n, T = 1e20, 1e6
    lam = H / np.sqrt(2 * np.pi * ME * K * T)
    g, gn, gT, eta, pf = state(n, T)
    print(f"classical    g {g:.6f} (ideal {np.log(n * lam**3 / 2) - 1:.6f})  P/nkT {gn:.6f} (1)  E/nkT {-gT:.6f} (1.5)")
    # Classical, ultrarelativistic: P = n k T, E = 3 n k T (plus pairs, so only roughly).
    g, gn, gT, eta, pf = state(1e25, 3e11)
    print(f"ultrarel.    P/nkT {gn:.4f}  E/nkT {-gT:.4f}  (3 without pairs; positrons/net {pf:.2e})")
    # Degenerate: the expansion and the integrals agree where both apply.
    for n in (3e30, 3e31):
        x = _x_fermi(n)
        T = MEC2 * (np.sqrt(1 + x * x) - 1) / K / 150.0       # eta about 150: integrals still exact
        a = state(n, T)
        D = PREF * x * np.sqrt(1 + x * x) / MEC2
        som = -(np.pi**2 / 6) * K * T * D / n
        print(f"degenerate   n {n:.0e} T {T:.2e}: g {a[0]:.8e}  Sommerfeld {som:.8e}  (rel {abs(a[0]/som-1):.1e}); E/F {a[2]/a[0]:.5f} (1)")
    # Thermodynamic consistency: dg/dln n and dg/dln T from finite differences of g.
    for n, T in ((3e30, 3e9), (3e28, 5e8), (3e30, 7e9)):
        h = 1e-4
        g0 = state(n, T)
        dn = (state(n * np.exp(h), T)[0] - state(n * np.exp(-h), T)[0]) / (2 * h)
        dT = (state(n, T * np.exp(h))[0] - state(n, T * np.exp(-h))[0]) / (2 * h)
        print(f"consistency  n {n:.0e} T {T:.0e}: dg/dln n {g0[1]:.8f} vs {dn:.8f};  dg/dln T {g0[2]:.8f} vs {dT:.8f}")


if __name__ == "__main__":
    _check()
